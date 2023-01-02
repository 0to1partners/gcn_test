# %%
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn.dense import DenseGraphConv
from torch_geometric.nn.norm import LayerNorm, BatchNorm

from torchmetrics import MeanSquaredError

from tqdm.auto import tqdm
import pandas as pd

import sys
import argparse
import os 

# addition of the path to the data/model folder
from model import PopulationWeightedGraphModel, AdditionalInfo
# from dataset import MovingPopDailyModule
from dataset import MovingPopWithAuxDataModule


class TrainModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['epochs','temporal_cardinalities','spatial_cardinalities'])
        self.node_cnt = kwargs['node_cnt']

        self.aux_temporal = AdditionalInfo(kwargs['temporal_columns'], kwargs['temporal_cardinalities'], kwargs['temporal_embedding_dim'])
        self.aux_spatial = AdditionalInfo(kwargs['spatial_columns'], kwargs['spatial_cardinalities'], kwargs['spatial_embedding_dim'])
        self.model = PopulationWeightedGraphModel(**kwargs)
        self.criterion = nn.MSELoss()
        self.lr = kwargs['lr']

    def forward(self, x, covid):
        y = self.model(x, covid)
        return y

    def training_step(self, batch, batch_idx):
        adj, x, y, spatial, temporal = batch

        _, y_hat = self(x, adj)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss)

        length = x.shape[0]
        sse = torch.sum((y_hat - y)**2, dim=(0, 2, 3))
        return {'loss' : loss, 'length': length, 'sse': sse}


    def validation_step(self, batch, batch_idx):
        adj, x, y, spatial, temporal = batch
        
        _, y_hat = self(x, adj)
        loss = self.criterion(y_hat, y)

        self.log('val_loss', loss)
        length = x.shape[0]
        sse = torch.sum((y_hat - y)**2, dim=(0, 2, 3))

        return {'length': length, 'sse': sse}
    

    def test_step(self, batch, batch_idx):
        adj, x, y, spatial, temporal = batch

        result = []
        b = x.shape[0]
        x = x[0:1]

        for i in range(b):
            _, y_hat = self(x, adj[i:i+1])
            result.append(y_hat)
            x = torch.cat([x[:,1:], y_hat], dim=1)

        y_hat = torch.cat(result, dim=0)
        sse = torch.sum((y_hat - y)**2, dim=(0, 2, 3))
        self.log('test_sse', sse)


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


    def validation_epoch_end(self, outputs):
        total_length = sum([x['length'] for x in outputs])

        total_sse = torch.stack(
            [x['sse'] for x in outputs], dim=0).sum(dim=0)/total_length

        for i in range(len(total_sse)):
            self.log(f'mse_days_{i}', total_sse[i])


    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ModelArgs')
        parser.add_argument('--adj_hidden_dim', type=int, default=4)
        parser.add_argument('--adj_num_layers', type=int, default=2)
        parser.add_argument('--adj_embedding_dim', type=int, default=2)
        parser.add_argument('--adj_channel', type=int, default=1)
        parser.add_argument('--adj_input_dim', type=int, default=20)
        parser.add_argument('--adj_output_dim', type=int, default=2)

        parser.add_argument('--node_dim', type=int, default=28)
        parser.add_argument('--node_cnt', type=int, default=25)
        parser.add_argument('--node_latent_dim', type=int, default=16)
        parser.add_argument('--node_hidden_dim', type=int, default=32)
        parser.add_argument('--node_num_layers', type=int, default=3)

        parser.add_argument('--seq_len', type=int, default=24)
        parser.add_argument('--pred_len', type=int, default=1)

        parser.add_argument('--temporal_embedding_dim', type=int, default=4)
        parser.add_argument('--temporal_columns', type=list, default=[
            'day',
            'holiday',
            'temp',
            ])
        parser.add_argument('--temporal_cardinalities', type=list, default=[7,2,1])
        
        parser.add_argument('--spatial_embedding_dim', type=int, default=4)
        parser.add_argument('--spatial_columns', type=list, default=[
            'cnt_worker_male_2019',
            'cnt_worker_female_2019',
            'culture_cnt_2020',
            'physical_facil_2019',
            'school_cnt_2020', 
            'student_cnt_2020',
            ])
        parser.add_argument('--spatial_cardinalities', type=list, default=[1,1,1,1,1,1])
        return parent_parser
    
    @staticmethod
    def add_train_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('TrainModule')
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--num_epochs','-e', type=int)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--lr', type=int, default=0.001)
        return parent_parser
    


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = TrainModule.add_model_specific_args(parser)
    parser = TrainModule.add_train_specific_args(parser)

    args = vars(parser.parse_args())
    
    seed_everything(args['seed'])

    # Moking data
    mp_data = np.random.randn(
        978, args['adj_input_dim'], args['node_cnt'], args['node_cnt'])
    covid_data = np.random.randn(978, args['node_cnt'], args['node_dim'])

    spatial = pd.read_csv('data/df_region_normalized.csv', index_col=0, encoding='cp949')
    temporal = pd.read_csv('data/df_time_day_normalized.csv', index_col=0, encoding='cp949')

    data_module = MovingPopWithAuxDataModule(
        mp_data = mp_data,
        covid_data = covid_data,
        temporal_data = temporal,
        temporal_columns = args['temporal_columns'],
        temporal_cardinalities = args['temporal_cardinalities'],
        spatial_data = spatial,
        spatial_columns = args['spatial_columns'],
        spatial_cardinalities = args['spatial_cardinalities'],
        batch_size = args['batch_size'],
        num_node = args['node_cnt'],
        seq_len = args['seq_len'],
        validation_rate = 0.2,
        num_workers = 4,
    )

    logger = TensorBoardLogger('lightning_logs', name='test')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = TrainModule(**args)

    trainer = Trainer(max_epochs=args['num_epochs'],
                      logger=logger,
                      callbacks=[lr_monitor],
                      accelerator='gpu', devices=1)
    
    trainer.fit(model, data_module)

    a = trainer.test(model, data_module)

# %%
