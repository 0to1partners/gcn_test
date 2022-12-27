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



# addition of the path to the data/model folder
from model import PopulationWeightedGraphModel
from dataset import MovingPopDailyModule


class TrainModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['epochs'])
        self.node_cnt = kwargs['node_cnt']

        self.model = PopulationWeightedGraphModel(**kwargs)
        self.criterion = nn.MSELoss()
        self.lr = kwargs['lr']

    def forward(self, x, covid):
        y = self.model(x, covid)
        return y

    def training_step(self, batch, batch_idx):
        adj, x, y = batch
        _, y_hat = self(x, adj)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss)

        length = x.shape[0]
        sse = torch.sum((y_hat - y)**2, dim=(0, 2, 3))

        return {'loss' : loss, 'length': length, 'sse': sse}

    def validation_step(self, batch, batch_idx):
        adj, x, y = batch
        _, y_hat = self(x, adj)
        loss = self.criterion(y_hat, y)

        self.log('val_loss', loss)
        length = x.shape[0]
        sse = torch.sum((y_hat - y)**2, dim=(0, 2, 3))

        return {'length': length, 'sse': sse}

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
        parser = parent_parser.add_argument_group('TrainModule')
        # parser.add_argument('--batch_size', type=int, default=32)
        # parser.add_argument('--num_workers', type=int, default=4)
        # parser.add_argument('--num_epochs', type=int, default=10)
        parser.add_argument('--seed', type=int, default=42)
        return parent_parser



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser = TrainModule.add_model_specific_args(parser)
    # parser = LifePopDailyModule.add_model_specific_args(parser)

    args = {
        # Fixed
        'epochs': 500,  # 에폭 수
        'batch_size': 3,
        'lr' : 0.001,

        # Hyperparameter
        'adj_hidden_dim': 8,
        'adj_num_layers': 2,
        'adj_embedding_dim': 2,
        'adj_channel': 1,

        'adj_input_dim': 20,
        'adj_output_dim': 2,

        'node_dim': 28,
        'node_cnt': 25,
        'node_latent_dim': 16,
        'node_hidden_dim': 32,
        'node_num_layers': 3,

        'seq_len': 24,
        'pred_len': 1,
    }
    seed_everything(42)

    # Moking data
    mp_data = np.random.randn(
        1005, args['adj_input_dim'], args['node_cnt'], args['node_cnt'])
    covid_data = np.random.randn(1005, args['node_cnt'], args['node_dim'])

    data_module = MovingPopDailyModule(mp_data,
                                       covid_data=covid_data, pred_len=args['pred_len'],
                                       validation_rate=0.2, seq_len=args['seq_len'],
                                       batch_size=args['batch_size'], num_workers=4)

    logger = TensorBoardLogger('lightning_logs', name='moving_pop')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = TrainModule(**args)

    trainer = Trainer(max_epochs=args['epochs'],
                      logger=logger,
                      callbacks=[lr_monitor],
                      accelerator='gpu', devices=1)
    trainer.fit(model, data_module)

# %%
