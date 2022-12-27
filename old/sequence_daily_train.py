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
from model import WeightedGraphModel
from dataset import LifePopDailyModule

# %%
class TrainModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['epochs'])
        self.embeddings = nn.ModuleDict()
        self.embedding_list = kwargs['adj_embedding_dict'].keys()
        self.node_cnt = kwargs['node_cnt']

        self.model = WeightedGraphModel(**kwargs)
        self.criterion = nn.MSELoss()
        self.lr = kwargs['lr']

    def forward(self, x, add_dict):
        y = self.model(x, add_dict)
        return y

    def training_step(self, batch, batch_idx):
        x, add_dict, y = batch
        y_latent, y_hat = self(x, add_dict)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, add_dict, y = batch
        y_latent, y_hat = self(x, add_dict)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

        length = x.shape[0]
        sse = torch.sum((y_hat - y)**2, dim=( 0, 2, 3 )) 

        return {'length': length, 'sse': sse}



    def validation_epoch_end(self, outputs):
        total_length = sum([x['length'] for x in outputs])
        
        total_sse = torch.stack([x['sse'] for x in outputs], dim=0).sum(dim=0)/total_length

        for i in range(len(total_sse)):
            self.log(f'mse_days_{i}', total_sse[i])


    def predict_step(self, batch, batch_idx):
        x_all, add_dict_all, y = batch
        print(f'{x_all.shape=}, {y.shape=}')
       
        # len(add_dict_all) - self.hparams['seq_len'] - 1
        loop_count = 90
        x_new = x_all[:, :self.hparams['seq_len'], :, :]

        y_hats = []
        for i in range(loop_count):
            add_dict = {k: v[:, i:i+self.hparams['seq_len']] for k, v in add_dict_all.items()}
            y_latent, y_hat = self(x_new, add_dict)

            x_new = torch.cat([x_new[:, 1:, :, :], y_hat], dim=1)

            y_hats.append(y_hat)

        y_hats = torch.cat(y_hats, dim=1)
        return {'y': y[:,:loop_count], 'y_pred': y_hats}

            
            
    
    def predict_epoch_end(self, outputs):
        y = torch.cat([x['y'] for x in outputs], dim=0)
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)

        return y, y_pred
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('TrainModule')
        # parser.add_argument('--batch_size', type=int, default=32)
        # parser.add_argument('--num_workers', type=int, default=4)
        # parser.add_argument('--num_epochs', type=int, default=10)
        parser.add_argument('--seed', type=int, default=42)
        return parent_parser

#%%
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser = TrainModule.add_model_specific_args(parser)
    # parser = LifePopDailyModule.add_model_specific_args(parser)

    args = {
        # Fixed
        'epochs': 1000,  # 에폭 수

        'node_cnt': 25,
        'node_dim': 28,

        # Hyperparameter
        'batch_size': 256,  # 배치 사이즈
        'lr': 0.001,  # 학습률

        'seq_len': 28,  # 시퀀스 길이
        'pred_len' : 1, # 예측 길이

        'adj_hidden_dim': 16,  # 인접 행렬 임베딩 차원
        'adj_num_layers': 4,  # 인접 행렬 임베딩 레이어 수
        'adj_embedding_dim': 8,
        'adj_embedding_dict': {'month': 12, 'wday': 7},


        'node_latent_dim': 16,
        'node_hidden_dim': 32,
        'node_num_layers': 4,
    }

    seed_everything(42)

    data = np.load('data/LOCAL_PEOPLE_GU_DAILY_2020_2021.npy')/10000
    # add_dict = {'wday': ([4, 5, 6, 0, 1, 2, 3]*53)[:365],
    #             'month': np.repeat(list(range(12)), np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
    #             }
    # 2년치
    add_dict = {'wday': ([2, 3, 4, 5, 6, 0, 1]*120)[:731],
                'month': 
                    np.concatenate([
                        np.repeat(list(range(12)), np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])),
                        np.repeat(list(range(12)), np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
                    ], axis=0)
                }

    add_data = pd.DataFrame(add_dict)  # 추가 데이터

    data_module = LifePopDailyModule(data, add_data=add_data, pred_len=args['pred_len'],
                                      validation_rate=0.2, seq_len=args['seq_len'],
                                      batch_size=args['batch_size'], num_workers=4)

    logger = TensorBoardLogger('lightning_logs', name='seq_daily_y2')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # model = TrainModule(**args)
    model = TrainModule.load_from_checkpoint(
            'lightning_logs/seq_daily_y2/version_0/checkpoints/epoch=999-step=3000.ckpt', **args)

    trainer = Trainer(max_epochs=args['epochs'],
                      logger=logger,
                      callbacks=[lr_monitor],
                      accelerator='gpu', devices=1)
    # trainer.fit(model, data_module)

    output = trainer.predict(model, datamodule=data_module)


#%%
    np.save('y.npy', output[0]['y'][0].detach().numpy())
    np.save('y_pred.npy', output[0]['y_pred'][0].detach().numpy())
