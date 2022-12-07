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
from model.sequence_model import WeightedGraphModel
from dataset.pop_seq_daily import LifePopDailyModule


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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


    def validation_epoch_end(self, outputs):
        total_length = sum([x['length'] for x in outputs])
        
        total_sse = torch.stack([x['sse'] for x in outputs], dim=0).sum(dim=0)/total_length

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


#%%

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser = TrainModule.add_model_specific_args(parser)
    # parser = LifePopDailyModule.add_model_specific_args(parser)

    args = {
        # Fixed
        'epochs': 500,  # 에폭 수

        'node_cnt': 25,
        'node_dim': 28,

        # Hyperparameter
        'batch_size': 256,  # 배치 사이즈
        'lr': 0.001,  # 학습률

        'seq_len': 28,  # 시퀀스 길이
        'pred_len' : 28, # 예측 길이

        'adj_hidden_dim': 16,  # 인접 행렬 임베딩 차원
        'adj_num_layers': 4,  # 인접 행렬 임베딩 레이어 수
        'adj_embedding_dim': 8,
        'adj_embedding_dict': {'month': 12, 'wday': 7},


        'node_latent_dim': 16,
        'node_hidden_dim': 32,
        'node_num_layers': 4,
    }

    seed_everything(42)

    data = np.load('data/LOCAL_PEOPLE_GU_DAILY_2021.npy')/10000
    add_dict = {'wday': ([4, 5, 6, 0, 1, 2, 3]*53)[:365],
                'month': np.repeat(list(range(12)), np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
                }

    add_data = pd.DataFrame(add_dict)  # 추가 데이터

    data_module = LifePopDailyModule(data, add_data=add_data, pred_len=args['pred_len'],
                                      validation_rate=0.2, seq_len=args['seq_len'],
                                      batch_size=args['batch_size'], num_workers=4)

    logger = TensorBoardLogger('lightning_logs', name='seq_daily')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = TrainModule(**args)

    trainer = Trainer(max_epochs=args['epochs'],
                      logger=logger,
                      callbacks=[lr_monitor],
                      accelerator='gpu', devices=1)
    trainer.fit(model, data_module)

# %%
