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
from model.gcn_baseline import WeightedGraphModel
from dataset.pop_hourly import LifePopHourlyModule


# %%


class TrainModule(pl.LightningModule): 
    def __init__(self, **kwargs): 
        super().__init__() 
        self.save_hyperparameters() 
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
        x, add_dict, y = batch # 
        y_latent, y_hat = self(x, add_dict)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, add_dict, y = batch
        y_latent, y_hat = self(x, add_dict)
        loss = self.criterion(y_hat, y)

        self.log('val_loss', loss)
        return loss

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


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser = TrainModule.add_model_specific_args(parser)
    # parser = LifePopDailyModule.add_model_specific_args(parser)

    args = {
        'epochs': 20, # 에폭 수
        'batch_size': 128, # 배치 사이즈
        'lr': 0.001, # 학습률

        'adj_hidden_dim': 64,  # 인접 행렬 임베딩 차원
        'adj_num_layers': 2, # 인접 행렬 임베딩 레이어 수
        'adj_embedding_dim': 8,
        'adj_embedding_dict': {'month': 12, 'wday': 7, 'hour': 24},

        'node_cnt': 25,
        'node_dim': 28,
        'node_latent_dim': 16,
        'node_hidden_dim': 64,
        'node_num_layers': 2,

        'num_graph_layers': 2
    }

    seed_everything(42)

    data = np.load('data/LOCAL_PEOPLE_GU_2021.npy')/10000
    add_dict = {'hour': list(range(24))*365,
                'wday': np.repeat([4, 5, 6, 0, 1, 2, 3]*53, 24)[:8760],
                'month': np.repeat(list(range(12)), np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])*24)
                }
    add_data = pd.DataFrame(add_dict) # 추가 데이터
    
    data_module = LifePopHourlyModule(data, add_data=add_data, validation_rate=0.2,
                                     batch_size=args['batch_size'], num_workers=4)

    logger = TensorBoardLogger('lightning_logs', name='baseline')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    model = TrainModule(**args)

    trainer = Trainer(max_epochs=args['epochs'],
                      logger=logger,
                      callbacks=[lr_monitor],
                      accelerator='gpu', devices=1)
    trainer.fit(model, data_module)


# %%