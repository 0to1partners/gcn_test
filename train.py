# %%
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

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

from model.gcn_baseline import WeightedGraphModel
from gcn_dataloader import LifePopDailyModule
import sys


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

    def forward(self, x, add_dict):
        y = self.model(x ,add_dict)
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
        return loss
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


if __name__ == '__main__':

    kwargs = {
        'node_cnt': 100,
        'hidden_channels': 32,
        'num_layers': 3,
        'batch_size': 32,
        'lr': 0.001,

        'adj_hidden_dim': 128,
        'adj_num_layers': 2,
        'adj_embedding_dim': 8,
        'adj_embedding_dict': {'month': 12, 'wday': 7, 'hour': 24},

        'node_dim': 28,
        'node_cnt': 25,
        'node_latent_dim': 16,
        'node_hidden_dim': 128,
        'node_num_layers': 2,

        'num_graph_layers': 1
    }

    data = np.load('data/LOCAL_PEOPLE_GU_2021.npy')
    add_dict = {'hour': np.random.randint(0, 24, size=(len(data))),
                'wday': np.random.randint(0, 7, size=(len(data))),
                'month': np.random.randint(0, 12, size=(len(data))), }
    add_data = pd.DataFrame(add_dict)

    data_module = LifePopDailyModule(data, add_data = add_data, validation_rate=0.2,
                                     batch_size=32, num_workers=4)

    model = TrainModule(**kwargs)

    trainer = Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, data_module)


# %%
