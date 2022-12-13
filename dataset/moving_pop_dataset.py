# %%
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import Data, Dataset
# from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import torch

import argparse

# %%


class MobingPopDailyDataset(Dataset):
    def __init__(self, mp_data, covid_data, seq_len, pred_len, is_pred=False, ):
        super().__init__()

        if mp_data.shape[0] != covid_data.shape[0]:
            raise Exception('Data length is not equal')
        if len(mp_data) < seq_len + 1:
            raise Exception('Data length is too short')

        self.mp_data = mp_data
        self.covid_data = covid_data
        self.is_pred = is_pred
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.mp_data) - self.seq_len - self.pred_len - 1

    def __getitem__(self, idx):
        '''
        Args:
            idx: index of the first element in the sequence
        Returns:
            x_adj : (batch, seq_len, node_cnt, node_cnt)
            x_node : (batch, seq_len, node_cnt, embedding_dim)
            y_node : (batch, node_cnt, embedding_dim)
        '''
        end = idx + self.seq_len
        return \
            torch.tensor(self.mp_data[idx+1: end+1], dtype=torch.float32), \
            torch.tensor(self.covid_data[idx:end], dtype=torch.float32), \
            torch.tensor(self.covid_data[end+1: end+self.pred_len+1], dtype=torch.float32)

    # def collate(self, batch):
    #     x, add_data, y = map(list, zip(*batch))
    #     x = torch.stack(x)

    #     y = torch.stack(y)
    #     add_data = torch.stack(add_data)
    #     add_data = torch.stack(add_data)

    #     # add_dict = {k: add_data[:, :, i]
    #                 # for i, k in enumerate(self.add_keys)}

    #     return x, add_dict, y


class MovingPopDailyModule(pl.LightningDataModule):
    def __init__(self, mp_data, covid_data, seq_len, pred_len, validation_rate, batch_size, num_workers=4):
        super().__init__()
        self.data = mp_data
        self.covid_data = covid_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.pred_len = pred_len

        length = len(self.data)
        self.split = int(length * (1 - validation_rate))

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = MobingPopDailyDataset(self.data[:self.split + self.pred_len],
                                                       covid_data=self.covid_data[:self.split + self.pred_len],
                                                       seq_len=self.seq_len,
                                                       pred_len=self.pred_len)

            self.val_dataset = MobingPopDailyDataset(self.data[self.split:],
                                                     covid_data=self.covid_data[self.split:],
                                                     seq_len=self.seq_len,
                                                     pred_len=self.pred_len)
        else:
            raise Exception('Not implemented')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=self.val_dataset.collate
        )

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--validation_rate', type=float, default=0.1)
        return parser


# %%
if __name__ == '__main__':
    '''
    Test dataloader
    '''
    hp = {
        'batch': 3,
        'pred_len': 1,

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

        # 'num_graph_layers': 3,
        'seq_len': 24,
        'pred_len': 1,
    }

    mp_data = torch.randn((1005,
                           hp['adj_input_dim'],
                           hp['node_cnt'],
                           hp['node_cnt']))

    covid_data = torch.randn(1005,
                            hp['node_cnt'],
                            hp['node_dim'])


    dm = MovingPopDailyModule(mp_data, covid_data, seq_len=24, pred_len=1,
                            validation_rate=0.2, batch_size=4)
    dm.setup(stage='fit')

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(len(train_loader))
    print(len(val_loader))

    for adj, x, y in train_loader:
        print(adj.shape)
        print(x.shape)
        print(y.shape)
        break


# %%
