# %%
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch

import argparse

try:
    from dataset.old.moving_pop_dataset import MovingPopDailyDataset
except:
    from dataset.old.moving_pop_dataset import MovingPopDailyDataset


class AuxSpatialDataset(Dataset):
    '''
    AuxSpatialDataset
        Dataset for using additional data for each region
    '''

    def __init__(self, spatial_data, columns, cnt_list):
        '''
        Args:
            spatial_data: DataFrame 
            columns: list of using columns (categorical columns should be 0 ~ n-1)
            cnt_list: list of column type (1: continuous, 2~: categorical cardinality)
        '''
        super().__init__()
        self.data = {}
        for i, k in enumerate(columns):
            dtype = torch.float32 if cnt_list[i] == 1 else torch.long
            self.data[k] = torch.tensor(spatial_data[k].values, dtype=dtype)

    def __len__(self):
        return 100000000

    def __getitem__(self, idx):
        '''
        Args:
            idx : index of time, so spatial data is not changed and always return same value.
        '''
        return None

    def collate(self, batch):
        '''
        Args:
            batch : batch is list of None which is same length with Batch
        Returns:
            output : dict of tensor (batch, node_cnt, embedding_dim)
        '''
        output = {}
        for k, v in self.data.items():
            output[k] = v.expand(len(batch), -1)
        return output


class AuxTemporalDataset(Dataset):
    '''
    AuxTemporalDataset
        Dataset for using additional data for each time
    '''

    def __init__(self, temporal_data, columns, cnt_list, seq_len):
        '''
        Args:
            temporal_data: DataFrame
            columns: list of using columns (categorical columns should be 0 ~ n-1)
            cnt_list: list of column type (1: continuous, 2~: categorical cardinality)
            seq_len: length of input sequence
        '''
        super().__init__()
        self.len = temporal_data.shape[0] - seq_len
        self.seq_len = seq_len
        self.data = {}
        for i, k in enumerate(columns):
            dtype = torch.float32 if cnt_list[i] == 1 else torch.long
            self.data[k] = torch.tensor(temporal_data[k].values, dtype=dtype)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        output = {}
        for k, v in self.data.items():
            output[k] = v[idx:idx+self.seq_len]
        return output

    def collate(self, batch):
        output = {}
        for k, v in self.data.items():
            output[k] = torch.stack([x[k] for x in batch], dim=0)
        return output


class MovingPopWithAuxDataModule(pl.LightningDataModule):
    def __init__(self, mp_data, covid_data,
                 temporal_data, temporal_columns, temporal_cnt_list,
                 spatial_data, spatial_columns, spatial_cnt_list,
                 seq_len, num_node, validation_rate, batch_size, num_workers=4):
        '''
        Args:
            mp_data: Moving Population Data (time, region, region, channel)
            covid_data: local confirmed case data (time, region, channel)
            temporal_data: DataFrame (seq, channel)
            temporal_columns: list of using columns
            temporal_cnt_list: list of cardinality of each column
            spatial_data: DataFrame (node, channel)
            spatial_columns: list of using columns
            spatial_cnt_list: list of cardinality of each column
            seq_len: input sequence length
            num_node: number of nodes
            validation_rate: validation rate
            batch_size: batch size
            num_workers: number of workers
        '''

        super().__init__()
        self.data = mp_data
        self.covid_data = covid_data
        self.temporal_data = temporal_data
        self.temporal_columns = temporal_columns
        self.temporal_cnt_list = temporal_cnt_list
        self.spatial_data = spatial_data
        self.spatial_columns = spatial_columns
        self.spatial_cnt_list = spatial_cnt_list
        self.batch_size = batch_size
        self.num_node = num_node
        self.seq_len = seq_len
        self.num_workers = num_workers
        ###############
        self.pred_len = 1

        length = self.temporal_data.shape[0]
        self.split = int(length * (1 - validation_rate))

    def setup(self, stage=None):
        '''
        Assign train/val datasets for use in dataloaders
        '''
        if stage == 'fit':
            # Train Data
            self.train_dataset = MovingPopDailyDataset(self.data[:self.split + self.pred_len],
                                                       covid_data=self.covid_data[:self.split +
                                                                                  self.pred_len],
                                                       seq_len=self.seq_len,
                                                       pred_len=self.pred_len)
            self.train_spatial = AuxSpatialDataset(
                self.spatial_data, self.spatial_columns, self.spatial_cnt_list)
            self.train_temporal = AuxTemporalDataset(self.temporal_data[:self.split + self.pred_len],
                                                     self.temporal_columns, self.temporal_cnt_list,
                                                     seq_len=self.seq_len)
            # Validation Data
            self.val_dataset = MovingPopDailyDataset(self.data[self.split:],
                                                     covid_data=self.covid_data[self.split:],
                                                     seq_len=self.seq_len, pred_len=self.pred_len)
            self.val_spatial = AuxSpatialDataset(
                self.spatial_data, self.spatial_columns, self.spatial_cnt_list)
            self.val_temporal = AuxTemporalDataset(self.temporal_data[self.split:],
                                                   self.temporal_columns, self.temporal_cnt_list,
                                                   seq_len=self.seq_len)

        if stage == 'test':
            self.test_dataset = MovingPopDailyDataset(self.data[self.split:], covid_data=self.covid_data[self.split:],
                                                      seq_len=self.seq_len, pred_len=self.pred_len)
            self.test_spatial = AuxSpatialDataset(
                self.spatial_data, self.spatial_columns, self.spatial_cnt_list)
            self.test_temporal = AuxTemporalDataset(self.temporal_data[self.split:],
                                                    self.temporal_columns, self.temporal_cnt_list,
                                                    seq_len=self.seq_len)

    def train_dataloader(self):
        return [
            DataLoader(self.train_dataset,
                       batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            DataLoader(self.train_spatial, batch_size=self.batch_size,
                       collate_fn=self.train_spatial.collate),
            DataLoader(self.train_temporal, batch_size=self.batch_size,
                       collate_fn=self.train_temporal.collate)
        ]

    def val_dataloader(self):
        return [
            DataLoader(self.val_dataset,
                       batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.val_spatial, batch_size=self.batch_size,
                       collate_fn=self.val_spatial.collate),
            DataLoader(self.val_temporal, batch_size=self.batch_size,
                       collate_fn=self.val_temporal.collate)
        ]

    def test_dataloader(self):
        return [
            DataLoader(self.test_dataset,
                       batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
            DataLoader(self.test_spatial, batch_size=self.batch_size,
                       collate_fn=self.test_spatial.collate),
            DataLoader(self.test_temporal, batch_size=self.batch_size,
                       collate_fn=self.test_temporal.collate)
        ]

 # %%
if __name__ == '__main__':
    args = {
        # Fixed
        'epochs': 500,  # 에폭 수
        'batch_size': 3,
        'lr': 0.001,

        # Hyperparameter
        # Adjacency Matrix Hyperparameter
        'adj_hidden_dim': 8,
        'adj_num_layers': 2,
        'adj_embedding_dim': 2,
        'adj_channel': 1,
        'adj_input_dim': 20,
        'adj_output_dim': 2,

        # Graph Module Hyperparameter
        'node_dim': 28,
        'node_cnt': 25,
        'node_latent_dim': 16,
        'node_hidden_dim': 32,
        'node_num_layers': 3,

        # Train & Predict Sequence Length
        'seq_len': 7,
        'pred_len': 1,

        # Auxiliary Data Information
        'aux_temporal_embedding_dim': 4,
        'aux_temporal_columns': [
            'day',
            'holiday',
            'temp',
        ],
        'aux_temporal_cardinalities': [7, 2, 1],

        'aux_spatial_embedding_dim': 4,
        'aux_spatial_columns': [
            'cnt_worker_male_2019',
            'cnt_worker_female_2019',
            'culture_cnt_2020',
            'physical_facil_2019',
            'school_cnt_2020',
            'student_cnt_2020',
        ],
        'aux_spatial_cardinalities': [1, 1, 1, 1, 1, 1],
    }

    import os
    import pandas as pd

    os.listdir('../data/')
    spatial = pd.read_csv('../data/df_region_normalized.csv',
                          index_col=0, encoding='cp949')
    temporal = pd.read_csv('../data/df_time_day_normalized.csv',
                           index_col=0, encoding='cp949')

    a = AuxSpatialDataset(
        spatial, args['aux_spatial_columns'], args['aux_spatial_cardinalities'])
    b = DataLoader(a, batch_size=args['batch_size'], collate_fn=a.collate)

    for i in b:
        print(i.keys())
        break

    a = AuxTemporalDataset(
        temporal, args['aux_temporal_columns'], args['aux_temporal_cardinalities'], args['seq_len'])
    b = DataLoader(a, batch_size=args['batch_size'], collate_fn=a.collate)

    for i in b:
        print(i.keys())
        break

    mp_data = torch.randn((1005,
                           args['adj_input_dim'],
                           args['node_cnt'],
                           args['node_cnt']))

    covid_data = torch.randn(1005,
                             args['node_cnt'],
                             args['node_dim'])

    dataset = MovingPopWithAuxDataModule(
        mp_data=mp_data,
        covid_data=covid_data,
        temporal_data=temporal,
        temporal_columns=args['aux_temporal_columns'],
        temporal_cnt_list=args['aux_temporal_cardinalities'],
        spatial_data=spatial,
        spatial_columns=args['aux_spatial_columns'],
        spatial_cnt_list=args['aux_spatial_cardinalities'],
        batch_size=args['batch_size'],
        num_node=args['node_cnt'],
        seq_len=args['seq_len'],
        validation_rate=0.2,
        num_workers=4,
    )
    dataset.prepare_data()
    dataset.setup(stage='fit')
    dataset.setup(stage='test')
    a = dataset.train_dataloader()

