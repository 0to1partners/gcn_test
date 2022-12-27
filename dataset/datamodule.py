# %%
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# custom datasets
try:
    from datasets import MovingPopDailyDataset, AuxTemporalDataset, AuxSpatialDataset
except:
    from dataset.datasets import MovingPopDailyDataset, AuxTemporalDataset, AuxSpatialDataset


class MovingPopWithAuxDataModule(pl.LightningDataModule):
    '''
    MovingPopWithAuxDataModule
        DataModule for using additional data
        Spatial and Temporal data are used as additional data
    '''

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

        self.total_length = self.data.shape[0]
        self.split = int( self.total_length * (1 - validation_rate))
        self.train_length = self.split - self.seq_len
        self.val_length = self.total_length - self.split - self.seq_len - 1
        
    def setup(self, stage=None):
        '''
        Assign train/val datasets for use in dataloaders
        '''
        if stage == 'fit' :
            self.train_dataset = MovingPopDailyDataset(self.data[:self.split + 1],
                                                       covid_data=self.covid_data[:self.split + 1],
                                                       seq_len=self.seq_len,
                                                       pred_len=self.pred_len)
            self.train_spatial = AuxSpatialDataset(self.spatial_data, self.spatial_columns,
                                                   self.spatial_cnt_list,
                                                   length=self.train_length)
            self.train_temporal = AuxTemporalDataset(self.temporal_data[:self.split + self.pred_len],
                                                     self.temporal_columns, self.temporal_cnt_list,
                                                     seq_len=self.seq_len,
                                                     length=self.train_length)

            self.val_dataset = MovingPopDailyDataset(self.data[self.split:self.total_length],
                                                     covid_data=self.covid_data[self.split:self.total_length],
                                                     seq_len=self.seq_len, pred_len=self.pred_len)
            self.val_spatial = AuxSpatialDataset(self.spatial_data, self.spatial_columns,
                                                 self.spatial_cnt_list,
                                                 length=self.val_length)
            self.val_temporal = AuxTemporalDataset(self.temporal_data[self.split:self.total_length],
                                                   self.temporal_columns, self.temporal_cnt_list,
                                                   seq_len=self.seq_len,
                                                   length=self.val_length)
        elif stage == 'test' :
            self.test_dataset = MovingPopDailyDataset(self.data[self.split:self.total_length],
                                                     covid_data=self.covid_data[self.split:self.total_length],
                                                     seq_len=self.seq_len, pred_len=self.pred_len)
            self.test_spatial = AuxSpatialDataset(self.spatial_data, self.spatial_columns,
                                                 self.spatial_cnt_list,
                                                 length=self.val_length)
            self.test_temporal = AuxTemporalDataset(self.temporal_data[self.split:self.total_length],
                                                   self.temporal_columns, self.temporal_cnt_list,
                                                   seq_len=self.seq_len,
                                                   length=self.val_length)
        else : 
            raise ValueError('Invaild Stage : {stage}')



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
            DataLoader(self.train_dataset,
                       batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
            DataLoader(self.train_spatial, batch_size=self.batch_size,
                       collate_fn=self.train_spatial.collate),
            DataLoader(self.train_temporal, batch_size=self.batch_size,
                       collate_fn=self.train_temporal.collate)
        ]
    # def val_dataloader(self):
    #     return [
    #         DataLoader(self.val_dataset,
    #                    batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers),
    #         DataLoader(self.val_spatial, batch_size=self.batch_size,
    #                    collate_fn=self.val_spatial.collate),
    #         DataLoader(self.val_temporal, batch_size=self.batch_size,
    #                    collate_fn=self.val_temporal.collate)
    #     ]
        
class MovingPopDailyModule(pl.LightningDataModule):
    '''
    MovingPopDailyModule
        DataModule for using only Moving Population Data
    '''
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
            self.train_dataset = MovingPopDailyDataset(self.data[:self.split + self.pred_len],
                                                       covid_data=self.covid_data[:self.split +
                                                                                  self.pred_len],
                                                       seq_len=self.seq_len,
                                                       pred_len=self.pred_len)

            self.val_dataset = MovingPopDailyDataset(self.data[self.split:],
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

    # @ staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = argparse.ArgumentParser(
    #         parents=[parent_parser], add_help=False)
    #     parser.add_argument('--batch_size', type=int, default=256)
    #     parser.add_argument('--num_workers', type=int, default=4)
    #     parser.add_argument('--validation_rate', type=float, default=0.1)
    #     return parser

# %%
if __name__ == '__main__':
    args = {
        # Fixed
        'epochs': 500,  # 에폭 수
        'batch_size': 5,
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
    b = dataset.val_dataloader()
    # c = dataset.test_dataloader()

    print(len(a[0]), len(a[1]), len(a[2]))
    print(len(b[0]), len(b[1]), len(b[2]))
    # print(len(c[0]), len(c[1]), len(c[2]))
# %%
