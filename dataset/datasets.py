# %%
import torch
from torch.utils.data import Dataset


class AuxSpatialDataset(Dataset):
    '''
    AuxSpatialDataset
        Dataset for using additional data for each region
    '''

    def __init__(self, spatial_data, columns, cnt_list, length):
        '''
        Args:
            spatial_data: DataFrame 
            columns: list of using columns (categorical columns should be 0 ~ n-1)
            cnt_list: list of column type (1: continuous, 2~: categorical cardinality)
            length: length of dataset
        '''
        super().__init__()
        self.len = length
        self.data = {}
        for i, k in enumerate(columns):
            dtype = torch.float32 if cnt_list[i] == 1 else torch.long
            self.data[k] = torch.tensor(spatial_data[k].values, dtype=dtype)

    def __len__(self):
        return self.len

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
            output : dict of tensor {column : (batch, node_cnt, embedding_dim)}
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

    def __init__(self, temporal_data, columns, cnt_list, seq_len, length):
        '''
        Args:
            temporal_data: DataFrame
            columns: list of using columns (categorical columns should be 0 ~ n-1)
            cnt_list: list of column type (1: continuous, 2~: categorical cardinality)
            seq_len: length of input sequence
        '''
        super().__init__()
        # self.len = temporal_data.shape[0] - seq_len
        self.len = length
        self.seq_len = seq_len
        self.data = {}
        for i, k in enumerate(columns):
            dtype = torch.float32 if cnt_list[i] == 1 else torch.long
            self.data[k] = torch.tensor(temporal_data[k].values, dtype=dtype)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        '''
        Args:
            idx : index of time
        Returns:
            output : dict of tensor {column : (seq_len, embedding_dim) }
        '''
        output = {}
        for k, v in self.data.items():
            output[k] = v[idx:idx+self.seq_len]
        return output

    def collate(self, batch):
        '''
        Args:
            batch : batch is list of output which is same length with Batch
        Returns:
            output : dict of tensor {column : (batch, seq_len, embedding_dim)}
        '''
        output = {}
        for k, v in self.data.items():
            output[k] = torch.stack([x[k] for x in batch], dim=0)
        return output


#########################################
class MovingPopDailyDataset(Dataset):
    '''
    MovingPopDailyDataset

        Use moving population data and local confirmed case data to make a training dataset.

        Moving Population Data
            The moving population data is on a daily basis and there is data for each region for departure and destination.
            Data has channels and channels are divided by sex and age.
            shape : (time, region, region, channel : sex * age )

        local confirmed case data
            The local confirmed case data is on a daily basis and there is data for each region. (e.g. confirmed case of covid)
            shape : (time, region, channel : covid + else )

    '''

    def __init__(self, mp_data, covid_data,

                 seq_len, pred_len):
        '''
        Args:
            mp_data: moving population data
            covid_data: local confirmed case data
            seq_len: length of input sequence
            pred_len: length of prediction
        '''
        super().__init__()

        if mp_data.shape[0] != covid_data.shape[0]:
            raise Exception('Data length is not equal')
        if len(mp_data) < seq_len + 1:
            raise Exception('Data length is too short')

        self.mp_data = mp_data
        self.covid_data = covid_data
        self.seq_len = seq_len
        self.pred_len = pred_len


    def __len__(self):
        '''
        Returns:
            length of dataset
        '''
        return self.covid_data.shape[0] - self.seq_len - 1

    def __getitem__(self, idx):
        '''
        Args:
            idx: index of the first element in the sequence
        Returns:
            x_adj  : t-n+1 ~ t+1        (batch, seq_len, node_cnt, node_cnt)
            x_node : t-n ~ t            (batch, seq_len, node_cnt, embedding_dim)
            y_node : t+1 ~ t+1+pred     (batch, node_cnt, embedding_dim)
        '''
        end = idx + self.seq_len
        return (
            torch.tensor(self.mp_data[idx+1: end + 1],
                         dtype=torch.float32),  # x_adj
            torch.tensor(self.covid_data[idx: end],
                         dtype=torch.float32),  # x_node
            torch.tensor(self.covid_data[end: end+1],
                         dtype=torch.float32),  # y_node
        )

class MovingPopDailyWithAuxDataset(Dataset):
    '''
    MovingPopDailyWithAuxDataset

        Use moving population data and local confirmed case data to make a training dataset.

        Moving Population Data
            The moving population data is on a daily basis and there is data for each region for departure and destination.
            Data has channels and channels are divided

        AuxSpatialDataset
            Dataset for using additional data for each region

        AuxTemporalDataset
            Dataset for using additional data for each time
    '''

    def __init__(self, mp_data, covid_data,
                 spatial_data, spatial_columns, spatial_cardinalities,
                 temporal_data, temporal_columns, temporal_cardinalities,
                 seq_len, pred_len):
        '''
        Args:
            mp_data: moving population data
            covid_data: local confirmed case data
            seq_len: length of input sequence
            pred_len: length of prediction
        '''
        super().__init__()

        if mp_data.shape[0] != covid_data.shape[0]:
            raise Exception('Data length is not equal')
        if len(mp_data) < seq_len + 1:
            raise Exception('Data length is too short')

        self.mp_data = mp_data
        self.covid_data = covid_data
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.spatial = {}
        for i, k in enumerate(spatial_columns):
            dtype = torch.float32 if spatial_cardinalities[i] == 1 else torch.long
            self.spatial[k] = torch.tensor(spatial_data[k].values, dtype=dtype)


        self.temporal = {}
        for i, k in enumerate(temporal_columns):
            dtype = torch.float32 if temporal_cardinalities[i] == 1 else torch.long
            self.temporal[k] = torch.tensor(temporal_data[k].values, dtype=dtype)

    def __len__(self):
        '''
        Returns:
            length of dataset
        '''
        return self.covid_data.shape[0] - self.seq_len - 1

    def __getitem__(self, idx):
        '''
        Args:
            idx: index of the first element in the sequence
        Returns:
            x_adj  : t-n+1 ~ t+1        (batch, seq_len, node_cnt, node_cnt)
            x_node : t-n ~ t            (batch, seq_len, node_cnt, embedding_dim)
            y_node : t+1 ~ t+1+pred     (batch, node_cnt, embedding_dim)
            spatial : independent of time (batch, node_cnt, embedding_dim) 
            temporal : independent of node (batch, seq_len, embedding_dim)
        '''
        end = idx + self.seq_len

        # spatial = {}
        # for k, v in self.spatial.items():
            # spatial[k] = v.expand(len(batch), -1)
        temporal = {}
        for k, v in self.temporal.items():
            temporal[k] = v[idx:idx+self.seq_len]
    
        return (
            torch.tensor(self.mp_data[idx+1: end + 1], dtype=torch.float32),# x_adj
            torch.tensor(self.covid_data[idx: end], dtype=torch.float32), # x_node
            torch.tensor(self.covid_data[end: end+1], dtype=torch.float32), # y_node
            self.spatial, # spatial
            temporal # temporal
        )

# Test Code
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
    from torch.utils.data import DataLoader

    os.listdir('../data/')
    spatial = pd.read_csv('../data/df_region_normalized.csv',
                          index_col=0, encoding='cp949')
    temporal = pd.read_csv('../data/df_time_day_normalized.csv',
                           index_col=0, encoding='cp949')

    a = AuxSpatialDataset(
        spatial, args['aux_spatial_columns'], args['aux_spatial_cardinalities'], 10)
    b = DataLoader(a, batch_size=args['batch_size'], collate_fn=a.collate)

    for i in b:
        print(i.keys())
        break

    a = AuxTemporalDataset(
        temporal, args['aux_temporal_columns'], args['aux_temporal_cardinalities'], args['seq_len'], 10)
    b = DataLoader(a, batch_size=args['batch_size'], collate_fn=a.collate)

    for i in b:
        print(i.keys())
        break

# %%
