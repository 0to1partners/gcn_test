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
class LifePopHourlyDataset(Dataset):
    def __init__(self, data, add_data=None, is_pred=False):
        super().__init__()
        self.data = data
        self.add_keys = list(add_data)
        self.add_data = np.array(add_data)
        self.is_pred = is_pred

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), \
            torch.tensor(self.add_data[idx+1], dtype=torch.int32), \
            torch.Tensor(self.data[idx+1])

    def collate(self, batch):
        x, add_data, y = map(list, zip(*batch))
        x = torch.stack(x)
        y = torch.stack(y)
        add_data = torch.stack(add_data)

        add_dict = {k: add_data[:,i]
                    for i, k in enumerate(self.add_keys)}

        return x, add_dict, y


class LifePopHourlyModule(pl.LightningDataModule):
    def __init__(self, data, add_data, validation_rate, batch_size, num_workers=4):
        super().__init__()
        self.data = data
        self.add_data = add_data
        self.batch_size = batch_size
        self.num_workers = num_workers

        length = len(self.data)
        self.split = int(length * (1 - validation_rate))

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = LifePopHourlyDataset(self.data[:self.split],
                                                     self.add_data[:self.split])
            self.val_dataset = LifePopHourlyDataset(self.data[self.split:],
                                                   self.add_data[self.split:])
        else:
            raise Exception('Not implemented')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--validation_rate', type=float, default=0.1)
        return parser


# %%
if __name__ == '__main__':
    '''
    Test dataloader
    '''
    data = np.load('data/LOCAL_PEOPLE_GU_2021.npy')
    add_dict = {'temp': np.random.randint(0, 5, size=(len(data))),
                'wday': np.random.randint(0, 7, size=(len(data))),
                'month': np.random.randint(0, 12, size=(len(data))), }
    add_data = pd.DataFrame(add_dict)

    dm = LifePopHourlyModule(data, add_data=add_data,
                            validation_rate=0.2, batch_size=32)
    dm.setup(stage='fit')

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(len(train_loader))
    print(len(val_loader))

    for x, add_dict, y in train_loader:
        print(x.shape)
        print(add_dict['temp'].shape)
        print(y.shape)
        break
        

    print(x.dtype)
    print(add_dict['temp'].dtype)
    print(y.dtype)

    print(add_dict.keys())

# %%
