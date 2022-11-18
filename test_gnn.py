
#%%
import torch
import torch.nn.functional as F
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics import Accuracy

#%%
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes

        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# %%

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        print(len(dataset))

    def setup(self, stage=None):
        seed_everything(42)
        self.dataset = self.dataset.shuffle()

    def train_dataloader(self):
        return DataLoader(self.dataset[:140], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset[140:170], batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset[170:], batch_size=self.batch_size)


#%%
class TrainModule(pl.LightningModule):
    # def __init__(self, model, dataset, lr=0.01, weight_decay=5e-4):
    def __init__(self, model, lr=0.01, weight_decay=5e-4):
        super().__init__()
        # self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        # self.dataset = dataset

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss)

        # print('이게문젠가?')
        # acc = Accuracy(out[batch.train_mask], batch.y[batch.train_mask])
        # print('이거네 이거?')
        # self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        self.log('val_loss', loss)
        
        # acc = Accuracy(out[batch.val_mask], batch.y[batch.val_mask])
        # self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask])
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

#%%
if __name__ == '__main__':

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # dataloader = DataLoader(dataset)
    data_module = DataModule(dataset)
    

    model = GCN(dataset.num_node_features, dataset.num_classes)
    logger = TensorBoardLogger('lightning_logs', name='test_gnn')

    train_module = TrainModule(model)

    trainer = pl.Trainer(gpus=1, max_epochs=200)
    trainer.fit(train_module, datamodule=data_module)
    # trainer.fit(train_module, dataloader)

# %%




