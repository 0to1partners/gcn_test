#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from pytorch_lightning import seed_everything


#%%


#%%

class GNN(nn.Module):
    # Graph Convolutional Network
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        for layer in self.layers:
            # x = layer(self.dropout(x))
            x = torch.spmm(adj, x)
            # print(x)
            # x = F.relu(x)
        return x

#%%
if __name__ == '__main__':
    seed_everything(42)
    # Test
    gnn = GNN(2, 2, 2, 2)
    
    # x = torch.randn(5,5)
    x = torch.ones((5,5))
    x = torch.tensor([
        [1,],
        [2,],
        [3,],
        [4,],
        [5,]
    ], dtype=torch.float)
    # x = torch.tensor([[1], [10]], dtype=torch.float)

    # b = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)

    print(x)
    
    adj = torch.tensor([
        [0,1,0,0,0],
        [1,0,1,1,0],
        [0,1,0,0,0],
        [0,1,0,0,1],
        [0,0,0,1,0],
        ], dtype=torch.float)

    degree = torch.sum(adj, dim=1)
    degree = torch.diag(degree)
    print(degree)

    laplacian = degree - adj
    # adj2 = torch.sparse()

    print(laplacian)
    normalized_laplacian = torch.inverse(degree) @ laplacian
    print(normalized_laplacian)

    # print(torch.spmm(adj,x))
    # print(gnn(x, adj))

# %%
