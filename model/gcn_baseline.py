# %%

from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.nn.dense import DenseGraphConv
from torch_geometric.nn.norm import LayerNorm, BatchNorm

import torch
import numpy as np
from tqdm.auto import tqdm

# %%


class AdjacencyModel(nn.Module):
    def __init__(self, adj_embedding_dim, node_cnt, hidden_dim, adj_num_layers, embedding_dict):
        '''
        args
        adj_embedding_dim : embedding dimension 
        node_cnt : total node count
        hidden_dim : hidden dimension of each layer
        adj_num_layers : number of layers
        embedding_dict : key is categorical tensor, value is cardinality
                example {'month': 12, 'wday': 7, 'hour': 24}

        output
            adjacency matrix : 
                with embedding dict -> (batch_size, node_cnt, node_cnt) 
                without -> (node_cnt, node_cnt)
        '''

        super().__init__()

        self.embeddings = nn.ModuleDict()
        self.node_cnt = node_cnt

        if embedding_dict:
            self.embedding_list = embedding_dict.keys()

            for k, n in embedding_dict.items():
                self.embeddings[k] = nn.Embedding(n, adj_embedding_dim)

            self.norm = nn.BatchNorm1d
        else:
            self.embedding_list = None
            self.latent = nn.Parameter(
                torch.Tensor(1, adj_embedding_dim).uniform_(0, 1))

            self.norm = nn.InstanceNorm1d

        self.layers = nn.ModuleList()

        if adj_num_layers <= 1:
            raise ValueError('adj_num_layers must be greater than 1')

        for i in range(adj_num_layers - 1):
            if i == 0:
                self.layers.append(nn.Linear(adj_embedding_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(self.norm(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, self.node_cnt**2))

    def forward(self, add_dict=None):

        if self.embedding_list:
            if add_dict is None:
                raise ValueError('Model need add_dict')
            emb_list = []

            for key, value in add_dict.items():
                if key not in self.embedding_list:
                    raise ValueError(f'{key} is not in embedding list')

                embedding = self.embeddings[key](value)
                emb_list.append(embedding)

            stacked = torch.stack(emb_list, dim=1)
            x = torch.sum(stacked, dim=1)
        else:
            x = self.latent

        for layer in self.layers:
            x = layer(x)

        x = x.view(-1, self.node_cnt, self.node_cnt).contiguous()

        return x


class NodeEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, node_latent_dim, num_layers):
        '''
        args
            input_dim : input dimension
            hidden_dim : hidden dimension of each layer
            node_latent_dim : latent dimension of each node
            num_layers : number of layers

        output
            node latent : (batch_size, node_cnt, node_latent_dim)
        '''
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):
            input_dim = hidden_dim if i > 0 else node_dim

            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, node_latent_dim))

    def forward(self, x):
        b, n, _ = x.shape
        x = x.reshape(b * n, -1)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape(b, n, -1)
        return x


class NodeDecoder(nn.Module):
    def __init__(self, node_latent_dim, hidden_dim, node_dim, num_layers):
        '''
        args
            node_latent_dim : latent dimension of each node
            hidden_dim : hidden dimension of each layer
            node_dim : output dimension
            num_layers : number of layers

        output
            node latent : (batch_size, node_cnt, node_dim)
        '''
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):
            input_dim = hidden_dim if i > 0 else node_latent_dim

            self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, node_dim))

    def forward(self, x):
        b, n, _ = x.shape
        x = x.reshape(b * n, -1)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape(b, n, -1)
        return x


class WeightedGraphModel(nn.Module):
    def __init__(self, **kwargs):
        '''
        args
            adj_hidden_dim : hidden dimension of each layer of adjacency model
            adj_num_layers : number of layers of adjacency model
            adj_embedding_dim : embedding dimension of adjacency model
            adj_embedding_dict : key is categorical tensor, value is cardinality

            node_dim : dimension of original node
            node_cnt : total node count            
            node_latent_dim : latent dimension of each node

            node_hidden_dim : hidden dimension of each layer of node encoder and decoder
            node_num_layers : number of layers of node encoder and decoder

            num_graph_layers : number of graph_layers

        output
            node latent : (batch_size, node_cnt, node_latent_dim)
            node_recon : (batch_size, node_cnt, node_dim)
            
        '''
        super(WeightedGraphModel, self).__init__()

        self.adj_module = AdjacencyModel(adj_embedding_dim=kwargs['adj_embedding_dim'],
                                         node_cnt=kwargs['node_cnt'],
                                         hidden_dim=kwargs['adj_hidden_dim'],
                                         adj_num_layers=kwargs['adj_num_layers'],
                                         embedding_dict=kwargs['adj_embedding_dict'])

        self.node_encoder = NodeEncoder(node_dim=kwargs['node_dim'],
                                        hidden_dim=kwargs['node_hidden_dim'],
                                        node_latent_dim=kwargs['node_latent_dim'],
                                        num_layers=kwargs['node_num_layers'])

        self.node_decoder = NodeDecoder(node_latent_dim=kwargs['node_latent_dim'],
                                        hidden_dim=kwargs['node_hidden_dim'],
                                        node_dim=kwargs['node_dim'],
                                        num_layers=kwargs['node_num_layers'])                           

        hidden_dim = kwargs['node_latent_dim']
        self.graph_layers = nn.ModuleList()
        for i in range(kwargs['num_graph_layers']):
            self.graph_layers.append(DenseGraphConv(
                hidden_dim, hidden_dim, aggr='mean', bias=False))
            self.graph_layers.append(nn.ReLU())

            self.graph_layers.append(BatchNorm(kwargs['node_cnt']))


    def forward(self, x, add_dict=None):

        x = self.node_encoder(x)
        adj = self.adj_module(add_dict)

        for layer in self.graph_layers:
            if isinstance(layer, DenseGraphConv):
                x = layer(x, adj)
            else:
                x = layer(x)

        recon = self.node_decoder(x)

        return x, recon




# %%
if __name__ == '__main__':
    '''
    test code
    '''

    print('Adjacency without embedding dict test', end=': \t')
    adj_module = AdjacencyModel(embedding_dim=10,
                                node_cnt=25,
                                hidden_dim=128,
                                num_layers=3)

    tmp = adj_module()
    print(tmp.shape)

    print('Adjacency with embedding dict test', end=': \t')
    adj_module = AdjacencyModel(embedding_dim=10,
                                node_cnt=25,
                                hidden_dim=128,
                                num_layers=3,
                                embedding_dict={'month': 12, 'wday': 7, 'hour': 24})

    batch = 10
    add_dict = {'month': torch.randint(0, 12, (batch,)),
                'wday': torch.randint(0, 7, (batch,)),
                'hour': torch.randint(0, 24, (batch,))}

    tmp = adj_module(add_dict)
    print(tmp.shape)

    print('Node encoder test', end=': \t')
    encoder = NodeEncoder(node_dim=28,
                          hidden_dim=128,
                          node_latent_dim=16,
                          num_layers=3)
    x_input = torch.randn(batch, 25, 28)
    tmp = encoder(x_input)
    print(tmp.shape)

    print('Node decoder test', end=': \t')
    decoder = NodeDecoder(node_latent_dim=16,
                          hidden_dim=128,
                          node_dim=28,
                          num_layers=3)

    tmp = decoder(torch.randn(batch, 25, 16))
    print(tmp.shape)

    print('Encoder, Decoder Connection test', end=': \t')
    tmp = decoder(encoder(x_input))
    print(tmp.shape)

    print('Graph Model test', end=': \t')
    

    args = {
        'adj_hidden_dim': 128,
        'adj_num_layers': 3,
        'adj_embedding_dim': 8,
        'adj_embedding_dict': {'month': 12, 'wday': 7, 'hour': 24},

        'node_dim': 28,
        'node_cnt': 25,
        'node_latent_dim': 16,
        'node_hidden_dim': 128,
        'node_num_layers': 3,

        'num_graph_layers': 3
    }
    model = WeightedGraphModel( **args)

    tmp = model(x_input, add_dict)
    print(tmp[0].shape, tmp[1].shape)

# %%
