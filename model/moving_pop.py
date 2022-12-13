# %%

from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch_geometric.nn.dense import DenseGraphConv
from torch_geometric.nn.norm import LayerNorm, BatchNorm

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm


try:
    from model.adjacency import AdjacencyModel, AdjacencyWithMP

except:
    from adjacency import AdjacencyModel, AdjacencyWithMP

# %%


class NodeEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, node_latent_dim, num_layers):
        '''
        args
            input_dim : input dimension
            hidden_dim : hidden dimension of each layer
            node_latent_dim : latent dimension of each node
            num_layers : number of layers
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
        '''
        args
            x : (batch_size, sequence, node_cnt, node_dim)
        output
            node latent : (batch_size, sequence, node_cnt, node_latent_dim)
        '''

        b, s, n, _ = x.shape
        x = x.reshape(b * s * n, -1)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape(b, s, n, -1)
        return x


class NodeDecoder(nn.Module):
    '''
    description
        decode node latent to node feature
    '''

    def __init__(self, node_latent_dim, hidden_dim, node_dim, num_layers):
        '''
        args
            node_latent_dim : latent dimension of each node
            hidden_dim : hidden dimension of each layer
            node_dim : output dimension
            num_layers : number of layers
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
        '''
        args
            x : (batch_size, sequence, node_cnt, node_latent_dim)
                or (batch_size, node_cnt, node_latent_dim)
        output
            node latent : (batch_size, sequence, node_cnt, node_dim)
                            or (batch_size, node_cnt, node_dim)
        '''
        is_sequence = (x.dim() == 4)

        if is_sequence == False:
            x = x.unsqueeze(1)

        b, s, n, _ = x.shape
        x = x.reshape(b * s * n, -1)

        for layer in self.layers:
            x = layer(x)

        if is_sequence == False:
            x = x.reshape(b, n, -1)
        else:
            x = x.reshape(b, s, n, -1)
        return x


class GraphTemporalConv(nn.Module):
    ''' 
    description
        Dense Graph Layer With 1D Convolution Layer
        2 * Dense Graph Layer + 2 * 1D Convolution Layer
    '''

    def __init__(self, hidden_dim, node_cnt, seq_len):
        '''
        args
            hidden_dim : hidden dimension of each layer
            node_cnt : number of nodes
            seq_len : length of sequence
        '''
        super().__init__()

        self.kernel_size = 5
        self.padding = self.kernel_size // 2

        self.gcn1 = DenseGraphConv(
            hidden_dim, hidden_dim, aggr='mean', bias=False)
        self.gcn2 = DenseGraphConv(
            hidden_dim, hidden_dim, aggr='mean', bias=False)

        self.batch_norm1 = nn.BatchNorm1d(node_cnt)
        self.batch_norm2 = nn.BatchNorm1d(node_cnt)

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.kernel_size,
                               padding=self.padding)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.kernel_size,
                               padding=self.padding)

        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, adj):
        '''
        args
            x : (batch_size, seq, node_cnt, in_channels)

        output
            x : (batch_size, seq, node_cnt, out_channels)
        '''
        b, s, n, c = x.shape  # b, s, n, c

        x = x.reshape(b * s, n, c)
        adj = adj.reshape(b * s, n, n)

        x = F.relu(self.gcn1(x, adj))  # b * s, n, c
        x = self.batch_norm1(x)
        x = F.relu(self.gcn2(x, adj))
        x = self.batch_norm2(x)

        x = x.reshape(b, s, n, c)

        x = x.permute(0, 2, 3, 1).contiguous()  # b, n, c, s
        x = x.view(b * n, c, s)  # b * n, c, s

        x = F.relu(self.conv1(x))  # b * n, c, s
        x = self.batch_norm3(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm4(x)
        x = x.view(b, n, c, s)  # b, n, c, s
        x = x.permute(0, 3, 1, 2).contiguous()  # b, s, n, c

        return x


class PopulationWeightedGraphModel(nn.Module):
    '''
    description
        Graph Model with Moving Population
    '''

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
        '''
        super().__init__()

        self.adj_module = AdjacencyWithMP(input_dim=kwargs['node_dim'],
                                          hidden_dim=kwargs['adj_hidden_dim'],
                                          output_dim=kwargs['adj_embedding_dim'],
                                          adj_num_layers=kwargs['adj_num_layers'])

        self.node_encoder = NodeEncoder(node_dim=kwargs['node_dim'],
                                        hidden_dim=kwargs['node_hidden_dim'],
                                        node_latent_dim=kwargs['node_latent_dim'],
                                        num_layers=kwargs['node_num_layers'])

        self.node_decoder = NodeDecoder(node_latent_dim=kwargs['node_latent_dim'],
                                        hidden_dim=kwargs['node_hidden_dim'],
                                        node_dim=kwargs['node_dim'],
                                        num_layers=kwargs['node_num_layers'])

        self.graph_conv = GraphTemporalConv(hidden_dim=kwargs['node_latent_dim'],
                                            node_cnt=kwargs['node_cnt'],
                                            seq_len=kwargs['seq_len'])

        self.aggregator = nn.Conv1d(kwargs['seq_len'], kwargs['pred_len'],
                                    kernel_size=1)

    def forward(self, x, add_dict=None):
        '''
        args
            x : (batch_size, node_cnt, node_dim)
            add_dict : additional dictionary

        output
            x : (batch_size, node_cnt, pred_len, node_dim)
        '''

        x = self.node_encoder(x)
        adj = self.adj_module(add_dict)

        b, s, n, c = x.shape

        # (batch_size, seq, node_cnt, node_latent_dim)
        x = self.graph_conv(x, adj)

        # (batch_size, node_cnt, seq, node_latent_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (batch_size * node_cnt, seq, node_latent_dim)
        x = x.view(b * n, s, c)
        # (batch_size * node_cnt, node_latent_dim, pred_len)
        x = self.aggregator(x)

        # (batch_size * node_cnt, pred_len, node_latent_dim)
        x = x.permute(0, 2, 1).contiguous()
        # (batch_size, pred_len, node_cnt, node_latent_dim)
        x = x.view(b, -1, n, c)

        recon = self.node_decoder(x)

        return x, recon


# %%
if __name__ == '__main__':
    '''
    test code
    '''

    print('Adjacency without embedding dict test', end=': \t')
    adj_module = AdjacencyModel(adj_embedding_dim=10,
                                node_cnt=25,
                                hidden_dim=128,
                                adj_num_layers=3,
                                embedding_dict={'month': 12, 'wday': 7})
    batch = 16

    add_dict = {'month': torch.randint(0, 12, (batch, 12, )),
                'wday': torch.randint(0, 7, (batch, 12, )),
                }

    tmp = adj_module(add_dict)
    print(tmp.shape)

    # print('Adjacency with embedding dict test', end=': \t')
    # adj_module = AdjacencyModel(adj_embedding_dim=10,
    #                             node_cnt=25,
    #                             hidden_dim=128,
    #                             adj_num_layers=3,
    #                             embedding_dict={'month': 12, 'wday': 7, 'hour': 24})

    # batch = 16
    # seq = 10
    # add_dict = {'month': torch.randint(0, 12, (batch, 10,)),
    #             'wday': torch.randint(0, 7, (batch, 10,)),
    #             'hour': torch.randint(0, 24, (batch, 10,))}

    # tmp = adj_module(add_dict)
    # print(tmp.shape)

    # batch = 16
    # seq = 10

    # print('Node encoder test', end=': \t')
    # encoder = NodeEncoder(node_dim=28,
    #                       hidden_dim=128,
    #                       node_latent_dim=16,
    #                       num_layers=3)
    # x_input = torch.randn(batch, seq,25, 28)
    # output = encoder(x_input)
    # print(output.shape)

    # print('Node decoder test', end=': \t')
    # decoder = NodeDecoder(node_latent_dim=16,
    #                       hidden_dim=128,
    #                       node_dim=28,
    #                       num_layers=3)

    # tmp = decoder(output)
    # print(tmp.shape)

    # print('Encoder, Decoder Connection test', end=': \t')
    # tmp = decoder(encoder(x_input))
    # print(tmp.shape)

    # print('Graph Conv test', end=': \t')
    # seq = 24
    # batch = 4
    # node_cnt = 25
    # emb = 10
    # model = GraphTemporalConv(hidden_dim=emb, node_cnt=node_cnt, seq_len=seq)

    # x_input = torch.randn(batch, seq, node_cnt, emb)
    # adj = torch.randn(batch, seq, node_cnt, node_cnt)
    # tmp = model(x_input, adj)
    # print(tmp.shape)

    print('Intergrated Architecture test', end=': \t')

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

        # 'num_graph_layers': 3,
        'seq_len': 24,
        'pred_len': 12,
    }
    model = WeightedGraphModel(**args)

    batch = 4
    seq = args['seq_len']

    x_input = torch.randn(batch, seq, 25, 28)
    add_dict = {'month': torch.randint(0, 12, (batch, seq,)),
                'wday': torch.randint(0, 7, (batch, seq,)),
                'hour': torch.randint(0, 24, (batch, seq,))}

    tmp = model(x_input, add_dict)

    print(tmp[0].shape, tmp[1].shape)

# %%
