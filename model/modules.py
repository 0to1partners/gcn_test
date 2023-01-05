import torch
from torch import nn

from torch_geometric.nn.dense import DenseGraphConv
from torch_geometric.nn.norm import LayerNorm, BatchNorm
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    '''
    description
        encode node feature to node latent
    '''
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
            adj : (batch_size, seq, adj_channel, node_cnt, node_cnt)

        output
            x : (batch_size, seq, node_cnt, out_channels)
        '''
        b, s, n, c = x.shape  # b, s, n, c
        adj_c = adj.shape[2]

        x = x.reshape(b * s, n, c)

        if adj_c == 1:
            adj = adj.repeat(1, 1, 2, 1, 1)

        adj = adj.permute(2, 0, 1, 3, 4).contiguous()  # b, s, n, n, c
        adj = adj.view(2, b * s, n, n)

        x = F.relu(self.gcn1(x, adj[0]))  # b * s, n, c
        x = self.batch_norm1(x)
        x = F.relu(self.gcn2(x, adj[1]))
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


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        '''
        args
            x : (batch_size, sequence, node_cnt)
        output
            moving average : (batch_size, sequence, node_cnt)
        '''
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1) # (batch_size, sequence + kernel_size - 1, node_cnt)
        x = x.permute(0, 2, 1) # (batch_size, node_cnt, sequence + kernel_size - 1
        x = self.avg(x) # (batch_size, node_cnt, sequence)
        x = x.permute(0, 2, 1) # (batch_size, sequence, node_cnt)
        return x
    

class DecompositionLayer(nn.Module):
    """
    Series decomposition block
    Split the time series into trend and residual
    """
    def __init__(self, kernel_size):
        '''
        args
            kernel_size : kernel size of moving average
        '''
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        '''
        args
            x : (batch_size, sequence, node_cnt)
        output
            res : (batch_size, sequence, node_cnt)
            moving_mean : (batch_size, sequence, node_cnt)
        '''
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean