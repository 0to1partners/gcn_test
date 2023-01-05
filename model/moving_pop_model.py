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
    from model.modules import NodeEncoder, NodeDecoder, GraphTemporalConv
    from model.additional_info import AdditionalInfo
    from model.adjacency import AdjacencyModel, AdjacencyWithMP

except:
    from modules import NodeEncoder, NodeDecoder, GraphTemporalConv
    from additional_info import AdditionalInfo
    from adjacency import AdjacencyModel, AdjacencyWithMP


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

            node_dim : dimension of original node
            node_cnt : total node count            
            node_latent_dim : latent dimension of each node

            node_hidden_dim : hidden dimension of each layer of node encoder and decoder
            node_num_layers : number of layers of node encoder and decoder

            num_graph_layers : number of graph_layers
        '''
        super().__init__()

        self.adj_module = AdjacencyWithMP(input_dim=kwargs['adj_input_dim'],
                                          hidden_dim=kwargs['adj_hidden_dim'],
                                          output_dim=kwargs['adj_embedding_dim'],
                                          adj_channel=kwargs['adj_channel'],
                                          adj_num_layers=kwargs['adj_num_layers'])
        
        self.spatial_encoder = AdditionalInfo(column_list=kwargs['spatial_columns'],
                                            cnt_list=kwargs['spatial_cardinalities'],
                                            add_embed_dim=kwargs['spatial_embedding_dim'])
        
        self.temporal_encoder = AdditionalInfo(column_list=kwargs['temporal_columns'],
                                            cnt_list=kwargs['temporal_cardinalities'],
                                            add_embed_dim=kwargs['temporal_embedding_dim'])

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

        self.aggregator = nn.Conv1d(
            kwargs['seq_len'], kwargs['pred_len'], kernel_size=1)

    def forward(self, x, moving_pop, spatial = None, temporal = None):
        '''
        args
            x : (batch_size, sequence, node_cnt, node_dim)
            moving_pop : (batch_size, sequence, node_cnt, node_dim)
        output
            x : (batch_size, pred_len ,node_cnt, , node_dim)
        '''

        # (batch_size, seq, node_cnt, node_latent_dim)
        x = self.node_encoder(x)
        # Add Spatial Data
        if spatial is not None:
            spatial = self.spatial_encoder(spatial).unsqueeze(1)
            x = x + spatial

        if temporal is not None:
            temporal = self.temporal_encoder(temporal)
            moving_pop = moving_pop + temporal.unsqueeze(-1).unsqueeze(-1)
            
        # (batch_size, seq, adj_channel, node_cnt, node_cnt)
        adj, adj_recon = self.adj_module(moving_pop)
        b, s, n, c = x.shape

        # (batch_size, seq, node_cnt, node_latent_dim)
        x = self.graph_conv(x, adj)
        # (batch_size, node_cnt, seq, node_latent_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        # (batch_size * node_cnt, seq, node_latent_dim)
        x = x.view(b * n, s, c)

        x = self.aggregator(x)  # (batch_size * node_cnt, 1, node_latent_dim)
        # (batch_size * node_cnt, node_latent_dim, seq)
        x = x.permute(0, 2, 1).contiguous()

        # (batch_size, pred_len, node_cnt, node_latent_dim)
        x = x.view(b, -1, n, c)
        # (batch_size, pred_len, node_cnt, node_dim)
        y = self.node_decoder(x)

        return x, y




if __name__ == '__main__':
    '''
    test code
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

    # Mok data
    mp_data = torch.randn((hp['batch'],
                           hp['seq_len'],
                           hp['adj_input_dim'],
                           hp['node_cnt'],
                           hp['node_cnt']))

    node_data = torch.randn(hp['batch'],
                            hp['seq_len'],
                            hp['node_cnt'],
                            hp['node_dim'])

    node_latent_data = torch.randn(hp['batch'],
                            hp['seq_len'],
                            hp['node_cnt'],
                            hp['node_latent_dim'])

    adj = torch.randn(hp['batch'], hp['seq_len'],hp['adj_channel'],
                        hp['node_cnt'], hp['node_cnt'])



    #1. AdjacencyWithMP
    print('#1 1 AdjacencyWithMP  test', end=': \t')
    adj_module = AdjacencyWithMP(input_dim=hp['adj_input_dim'],
                                 hidden_dim=hp['adj_hidden_dim'],
                                 output_dim=hp['adj_output_dim'],
                                 adj_channel=hp['adj_channel'],
                                 adj_num_layers=hp['adj_num_layers'])

    tmp = adj_module(mp_data)
    print(tmp[0].shape, tmp[1].shape)


    #2. NodeEncoder, NodeDecoder
    print('#2 Node encoder test', end=': \t')
    encoder = NodeEncoder(node_dim=hp['node_dim'],
                          hidden_dim=hp['node_hidden_dim'],
                          node_latent_dim=hp['node_latent_dim'],
                          num_layers=hp['node_num_layers'])
    output = encoder(node_data)
    print(output.shape)

    print('#3 Node decoder test', end=': \t')
    decoder = NodeDecoder(node_latent_dim=hp['node_latent_dim'],
                          hidden_dim=hp['node_hidden_dim'],
                          node_dim=hp['node_dim'],
                          num_layers=hp['node_num_layers'])
    print(decoder(output).shape)

    print('#4 Encoder, Decoder Connection test', end=': \t')
    print(decoder(encoder(node_data)).shape)


    #5. GraphTemporalConv
    print('#5 Graph Conv test', end=': \t')
    model = GraphTemporalConv(hidden_dim=hp['node_latent_dim'],
                              node_cnt=hp['node_cnt'],
                              seq_len=hp['seq_len'])
    print(model(node_latent_data, adj).shape)


    #6 Intergrated Architecture
    print('#6 Intergrated Architecture test', end=': \t')
    model = PopulationWeightedGraphModel(**hp)
    tmp = model(node_data, mp_data)
    print(tmp[0].shape, tmp[1].shape)

# %%
