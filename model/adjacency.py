
import torch
import torch.nn as nn
import torch.nn.functional as F




class PopulationConverter(nn.Module):
    '''
    Encode population to adjacency matrix and vice versa
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, adj_channel, is_gate):
        '''
        args
            input_dim : input dimension
            hidden_dim : hidden dimension of each layer
            output_dim : output dimension
            num_layers : number of layers
            is_gate : if True, output is sigmoid
        '''
        super().__init__()
        self.adj_channel = adj_channel

        if num_layers <= 1:
            raise ValueError('num_layers must be greater than 1')

        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):
            if i == 0:
                self.layers.append(nn.Conv2d(input_dim, hidden_dim, kernel_size=1))
            else:
                self.layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm2d(hidden_dim))

        self.layers.append(nn.Conv2d(hidden_dim, output_dim * self.adj_channel, kernel_size=1))
        
        if is_gate:
            self.layers.append(nn.Sigmoid())


    def forward(self, x):
        '''
        args
            x : (batch, seq, input_dim, node_cnt, node_cnt)
        output
            x : (batch, seq, adj_channel, node_cnt, node_cnt)
        '''

        b, s, e, n = x.shape[:4]
        x = x.view(b * s, e, n, n) # (batch * seq, input_dim, node_cnt, node_cnt)

        for layer in self.layers:
            x = layer(x) # (batch * seq, output_dim * adj_channel, node_cnt, node_cnt)

        x = x.view(b, s, -1, n, n) # (batch, seq, output_dim, node_cnt, node_cnt)

        return x



class AdjacencyWithMP(nn.Module):
    '''
    AdjacencyWithMP
        Convert population to adjacency matrix and vice versa
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, adj_channel, adj_num_layers):
        '''
        args
            input_dim : input dimension
            hidden_dim : dimension of each hidden layer
            output_dim : output dimension
            adj_channel : number of adjacency matrix
            adj_num_layers : number of layers
        '''
        super().__init__()

        self.adj_channel = adj_channel
        self.encoder = PopulationConverter(input_dim, hidden_dim, output_dim, adj_num_layers, adj_channel, is_gate=True)
        self.decoder = PopulationConverter(output_dim, hidden_dim, input_dim, adj_num_layers, 1, is_gate=False)

    def forward(self, x):
        '''
        args
            x : (batch, seq, input_dim, node_cnt, node_cnt)
        output
            latent : (batch, output_dim, adj_channel, adj_channel, node_cnt, node_cnt)
            recon : (batch, input_dim, node_cnt, node_cnt)
        '''
        b, s, c, n, _ = x.shape
        latent = self.encoder(x) # (batch, seq, adj_channel, node_cnt, node_cnt)

        recon = self.decoder(latent) # (batch, seq, input_dim, node_cnt, node_cnt)
        recon = recon.view(b, -1, n, n)

        return latent, recon



class AdjacencyModel(nn.Module):
    def __init__(self, adj_embedding_dim, node_cnt, hidden_dim, adj_num_layers, embedding_dict):
        '''
        args
            adj_embedding_dim : additional attribute embedding dimension 
            node_cnt : total node count
            hidden_dim : hidden dimension of each layer
            adj_num_layers : number of layers
            embedding_dict : key is categorical tensor, value is cardinality
                    example {'month': 12, 'wday': 7, 'hour': 24}
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
        ''' 
        args
            add_dict : key is categorical tensor, value is cardinality
                example {'month': (batch, seq, 1), 'wday': ... }
        output 
            adjacency matrix : (batch_size, seq, node_cnt, node_cnt)
        '''

        # if self.embedding_list:
        if add_dict is None:
            raise ValueError('Model need add_dict')
        emb_list = []

        batch = add_dict[list(add_dict.keys())[0]].shape[0]

        for key, value in add_dict.items():
            if key not in self.embedding_list:
                raise ValueError(f'{key} is not in embedding list')

            embedding = self.embeddings[key](value)
            emb_list.append(embedding)

        stacked = torch.stack(emb_list, dim=2)
        x = torch.sum(stacked, dim=2)

        b, s, e = x.shape

        x = x.reshape(-1, e)
        for layer in self.layers:
            x = layer(x)

        x = x.view(batch, -1, self.node_cnt, self.node_cnt).contiguous()

        return x
