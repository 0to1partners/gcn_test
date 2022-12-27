
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditionalInfo(nn.Module):
    '''
    AdditionalInfo
        Summarize additional info embedding
        If the data is temporal, seq_len is the number of time points
        If the data is spatial, seq_len is the number of spatial points
    '''
    def __init__(self, column_list, cnt_list, add_embed_dim) : 
        '''
        args 
            column_list : list of column name
            cnt_list : list of unique value count ( if cnt == 1, it is continuous variable )
            add_embed_dim : embedding dimension
        '''
        super().__init__()
        self.embeddings = nn.ModuleDict()

        for i, col in enumerate(column_list):
            if cnt_list[i] > 1:
                self.embeddings[col] = nn.Embedding(cnt_list[i], add_embed_dim)
            elif cnt_list[i] == 1:
                self.embeddings[col] = nn.Sequential(
                    nn.Unflatten(1, (-1, 1)),
                    nn.Linear(1, add_embed_dim)
                )

    def forward(self, x):
        '''
        args
            x : dict of tensor { channel : (batch, seq_len) }
        return
            out : summary of additional info tensor (batch, seq_len, add_embed_dim)
        '''
        outputs = []
        for k, v in x.items(): 
            outputs.append( self.embeddings[k](v) ) # (batch, seq_len, add_embed_dim)
        out = torch.stack(outputs, dim = 1) # (batch, channel, seq_len, add_embed_dim)
        out = out.sum(dim=1) # (batch, seq_len, add_embed_dim)
        return out


# Test
if __name__ == '__main__':

    b = 2
    s = 3
    area = 6
    dim = 5
    embedding_dim = 4

    model = AdditionalInfo(['tmp', 'tmp2'], [3, 1], embedding_dim)


    # Make dummy data
    data = {
        'tmp' : torch.ones(b, s, dtype=torch.long) * torch.arange(s),
        'tmp2' : (torch.ones(b, s, dtype=torch.float32) * torch.arange(s)),
    }
    print(data['tmp'])
    print(data['tmp2'])
    print(model)

    out = model(data)    
    print(out.shape)

# %%