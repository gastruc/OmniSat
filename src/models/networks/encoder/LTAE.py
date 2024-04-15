from models.networks.encoder.utils.ltae import LTAE2d
import torch
import torch.nn as nn

modality_to_channels = {'aerial': 4, 's2': 10, 's1-asc': 2, 's1-des': 2, 's1': 2}

class LTAE(nn.Module):
    """
    Initializes LTAE model
    """
    def __init__(self,
                 modalities: list = [],
                 n_head: int = 16, 
                 d_k: int = 8,
                 mlp: list = [10, 32, 128],
                 output_dim: int = 128,
                 dropout: float = 0.2,
                 mlp_in: list = [10, 32, 128],
                 T: int = 367,
                 in_norm: bool = True,
                 return_att: bool = True,
                 positional_encoding: bool =  True
                 ):
        super(LTAE, self).__init__()
        self.modality = modalities[0]
        in_channels = modality_to_channels[self.modality]
        mlp.append(output_dim)
        self.encoder = LTAE2d(in_channels, n_head, d_k, mlp, dropout, mlp_in, T, in_norm, return_att, positional_encoding)

    def forward(self, x):
        x = self.encoder(x[self.modality], batch_positions=x['_'.join([self.modality, "dates"])], pad_mask=None)
        x = torch.mean(x, dim=(2, 3), keepdim=True).squeeze()
        return x