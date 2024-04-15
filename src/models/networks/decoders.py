import torch
import torch.nn as nn
import numpy as np

class DecoderModalities(nn.Module):
    """
    Initialize decoders for all modalities.
    """
    def __init__(self,
                 projectors: dict = {}, 
                 modalities: list = [],
                 ):
        super(DecoderModalities, self).__init__()
        self.modalities = modalities

        for i in range(len(modalities)):
            setattr(self, '_'.join(['projector', modalities[i]]), projectors[modalities[i]])


    def forward(self, x, att):
        """
        Affects each encoding with its dedicated decoder
        """
        for i in range(len(self.modalities)):
            if self.modalities[i] == "aerial":
                att['_'.join(['reconstruct', self.modalities[i]])] = getattr(self, 
                            '_'.join(['projector', self.modalities[i]]))(x, att['indices'], att['sizes'])
            elif self.modalities[i].split('-')[-1] == 'mono':
                att['_'.join(['reconstruct', self.modalities[i]])] = getattr(self, 
                            '_'.join(['projector', self.modalities[i]]))(x)
            else:
                att['_'.join(['reconstruct', self.modalities[i]])] = getattr(self,
                                 '_'.join(['projector', self.modalities[i]]))(x,
                                 att['_'.join(['attention', self.modalities[i]])], att['_'.join(['dates', self.modalities[i]])])
        return att

class DecoderSentinelMono(nn.Module):
    """
    Decoder for monodate sentinel data
    """
    def __init__(self,
                 in_channels: int = 10,
                 inter_dim: list = [],
                 embed_dim: int = 128,
                 ):
        super(DecoderSentinelMono, self).__init__()
        layers = []
        if len(inter_dim) > 0:
            inter_dim.insert(0, embed_dim)
            for i in range(len(inter_dim) - 1):
                layers.extend(
                    [
                        nn.Linear(inter_dim[i], inter_dim[i + 1]),
                        nn.ReLU(),
                    ]
                )
            layers.append(nn.Linear(inter_dim[-1], in_channels, bias=True))
        else:
            layers.append(nn.Linear(embed_dim, in_channels, bias=True))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)
    
class DecoderAllSentinel(nn.Module):
    """
    Decoder for sentinel data from OmniSat without date filtering
    """
    def __init__(self,
                 in_channels: int = 10,
                 inter_dim: list = [],
                 embed_dim: int = 128,
                 T: int = 367,
                 ):
        super(DecoderAllSentinel, self).__init__()
        layers = []
        if len(inter_dim) > 0:
            inter_dim.insert(0, embed_dim)
            for i in range(len(inter_dim) - 1):
                layers.extend(
                    [
                        nn.Linear(inter_dim[i], inter_dim[i + 1]),
                        #nn.BatchNorm1d(inter_dim[i + 1]),
                        nn.ReLU(),
                    ]
                )
            layers.append(nn.Linear(inter_dim[-1], in_channels, bias=True))
        else:
            layers.append(nn.Linear(embed_dim, in_channels, bias=True))
        self.decode = nn.Sequential(*layers)
        self.temp_encoding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(T, embed_dim, T=T),
            freeze=True)

    def forward(self, x, attentions, dates, threshold = 0.25):
        recons = []
        masks = []
        for i in range(len(attentions)):
            attentions_i = attentions[i, :torch.nonzero(dates[i] != 0)[-1].item()]
            mask = torch.ones_like(attentions_i, dtype=torch.bool)
            x_i = x[i].unsqueeze(1).expand(x.shape[1], sum(mask), x.shape[2]) + self.temp_encoding(dates[i, :len(mask)][mask])
            indices_mask = torch.zeros((sum(mask), 2)).to(attentions.device) + i
            indices_mask[:, 1] = torch.nonzero(mask).squeeze()
            masks.append(indices_mask.to(torch.int))
            recons.append(self.decode(x_i))
        recons = torch.cat(recons, dim=1).permute((1, 2, 0))
        masks = torch.cat(masks, dim=0)
        return recons, masks
    
class DecoderSentinel(nn.Module):
    """
    Decoder for sentinel data from OmniSat
    """
    def __init__(self,
                 in_channels: int = 10,
                 inter_dim: list = [],
                 embed_dim: int = 128,
                 T: int = 367,
                 ):
        super(DecoderSentinel, self).__init__()
        layers = []
        if len(inter_dim) > 0:
            inter_dim.insert(0, embed_dim)
            for i in range(len(inter_dim) - 1):
                layers.extend(
                    [
                        nn.Linear(inter_dim[i], inter_dim[i + 1]),
                        nn.ReLU(),
                    ]
                )
            layers.append(nn.Linear(inter_dim[-1], in_channels, bias=True))
        else:
            layers.append(nn.Linear(embed_dim, in_channels, bias=True))
        self.decode = nn.Sequential(*layers)
        self.temp_encoding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(T, embed_dim, T=T),
            freeze=True)

    def forward(self, x, attentions, dates, threshold = 0.25):
        recons = []
        masks = []
        for i in range(len(attentions)):
            attentions_i = attentions[i, :torch.nonzero(dates[i] != 0)[-1].item()]
            quartiles = torch.quantile(attentions_i, q=torch.tensor([1 - threshold]).to(attentions.device), dim=0, keepdim=True)
            mask = attentions_i > quartiles[0]
            x_i = x[i].unsqueeze(1).expand(x.shape[1], sum(mask), x.shape[2]) + self.temp_encoding(dates[i, :len(mask)][mask])
            indices_mask = torch.zeros((sum(mask), 2)).to(attentions.device) + i
            indices_mask[:, 1] = torch.nonzero(mask).squeeze()
            masks.append(indices_mask.to(torch.int))
            recons.append(self.decode(x_i))
        recons = torch.cat(recons, dim=1).permute((1, 2, 0))
        masks = torch.cat(masks, dim=0)
        return recons, masks
    
    
class DecoderAerial(nn.Module):
    """
    Decoder for aerial data with a linear layer to C*patch_size**2
    """
    def __init__(self,
                 in_channels: int = 10,
                 patch_size: int = 10,
                 inter_dim: list = [],
                 embed_dim: int = 128
                 ):
        super(DecoderAerial, self).__init__()
        layers = []
        if len(inter_dim) > 0:
            inter_dim.insert(0, embed_dim)
            for i in range(len(inter_dim) - 1):
                layers.extend(
                    [
                        nn.Linear(inter_dim[i], inter_dim[i + 1]),
                        nn.ReLU(),
                    ]
                )
            layers.append(nn.Linear(inter_dim[-1], in_channels * patch_size * patch_size, bias=True))
        else:
            layers.append(nn.Linear(embed_dim, in_channels * patch_size * patch_size, bias=True))
        self.decode = nn.Sequential(*layers)

    def forward(self, x, r, v):
        x = self.decode(x)
        return x
    
class DecoderDeconvAerial(nn.Module):
    """
    Decoder for aerial data with deconvolutions and use of index of maxpools from projector to unpool
    """
    def __init__(self,
                 in_channels: int = 10,
                 embed_dim: int = 128
                 ):
        super(DecoderDeconvAerial, self).__init__()
        self.decode = nn.ModuleList([nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxUnpool2d(kernel_size=2, stride=None),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxUnpool2d(kernel_size=2, stride=None),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxUnpool2d(kernel_size=2, stride=None),
            nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=3, stride=2, padding=1, bias=True)
            ])
        self.max_depools = [True, False, True, False, True, False, True, False]

    def forward(self, x, indices, sizes):
        del sizes[-3]
        del sizes[-2]
        c = 0
        shape = x.shape
        x = x.unsqueeze(-1).unsqueeze(-1).flatten(0,1)
        for i in range (len(self.decode)):
            if self.max_depools[i]:
                x = self.decode[i](x, indices[c], output_size=sizes[i])
                c += 1
            else:
                x = self.decode[i](x, output_size=sizes[i])
        x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3])
        return x
    
class DecoderDeconvAerialPastis(nn.Module):
    """
    Decoder for aerial data with deconvolutions and use of index of maxpools from projector to unpool specific to Pastis
    """
    def __init__(self,
                 in_channels: int = 10,
                 embed_dim: int = 128
                 ):
        super(DecoderDeconvAerialPastis, self).__init__()
        self.decode = nn.ModuleList([
            nn.MaxUnpool2d(kernel_size=2, stride=None),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxUnpool2d(kernel_size=2, stride=None),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxUnpool2d(kernel_size=2, stride=None),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxUnpool2d(kernel_size=2, stride=None),
            nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=3, stride=2, padding=1, bias=True)
            ])
        self.max_depools = [True, False, True, False, True, False, True, False]

    def forward(self, x, indices, sizes):
        del sizes[-3]
        del sizes[-2]
        c = 0
        shape = x.shape
        x = x.unsqueeze(-1).unsqueeze(-1).flatten(0,1)
        for i in range (len(self.decode)):
            if self.max_depools[i]:
                x = self.decode[i](x, indices[c], output_size=sizes[i])
                c += 1
            else:
                x = self.decode[i](x, output_size=sizes[i])
        x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3])
        return x
    
class DecoderDeconvNoIndicesAerial(nn.Module):
    """
    Decoder for aerial data with deconvolutions without index bypass
    """
    def __init__(self,
                 in_channels: int = 10,
                 embed_dim: int = 128
                 ):
        super(DecoderDeconvNoIndicesAerial, self).__init__()
        self.decode = nn.ModuleList([nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=3, stride=2, padding=1, bias=True),
            ])

    def forward(self, x, indices, sizes):
        shape = x.shape
        x = x.unsqueeze(-1).unsqueeze(-1).flatten(0,1)
        for i in range (len(self.decode) - 1):
            x = self.decode[i](x)
        x = self.decode[-1](x, output_size=sizes[-1])
        x = x.view(shape[0], shape[1], x.shape[1], x.shape[2], x.shape[3])
        return x
    
def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    ''' Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)'''

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)