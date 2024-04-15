from torch import nn, einsum
from einops import rearrange
import math
import itertools
import torch
from collections import OrderedDict
import warnings


class PretrainedCROMA(nn.Module):
    def __init__(self, 
                 pretrained_path='CROMA_base.pt', 
                 size='base', 
                 modality='both', 
                 image_resolution=120, 
                 num_classes=13, 
                 modalities=[],
                 in_c_sar=2,
                 in_c_opt=12):
        """
        NOTE: image_resolution is not the spatial, spectral, or temporal resolution. It is the height and width of the image, in pixels.
        E.g., CROMA was pretrained on 120x120px images, hence image_resolution is 120 by default
        """
        super().__init__()
        # check types
        assert type(pretrained_path) == str, f'pretrained_path must be a string, not {type(pretrained_path)}'
        assert type(size) == str, f'size must be a string, not {type(size)}'
        assert type(modality) == str, f'modality must be a string, not {type(modality)}'
        assert type(image_resolution) == int, f'image_resolution must be an int, not {type(image_resolution)}'

        # check values
        assert size in ['base', 'large'], f'size must be either base or large, not {size}'
        assert image_resolution % 8 == 0, f'image_resolution must be a multiple of 8, not {image_resolution}'
        assert modality in ['both', 'SAR', 'optical'], f'modality must be either both, SAR, or optical, not {modality}'

        # warn the user if the path contains a different size than the size parameter
        if size == 'base' and 'large' in pretrained_path:
            warnings.warn("The size is set to base, but the word large appears in the pretrained path!")
        elif size == 'large' and 'base' in pretrained_path:
            warnings.warn("The size is set to large, but the word base appears in the pretrained path!")

        if size == 'base':
            self.encoder_dim = 384
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 8
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 8

        self.modality = modality
        for modalitie in modalities:
            if modalitie.split('-')[0] == 's1':
                self.sar_modality = modalitie
            if modalitie.split('-')[0] == 's2':
                self.optical_modality = modalitie
        self.num_patches = int((image_resolution/8)**2)
        self.s1_channels = in_c_sar
        self.s2_channels = in_c_opt
        self.attn_bias = get_2dalibi(num_heads=self.num_heads, num_patches=self.num_patches)

        pretrained = torch.load(pretrained_path)

        if modality in ['SAR', 'both']:
            print(f'Initializing SAR encoder')
            self.s1_encoder = ViT(dim=self.encoder_dim, depth=int(self.encoder_depth/2), in_channels=self.s1_channels)
            self.GAP_FFN_s1 = nn.Sequential(
                    nn.LayerNorm(self.encoder_dim),
                    nn.Linear(self.encoder_dim, int(4*self.encoder_dim)),  # (BSZ, num_patches, inner_dim)
                    nn.GELU(),  # (BSZ, num_patches, inner_dim)
                    nn.Linear(int(4*self.encoder_dim), self.encoder_dim)  # (BSZ, num_patches, dim)
                )
            
            # load weights
            d_s1_encoder = {}
            d_s1_FFN = {}
            for key in pretrained["state_dict"].keys():
                if 'radar_encoder' in key:
                    d_s1_encoder['.'.join(key.split('.')[2:])] = pretrained["state_dict"][key]
                if 'GAP_FFN_radar' in key:
                    d_s1_FFN['.'.join(key.split('.')[2:])] = pretrained["state_dict"][key]
            self.s1_encoder.load_state_dict(d_s1_encoder)
            self.GAP_FFN_s1.load_state_dict(d_s1_FFN)

        if modality in ['optical', 'both']:
            print(f'Initializing optical encoder')
            self.s2_encoder = ViT(dim=self.encoder_dim, depth=self.encoder_depth, in_channels=self.s2_channels)
            self.GAP_FFN_s2 = nn.Sequential(
                    nn.LayerNorm(self.encoder_dim),
                    nn.Linear(self.encoder_dim, int(4*self.encoder_dim)),  # (BSZ, num_patches, inner_dim)
                    nn.GELU(),  # (BSZ, num_patches, inner_dim)
                    nn.Linear(int(4*self.encoder_dim), self.encoder_dim)  # (BSZ, num_patches, dim)
                )
            
            # load weights
            d_s2_encoder = {}
            d_s2_FFN = {}
            for key in pretrained["state_dict"].keys():
                if 'optical_encoder' in key:
                    d_s2_encoder['.'.join(key.split('.')[2:])] = pretrained["state_dict"][key]
                if 'GAP_FFN_optical' in key:
                    d_s2_FFN['.'.join(key.split('.')[2:])] = pretrained["state_dict"][key]
            self.s2_encoder.load_state_dict(d_s2_encoder)
            self.GAP_FFN_s2.load_state_dict(d_s2_FFN)

        if modality == 'both':
            print(f'Initializing joint SAR-optical encoder')
            self.cross_encoder = BaseTransformerCrossAttn(dim=self.encoder_dim,
                                                        depth=int(self.encoder_depth/2),
                                                        num_heads=self.num_heads,
                                                        )
            
            # load weights
            d_cross_encoder = {}
            for key in pretrained["state_dict"].keys():
                if 'cross_encoder' in key:
                    d_cross_encoder['.'.join(key.split('.')[2:])] = pretrained["state_dict"][key]
            self.cross_encoder.load_state_dict(d_cross_encoder)

        self.head = nn.Linear(self.encoder_dim, num_classes) 

    def forward(self, images):
        if self.modality in ['SAR', 'both']:
            SAR_encodings = self.s1_encoder(imgs=images[self.sar_modality], attn_bias=self.attn_bias.to(images[self.sar_modality].device))  # (bsz, num_patches, encoder_dim)
            SAR_GAP = self.GAP_FFN_s1(SAR_encodings.mean(dim=1))  # (bsz, encoder_dim)
            out = SAR_GAP

        if self.modality in ['optical', 'both']:
            optical_encodings = self.s2_encoder(imgs=images[self.optical_modality], attn_bias=self.attn_bias.to(images[self.optical_modality].device))  # (bsz, num_patches, encoder_dim)
            optical_GAP = self.GAP_FFN_s2(optical_encodings.mean(dim=1))  # (bsz, encoder_dim)
            out = optical_GAP

        if self.modality == 'both':
            joint_encodings = self.cross_encoder(x=SAR_encodings, context=optical_encodings, relative_position_bias=
                                                 self.attn_bias.to(images[self.optical_modality].device))  # (bsz, num_patches, encoder_dim)
            joint_GAP = joint_encodings.mean(dim=1)  # (bsz, encoder_dim)
            out = joint_GAP

        return self.head(out)

def get_2dalibi(num_heads, num_patches):
    # inspired by: https://github.com/ofirpress/attention_with_linear_biases
    points = list(itertools.product(range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))))

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                               :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


class FFN(nn.Module):
    def __init__(self,
                 dim,
                 mult=4,
                 dropout=0.,
                 ):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)  # (BSZ, num_patches, dim)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 dropout=0.,
                 ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be evenly divisible by num_heads'
        dim_head = int(dim / num_heads)
        self.scale = dim_head ** -0.5

        self.create_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.create_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = attention_scores + relative_position_bias  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(dim=-1)  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (BSZ, num_patches, dim)
        return self.out(out)  # (BSZ, num_patches, dim)


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 dropout=0.,
                 ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be evenly divisible by num_heads'
        dim_head = int(dim / num_heads)
        self.scale = dim_head ** -0.5

        self.create_q = nn.Linear(dim, dim, bias=False)
        self.create_k = nn.Linear(dim, dim, bias=False)
        self.create_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        context = self.input_norm(context)  # (BSZ, num_patches, dim)

        q = self.create_q(x)  # (BSZ, num_patches, dim)
        k = self.create_k(context)  # (BSZ, num_patches, dim)
        v = self.create_v(context)  # (BSZ, num_patches, dim)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = attention_scores + relative_position_bias  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(dim=-1)  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class BaseTransformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads=8,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 ff_mult=4,
                 final_norm=True,
                 ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, relative_position_bias=False):
        for self_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        if self.final_norm:
            return self.norm_out(x)
        else:
            return x


class BaseTransformerCrossAttn(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads=8,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 ff_mult=4,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                CrossAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context, relative_position_bias):
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = cross_attn(x, context, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        x = self.norm_out(x)
        return x  # (BSZ, num_patches, dim)


class ViT(nn.Module):
    def __init__(self, dim, depth, in_channels):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.dim = dim
        self.num_heads = 16  # always 16, for base and large models
        self.patch_size = 8  # always 8, for base and large models

        pixels_per_patch = int(self.patch_size * self.patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(dim=self.dim,
                                           depth=self.depth,
                                           num_heads=self.num_heads,
                                           )

    def forward(self, imgs, attn_bias):

        x = rearrange(imgs, 'b c (h i) (w j) -> b (h w) (c i j)', c=self.in_channels, i=self.patch_size, j=self.patch_size)
        # x is shape -> (bsz, num_patches, self.channels*self.patch_size*self.patch_size)

        x = self.linear_input(x)  # (bsz, num_patches, dim)
        x = self.transformer(x, relative_position_bias=attn_bias)
        return x