import logging
from typing import List, Optional, Callable

import torch
from torch import nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

class TransposedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gp_norm, stride=[1, 1], upsample=None):
        super(TransposedResidualBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(
            out_channels, in_channels, kernel_size=3, stride=stride[1], 
            padding=1, output_padding=stride[1] - 1, bias=False
        )

        self.conv2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=3, stride=stride[0], 
            padding=1, output_padding=stride[0] - 1, bias=False
        )

        self.bn = nn.GroupNorm(gp_norm, out_channels)
        self.upsample = upsample

    def forward(self, x):
        residual = x
        if self.upsample is not None:
            residual = self.upsample(residual)

        out = self.conv1(x)
        out = self.bn(out)
        out = F.gelu(out)

        out = self.conv2(out)
        out = self.bn(out)

        out = out + residual
        out = F.gelu(out)

        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gp_norm, stride=[1, 1], downsample=None):
        """
        A basic residual block of ResNet
        Parameters
        ----------
            in_channels: Number of channels that the input have
            out_channels: Number of channels that the output have
            stride: strides in convolutional layers
            downsample: A callable to be applied before addition of residual mapping
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride[0], 
            padding=1, bias=False
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride[1], 
            padding=1, bias=False
        )

        #self.bn = nn.BatchNorm2d(out_channels, momentum=1/gp_norm)
        self.bn = nn.GroupNorm(gp_norm, out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        if(self.downsample is not None):
            residual = self.downsample(residual)

        out = F.gelu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out = out + residual
        out = F.gelu(out)
        return out

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            res: bool = False,
            gp_norm: int = 4
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        bias = not bias
        if res:
            self.proj = nn.ModuleList([nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1, bias=True),
                            nn.BatchNorm2d(embed_dim),
                            nn.GELU(),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, gp_norm, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, gp_norm, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, gp_norm, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)])
            self.max_pools = [False, False, False, True, False, True, False, True, False, True]
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.res = res

    def forward(self, x):
        B, C, H, W = x.shape
        if self.res:
            grid_size = (H // self.patch_size[0], W // self.patch_size[1])
            num_patches = grid_size[0] * grid_size[1]
            x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
            x = x.flatten(2, 3)
            x = torch.permute(x,(0,2,1,3,4))
            x = x.flatten(0,1)
            indices = []
            sizes = []
            for i in range (len(self.proj)):
                sizes.insert(0, x.shape)
                if self.max_pools[i]:
                    x, indice = self.proj[i](x)
                    indices.insert(0, indice)
                else:
                    x = self.proj[i](x)
            x = torch.reshape(x, (B, num_patches, self.embed_dim, 1, 1)).squeeze()
            return x, indices, sizes
        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            x = self.norm(x)
        return x
    
class PatchEmbed1(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            res: bool = False,
            gp_norm: int = 4
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        bias = not bias
        if res:
            self.proj = nn.ModuleList([nn.Conv2d(in_chans, 64, kernel_size=2, stride=1, padding=2, bias=True),
                            nn.BatchNorm2d(64),
                            nn.GELU(),
                            nn.Conv2d(64, embed_dim, kernel_size=2, stride=2, bias=True),
                            nn.BatchNorm2d(embed_dim),
                            nn.GELU(),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)])
            self.max_pools = [False, False, False, False, False, False, True, False, True, False, True, False, True]
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.res = res

    def forward(self, x):
        B, C, H, W = x.shape
        if self.res:
            grid_size = (H // self.patch_size[0], W // self.patch_size[1])
            num_patches = grid_size[0] * grid_size[1]
            x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
            x = x.flatten(2, 3)
            x = torch.permute(x,(0,2,1,3,4))
            x = x.flatten(0,1)
            indices = []
            sizes = []
            for i in range (len(self.proj)):
                sizes.insert(0, x.shape)
                if self.max_pools[i]:
                    x, indice = self.proj[i](x)
                    indices.insert(0, indice)
                else:
                    x = self.proj[i](x)
            x = torch.reshape(x, (B, num_patches, self.embed_dim, 1, 1)).squeeze()
        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            x = self.norm(x)
        return x
    
class PatchEmbedPastis(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            res: bool = False,
            gp_norm: int = 4
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        bias = not bias
        if res:
            self.proj = nn.ModuleList([nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1, bias=True),
                            nn.BatchNorm2d(embed_dim),
                            nn.GELU(),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, gp_norm, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, gp_norm, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, gp_norm, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ])
            self.max_pools = [False, False, False, True, False, True, False, True, False, True]
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.res = res

    def forward(self, x):
        B, C, H, W = x.shape
        if self.res:
            grid_size = (H // self.patch_size[0], W // self.patch_size[1])
            num_patches = grid_size[0] * grid_size[1]
            x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
            x = x.flatten(2, 3)
            x = torch.permute(x,(0,2,1,3,4))
            x = x.flatten(0,1)
            indices = []
            sizes = []
            for i in range (len(self.proj)):
                sizes.insert(0, x.shape)
                if self.max_pools[i]:
                    x, indice = self.proj[i](x)
                    indices.insert(0, indice)
                else:
                    x = self.proj[i](x)
            x = torch.reshape(x, (B, num_patches, self.embed_dim, 1, 1)).squeeze()
            return x, indices, sizes
        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            x = self.norm(x)
        return x
    
class PatchEmbed2(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            res: bool = False,
            gp_norm: int = 4
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        bias = not bias
        if res:
            self.proj = nn.ModuleList([nn.Conv2d(in_chans, 64, kernel_size=3, stride=3, padding=2, bias=True),
                            nn.BatchNorm2d(64),
                            nn.GELU(),
                            nn.Conv2d(64, embed_dim, kernel_size=2, stride=2, bias=True),
                            nn.BatchNorm2d(embed_dim),
                            nn.GELU(),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True)])
            self.max_pools = [False, False, False, False, False, False, True, False, True, False, True]
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.res = res

    def forward(self, x):
        B, C, H, W = x.shape
        if self.res:
            grid_size = (H // self.patch_size[0], W // self.patch_size[1])
            num_patches = grid_size[0] * grid_size[1]
            x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
            x = x.flatten(2, 3)
            x = torch.permute(x,(0,2,1,3,4))
            x = x.flatten(0,1)
            indices = []
            sizes = []
            for i in range (len(self.proj)):
                sizes.insert(0, x.shape)
                if self.max_pools[i]:
                    x, indice = self.proj[i](x)
                    indices.insert(0, indice)
                else:
                    x = self.proj[i](x)
            x = torch.reshape(x, (B, num_patches, self.embed_dim, 1, 1)).squeeze()
        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            x = self.norm(x)
        return x
    
class PatchEmbed3(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            res: bool = False,
            gp_norm: int = 4
    ):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        bias = not bias
        if res:
            self.proj = nn.ModuleList([nn.Conv2d(in_chans, 64, kernel_size=2, stride=1, padding=1, bias=True),
                            nn.BatchNorm2d(64),
                            nn.GELU(),
                            nn.Conv2d(64, embed_dim, kernel_size=2, stride=1, bias=True),
                            nn.BatchNorm2d(embed_dim),
                            nn.GELU(),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True),
                            ResidualBlock(embed_dim, embed_dim, 4, stride=[1, 1]),
                            nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True)])
            self.max_pools = [False, False, False, False, False, False, True, False, True, False, True, False, True, False, True]
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.res = res

    def forward(self, x):
        B, C, H, W = x.shape
        if self.res:
            grid_size = (H // self.patch_size[0], W // self.patch_size[1])
            num_patches = grid_size[0] * grid_size[1]
            x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
            x = x.flatten(2, 3)
            x = torch.permute(x,(0,2,1,3,4))
            x = x.flatten(0,1)
            indices = []
            sizes = []
            for i in range (len(self.proj)):
                sizes.insert(0, x.shape)
                if self.max_pools[i]:
                    x, indice = self.proj[i](x)
                    indices.insert(0, indice)
                else:
                    x = self.proj[i](x)
            x = torch.reshape(x, (B, num_patches, self.embed_dim, 1, 1)).squeeze()
        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            x = self.norm(x)
        return x