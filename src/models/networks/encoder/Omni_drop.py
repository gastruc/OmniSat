from typing import Callable, Optional
from functools import partial

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, PatchDropout

from models.networks.encoder.utils.utils_ViT import RPEBlock, CrossRPEBlock


class OmniModule(nn.Module):
    """
    Omni module associated to MAEOmni_drop pretraining with token dropping
    """
    def __init__(self, 
                 projectors: dict = {},
                 modalities: list = [],
                 num_patches: int = 0,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale = None,
                 class_token: bool = True,
                 pre_norm: bool = False,
                 drop_rate: float = 0.,
                 pos_drop_rate: float = 0.,
                 patch_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 norm_layer: Optional[Callable] = None,
                 ):
        super(OmniModule, self).__init__()
        self.modalities = modalities

        self.num_prefix_tokens = 1 if class_token else 0
        self.num_patches = num_patches + self.num_prefix_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

        for i in range(len(modalities)):
            if modalities[i].split('-')[-1] == 'mono':
                m = '-'.join(modalities[i].split('-')[:-1])
            else:
                m = modalities[i]
            setattr(self, '_'.join(['projector', modalities[i]]), projectors[m])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            RPEBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)] )
        self.cross_block = CrossRPEBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, modalities=modalities,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_patches=self.num_patches)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def forward_proj(self, x):
        tokens = []
        out = {}
        for modality in self.modalities:
            if modality == "aerial":
                out['_'.join(['tokens', modality])], out['indices'], out['sizes'] = getattr(self, '_'.join(['projector', modality]))(x[modality])
            elif modality.split('-')[-1] == 'mono':
                sentinel_tokens, out['_'.join(['attention', modality])] = getattr(
                    self, '_'.join(['projector', modality]))(x[modality].unsqueeze(1), torch.zeros(x[modality].shape[0], 1).to(x[modality].device) + 120)
                out['_'.join(['tokens', modality])] = sentinel_tokens.view(sentinel_tokens.shape[0], sentinel_tokens.shape[1], -1).permute(0, 2, 1)
            else:
                out['_'.join(["dates", modality])] = x['_'.join([modality, "dates"])]
                sentinel_tokens, out['_'.join(['attention', modality])] = getattr(
                    self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])])
                out['_'.join(['tokens', modality])] = sentinel_tokens.view(sentinel_tokens.shape[0], sentinel_tokens.shape[1], -1).permute(0, 2, 1)
            tokens.append(out['_'.join(['tokens', modality])]  + self.pos_embed[:, 1:, :])
        tokens = torch.cat(tokens, dim=1)
        return tokens, out
    
    def forward_transformer(self, x, mask):
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, x), dim=1)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.cross_block(tokens, mask)
        return tokens

    def forward(self, x):
        """
         Args:
            x: dict that contains
        """
        tokens = []
        for modality in self.modalities:
            if modality == "aerial":
                token, _, _ = getattr(self, '_'.join(['projector', modality]))(x[modality])
            elif modality.split('-')[-1] == 'mono':
                token, _ = getattr(
                    self, '_'.join(['projector', modality]))(x[modality].unsqueeze(1), torch.zeros(x[modality].shape[0], 1).to(x[modality].device) + 120)
                token = token.view(token.shape[0], token.shape[1], -1).permute(0, 2, 1)
            else:
                token, _ = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])])
                token = token.view(token.shape[0], token.shape[1], -1).permute(0, 2, 1)
            tokens.append(token + self.pos_embed[:, 1:, :])
            
        tokens = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(token.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.cross_block(tokens)
        return tokens