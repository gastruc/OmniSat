from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from models.networks.encoder.utils.satmae_helpers import get_2d_sincos_pos_embed
from models.networks.encoder.ScaleViT import ScaleVitEncoder

class ScaleMAE(ScaleVitEncoder):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=8, spatial_mask=False,
                 channel_groups=[[0,1,2,3]],
                 channel_group_gsds=[0.1],
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, decoder_channel_embed=128,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, modalities=[]):
        
        in_c = len([item for sublist in channel_groups for item in sublist])
        # # This inits the encoder specific details
        super().__init__(img_size=img_size, modalities=modalities,
                         patch_size=patch_size, in_chans=in_c, embed_dim=embed_dim,
                         channel_groups=channel_groups, channel_group_gsds=channel_group_gsds, num_heads=num_heads, depth=depth,
                         mlp_ratio=mlp_ratio, norm_layer=norm_layer)

        self.modality = modalities[0]
        self.in_c = in_c
        self.patch_size = patch_size
        self.channel_groups = channel_groups
        self.spatial_mask = spatial_mask  # Whether to mask all channels of same spatial location
        num_groups = len(channel_groups)
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        num_patches = self.patch_embed[0].num_patches
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim - decoder_channel_embed),
            requires_grad=False)  # fixed sin-cos embedding
        # Extra channel for decoder to represent special place for cls token
        self.decoder_channel_embed = nn.Parameter(torch.zeros(1, num_groups + 1, decoder_channel_embed),
                                                  requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.ModuleList([nn.Linear(decoder_embed_dim, len(group) * patch_size**2)
                                           for group in channel_groups])
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed[0].num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed[0].num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        for patch_embed in self.patch_embed:
            w = patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, p, c):
        """
        imgs: (N, C, H, W)
        p: Patch embed patch size
        c: Num channels
        x: (N, L, C*patch_size**2)
        """
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # c = self.in_c
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, p, c):
        """
        x: (N, L, C*patch_size**2)
        p: Patch embed patch size
        c: Num channels
        imgs: (N, C, H, W)
        """
        # c = self.in_c
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, c, p, p))
        x = torch.einsum('nhwcpq->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x is (N, C, H, W)
        b, _, h, w = x.shape

        x_c_embed = []
        current = 0
        for i, group in enumerate(self.channel_groups):
            # The order of each group is just the order of x
            interval = torch.arange(current, current+len(group))
            current += len(group)
            patch = self.patch_embed[i](x[:,interval,:,:])
            x_c_embed.append(patch)  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        pos_embed = self.pos_embed[:, :, 1:, :]

        x = x + pos_embed

        if self.spatial_mask:
            # Mask spatial location across all channels (i.e. spatial location as either all/no channels)
            x = x.permute(0, 2, 1, 3).reshape(b, L, -1)  # (N, L, G*D)
            x, mask, ids_restore = self.random_masking(x, mask_ratio)  # (N, 0.25*L, G*D)
            x = x.view(b, x.shape[1], G, D).permute(0, 2, 1, 3).reshape(b, -1, D)  # (N, 0.25*G*L, D)
            mask = mask.repeat(1, G)  # (N, G*L)
            mask = mask.view(b, G, L)
        else:
            # Independently mask each channel (i.e. spatial location has subset of channels visible)
            x, mask, ids_restore = self.random_masking(x.view(b, -1, D), mask_ratio)  # (N, 0.25*G*L, D)
            mask = mask.view(b, G, L)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (N, G*L + 1, D)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # (N, 1 + G*0.25*L, D)

        # append mask tokens to sequence
        G = len(self.channel_groups)
        if self.spatial_mask:
            N, L = ids_restore.shape

            x_ = x[:, 1:, :].view(N, G, -1, x.shape[2]).permute(0, 2, 1, 3)  # (N, 0.25*L, G, D)
            _, ml, _, D = x_.shape
            x_ = x_.reshape(N, ml, G * D)  # (N, 0.25*L, G*D)

            mask_tokens = self.mask_token.repeat(N, L - ml, G)
            x_ = torch.cat((x_, mask_tokens), dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))  # (N, L, G*D)
            x_ = x_.view(N, L, G, D).permute(0, 2, 1, 3).reshape(N, -1, D)  # (N, G*L, D)
            x = torch.cat((x[:, :1, :], x_), dim=1)  # append cls token  (N, 1 + G*L, D)
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  (N, 1 + c*L, D)

        """
        x = x + self.decoder_pos_embed  # (N, 1+G*L, D)
        """

        # add pos and channel embed
        channel_embed = self.decoder_channel_embed[:, :-1, :].unsqueeze(2)  # (1, G, 1, cD)
        pos_embed = self.decoder_pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)

        channel_embed = channel_embed.expand(-1, -1, pos_embed.shape[2], -1)  # (1, G, L, cD)
        pos_embed = pos_embed.expand(-1, channel_embed.shape[1], -1, -1)  # (1, G, L, pD)
        pos_channel = torch.cat((pos_embed, channel_embed), dim=-1)  # (1, G, L, D)
        pos_channel = pos_channel.view(1, -1, pos_channel.shape[-1])  # (1, G*L, D)

        extra = torch.cat((self.decoder_pos_embed[:, :1, :],
                           self.decoder_channel_embed[:, -1:, :]), dim=-1)  # (1, 1, D)

        pos_channel = torch.cat((extra, pos_channel), dim=1)  # (1, 1+G*L, D)
        x = x + pos_channel  # (N, 1+G*L, D)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # remove cls token
        x = x[:, 1:, :]

        # Separate channel axis
        N, GL, D = x.shape
        x = x.view(N, G, GL//G, D)

        # predictor projection
        x_c_patch = []
        for i, group in enumerate(self.channel_groups):
            x_c = x[:, i]  # (N, L, D)
            dec = self.decoder_pred[i](x_c)  # (N, L, g_c * p^2)
            dec = dec.view(N, x_c.shape[1], -1, int(self.patch_size**2))  # (N, L, g_c, p^2)
            dec = torch.einsum('nlcp->nclp', dec)  # (N, g_c, L, p^2)
            x_c_patch.append(dec)

        x = torch.cat(x_c_patch, dim=1)  # (N, c, L, p**2)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, c, H, W]
        pred: [N, L, c*p*p]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs, self.patch_embed[0].patch_size[0], self.in_c)  # (N, L, C*P*P)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        N, L, _ = target.shape
        target = target.view(N, L, self.in_c, -1)  # (N, L, C, p^2)
        target = torch.einsum('nlcp->nclp', target)  # (N, C, L, p^2)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, C, L], mean loss per patch

        total_loss, num_removed = 0., 0.
        current = 0
        for i, group in enumerate(self.channel_groups):
            interval = torch.arange(current, current+len(group))
            # #ce que je rajoute
            # batch_mask, _ = torch.max(target[:, interval, :].flatten(start_dim=1), dim=1)
            # mask_final = (batch_mask != 0).unsqueeze(-1).expand(N, mask.shape[2]) * mask[:, i]
            # import ipdb; ipdb.set_trace()
            #
            current += len(group)
            group_loss = loss[:, interval, :].mean(dim=1)  # (N, L)
            # total_loss += (group_loss * mask_final).sum()
            # num_removed += mask_final.sum()  # mean loss on removed patches
            total_loss += (group_loss * mask[:, i]).sum()
            num_removed += mask[:, i].sum()  # mean loss on removed patches

        return total_loss/num_removed

    def forward(self, imgs, mask_ratio, spectral_mask_ratio):

        # Group the images here
        x_c = []
        for i, group in enumerate(self.channel_groups):
            for band in group:
                x_c.append(imgs[self.modality][:, band].unsqueeze(1))
        x_c = torch.cat(x_c, dim=1)
        imgs = x_c

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, C, L, p*p]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
