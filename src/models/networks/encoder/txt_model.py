"""
U-T&T Implementation
Author: Anatol Garioud (https://github.com/IGNF/FLAIR-2/tree/main/src)
Modified by Nicolas Gonthier to also inject S1
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import segmentation_models_pytorch as smp  

from models.networks.encoder.utils.utae_model import UTAE
from models.networks.encoder.utils.fusion_utils import *

    
class TimeTexture_flair(nn.Module):
    """ 
     U-Tae implementation for Sentinel-2 super-patch;
     U-Net smp implementation for aerial imagery;
     Added U-Tae feat maps as attention to encoder featmaps of unet.
    """      
    
    def __init__(self, modalities=["aerial"], sat_patch_size= 6, drop_utae_modality= 0.0, aerial_metadata= False, geo_enc_size= 32,
                 num_channels_aerial= 4, num_channels_sat= 10, encoder_weights='None',num_classes = 1,
                 num_channels_s1= 2, encoder_widths= [64,64,128,128], decoder_widths= [64,64,128,128], out_conv= [32, 13], str_conv_k= 4, str_conv_s= 2,
                 str_conv_p= 1, agg_mode= 'att_group', encoder_norm= 'group', n_head= 16, d_model= 256, d_k= 4, pad_value=0, padding_mode= "reflect", 
                 output_segsem=None, segsem=False, remove_decoder_vhr=False
                 ):
        
        super(TimeTexture_flair, self).__init__()   
        
        self.sat_patch_size = sat_patch_size
        self.modalities = modalities
        self.num_classes = num_classes 
        self.segsem = segsem
        self.remove_decoder_vhr = remove_decoder_vhr
        
        if "aerial" in self.modalities: 
            if encoder_weights=='None':
                encoder_weights = None 
            self.arch_vhr = smp.create_model(
                                            arch="unet", 
                                            encoder_name="resnet34", 
                                            classes=num_classes, 
                                            in_channels=num_channels_aerial,
                                            encoder_weights=encoder_weights
                                            )

        if "s2" in self.modalities:
            self.arch_hr  = UTAE(
                                input_dim=num_channels_sat,
                                encoder_widths=encoder_widths, 
                                decoder_widths=decoder_widths,
                                out_conv=out_conv,
                                str_conv_k=str_conv_k,
                                str_conv_s=str_conv_s,
                                str_conv_p=str_conv_p,
                                agg_mode=agg_mode, 
                                encoder_norm=encoder_norm,
                                n_head=n_head, 
                                d_model=d_model, 
                                d_k=d_k,
                                encoder=False,
                                return_maps=True,
                                pad_value=pad_value,
                                padding_mode=padding_mode,
                                )
        
        if "s1-asc" in self.modalities:
            self.arch_s1  = UTAE(
                                input_dim=num_channels_s1,
                                encoder_widths=encoder_widths, 
                                decoder_widths=decoder_widths,
                                out_conv=out_conv,
                                str_conv_k=str_conv_k,
                                str_conv_s=str_conv_s,
                                str_conv_p=str_conv_p,
                                agg_mode=agg_mode, 
                                encoder_norm=encoder_norm,
                                n_head=n_head, 
                                d_model=d_model, 
                                d_k=d_k,
                                encoder=False,
                                return_maps=True,
                                pad_value=pad_value,
                                padding_mode=padding_mode,
                                )
        
        if "s2" in self.modalities and "aerial" in self.modalities:
            self.fm_utae_featmap_cropped = FM_cropped(self.arch_hr.encoder_widths[0], 
                                                        list(self.arch_vhr.encoder.out_channels),
                                                        )
            self.fm_utae_featmap_collapsed = FM_collapsed(self.arch_hr.encoder_widths[0], 
                                                            list(self.arch_vhr.encoder.out_channels),
                                                            )
        
        if "s1-asc" in self.modalities and "aerial" in self.modalities:
            self.fm_utae_featmap_cropped_s1 = FM_cropped(self.arch_s1.encoder_widths[0], 
                                                        list(self.arch_vhr.encoder.out_channels),
                                                        )
            self.fm_utae_featmap_collapsed_s1 = FM_collapsed(self.arch_s1.encoder_widths[0], 
                                                            list(self.arch_vhr.encoder.out_channels),
                                                            ) 
            
        if aerial_metadata == True:
            i = 512; last_spatial_dim = int([(i:=i/2) for u in range(len(self.arch_vhr.encoder.out_channels)-1)][-1])
            self.mtd_mlp = mtd_encoding_mlp(geo_enc_size+13, last_spatial_dim)
        

        #self.reshape_utae_output = nn.Sequential(nn.Upsample(size=(200,200), mode='nearest'),
        #                                         #nn.Conv2d(self.arch_hr.encoder_widths[0], num_classes, 1) 
        #                                        )
        
        self.aerial_metadata = aerial_metadata
        self.drop_utae_modality = drop_utae_modality

        # S1 and S2 segmentation heads
        if self.segsem: 
            if "s2" in self.modalities:
                self.segsem_head_s2 = nn.Sequential(nn.Conv2d(self.arch_hr.encoder_widths[0], num_classes, 1)) # nn.Upsample(size=(512,512), mode='nearest'),
            if "s1-asc" in self.modalities:
                self.segsem_head_s1 = nn.Sequential(nn.Conv2d(self.arch_s1.encoder_widths[0], num_classes, 1))
            if "aerial" in self.modalities:
                self.reshape_unet = nn.Sequential(nn.AdaptiveAvgPool2d(output_segsem))

            
    def forward(self, x, metadata=None):
        
        out = {}
        
        if "aerial" in self.modalities:
            bpatch = x["aerial"]
            unet_fmaps_enc = self.arch_vhr.encoder(bpatch)  ### unet feature maps           
        
        ### aerial metadatat encoding and adding to u-net feature maps
        if self.aerial_metadata == True:
            x_enc = self.mtd_mlp(metadata)
            x_enc = x_enc.unsqueeze(1).unsqueeze(-1).repeat(1,unet_fmaps_enc[-1].size()[1],1,unet_fmaps_enc[-1].size()[-1])
            unet_fmaps_enc[-1] = torch.add(unet_fmaps_enc[-1], x_enc) 
        
        ### cropped fusion module
        if "s2" or "s1-asc" in self.modalities: transform = T.CenterCrop((self.sat_patch_size, self.sat_patch_size))
        
        # S2 case
        if "s2" in self.modalities:
            bspatch = x["s2"]
            _ , utae_fmaps_dec = self.arch_hr(bspatch, x["s2_dates"])  ### utae class scores and feature maps 

            if "aerial" in self.modalities:
                utae_last_fmaps_reshape_cropped = transform(utae_fmaps_dec[-1])    
                utae_last_fmaps_reshape_cropped = self.fm_utae_featmap_cropped(utae_last_fmaps_reshape_cropped, [i.size()[-1] for i in unet_fmaps_enc])       
                
                ### collapsed fusion module       
                utae_fmaps_dec_squeezed = torch.mean(utae_fmaps_dec[-1][0], dim=(-2,-1))
                utae_last_fmaps_reshape_collapsed = self.fm_utae_featmap_collapsed(utae_fmaps_dec_squeezed, [i.size()[-1] for i in unet_fmaps_enc])  ### reshape last feature map of utae to match feature maps enc. unet
                
                ### adding cropped/collasped
                utae_last_fmaps_reshape = [torch.add(i,j) for i,j in zip(utae_last_fmaps_reshape_cropped, utae_last_fmaps_reshape_collapsed)]

        #S1 case 
        if "s1-asc" in self.modalities:
            bs1patch = x["s1-asc"]
            _ , utae_fmaps_dec_s1 = self.arch_s1(bs1patch, x["s1-asc_dates"])  ### utae class scores and feature maps 

            if "aerial" in self.modalities:
                utae_last_fmaps_reshape_cropped_s1 = transform(utae_fmaps_dec_s1[-1])    
                utae_last_fmaps_reshape_cropped_s1 = self.fm_utae_featmap_cropped_s1(utae_last_fmaps_reshape_cropped_s1, [i.size()[-1] for i in unet_fmaps_enc])       
                
                ### collapsed fusion module       
                utae_fmaps_dec_squeezed_s1 = torch.mean(utae_fmaps_dec_s1[-1][0], dim=(-2,-1))
                utae_last_fmaps_reshape_collapsed_s1 = self.fm_utae_featmap_collapsed_s1(utae_fmaps_dec_squeezed_s1, [i.size()[-1] for i in unet_fmaps_enc])  ### reshape last feature map of utae to match feature maps enc. unet
                
                ### adding cropped/collasped
                utae_last_fmaps_reshape_s1 = [torch.add(i,j) for i,j in zip(utae_last_fmaps_reshape_cropped_s1, utae_last_fmaps_reshape_collapsed_s1)]

        ### modality fusion (and dropout)
        if "aerial" in self.modalities:
            if torch.rand(1) > self.drop_utae_modality and ("s2" in self.modalities or "s1-asc" in self.modalities):
                if "s2" in self.modalities and "s1-asc" in self.modalities:
                    utae_fmaps_s1_s2 = [torch.add(i,j) for i,j in zip(utae_last_fmaps_reshape, utae_last_fmaps_reshape_s1)]  ### add utae mask to unet feats map
                    unet_utae_fmaps = [torch.add(i,j) for i,j in zip(unet_fmaps_enc, utae_fmaps_s1_s2)] 
                elif "s2" in self.modalities:
                    unet_utae_fmaps = [torch.add(i,j) for i,j in zip(unet_fmaps_enc, utae_last_fmaps_reshape)]  ### add utae mask to unet feats map
                elif "s1-asc" in self.modalities:
                    unet_utae_fmaps = [torch.add(i,j) for i,j in zip(unet_fmaps_enc, utae_last_fmaps_reshape_s1)]  ### add utae mask to unet feats map
            else:
                unet_utae_fmaps = unet_fmaps_enc

            if not self.remove_decoder_vhr:
                unet_out = self.arch_vhr.decoder(*unet_utae_fmaps)  ### unet decoder
            else: 
                unet_out = unet_utae_fmaps[-1]   ### unet encoder output
            if self.segsem: 
                unet_out = self.arch_vhr.segmentation_head(unet_out) ### unet class scores 
                unet_out = self.reshape_unet(unet_out)
            out['aerial'] = unet_out

        if "s2" in self.modalities:
            utae_out = utae_fmaps_dec[-1] 
            if self.segsem:
                utae_out = self.segsem_head_s2(utae_out)
            out['s2'] = utae_out
        
        if "s1-asc" in self.modalities:
            utae_out_s1 = utae_fmaps_dec_s1[-1] 
            if self.segsem:
                utae_out_s1 = self.segsem_head_s1(utae_out_s1)
            out['s1-asc'] = utae_out_s1

        return out




