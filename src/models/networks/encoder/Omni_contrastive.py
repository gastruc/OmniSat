import torch
import torch.nn as nn

class OmniModule(nn.Module):
    """
    Omni module with a contrastive only
    """
    def __init__(self, 
                 projectors: dict = {},
                 modalities: list = [],
                 ):
        super(OmniModule, self).__init__()
        self.modalities = modalities

        for i in range(len(modalities)):
            if modalities[i].split('-')[-1] == 'mono':
                m = '-'.join(modalities[i].split('-')[:-1])
            else:
                m = modalities[i]
            setattr(self, '_'.join(['projector', modalities[i]]), projectors[m])

    def forward(self, x):
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
        return out, []
    
