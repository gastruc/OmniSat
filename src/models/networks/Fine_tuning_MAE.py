import torch
import torch.nn as nn

class Fine(nn.Module):
    """
    Initialize Fine Tuning of SatMAE/ScaleMAE after pretraining
    Args:
        encoder (torch.nn.Module): initialized model
        path (str): path of checkpoint of model to load
        output_size (int): size of output returned by encoder
        inter_dim (list): list of hidden dims of mlp after encoder
        p_drop (float): dropout parameter of mlp after encoder
        name (str): name of the weights from checkpoint to use
        freeze (bool); if True, freeze encoder to perform linear probing
        n_class (int): output_size of mlp
        pooling_method (str): type of pooling of tokens after transformer
        modalities (list): list of modalities to use
        last_block (bool): if True freeze all encoder except last block of transformer
        proj_only (bool): if True, load only weights from projectors
    """
    def __init__(self, 
                 encoder: torch.nn.Module,
                 path: str = '',
                 output_size: int = 256,
                 inter_dim: list = [],
                 p_drop: float = 0.3,
                 name: str = 'encoder',
                 freeze: bool = True,
                 n_class: int = 15,
                 pooling_method: str = 'token',
                 modalities: list = [],
                 last_block: bool = False,
                ):
        super().__init__()

        self.size = output_size
        self.freeze = freeze
        self.global_pool = pooling_method
        self.modality = modalities[0]

        for i in range(len(modalities)):
            if modalities[i].split('-')[-1] == 'mono':
                modalities[i] = '-'.join(modalities[i].split('-')[:-1])
        
        u = torch.load(path)
        d = {}
        for key in u["state_dict"].keys():
            if not("decoder" in key):
                d['.'.join(key.split('.')[1:])] = u["state_dict"][key]

        del(d['mask_token'])

        encoder.load_state_dict(d)        
        self.model = encoder

        if last_block:
            model_parameters = self.model.named_parameters()
            for name, param in model_parameters:
                if len(name.split(".")) > 1:
                    if name.split(".")[1] == "5":
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

        if self.freeze:
            for param in self.model.parameters():
                    param.requires_grad = False

        
        # set n_class to 0 if we want headless model
        self.n_class = n_class
        if n_class:
            if len(inter_dim) > 0:
                layers = [nn.Linear(self.size, inter_dim[0])]
                layers.append(nn.BatchNorm1d(inter_dim[0]))
                layers.append(nn.Dropout(p = p_drop))
                layers.append(nn.ReLU())
                for i in range(len(inter_dim) - 1):
                    layers.append(nn.Linear(inter_dim[i], inter_dim[i + 1]))
                    layers.append(nn.BatchNorm1d(inter_dim[i + 1]))
                    layers.append(nn.Dropout(p = p_drop))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(inter_dim[-1], n_class))
            else:
                layers = [nn.Linear(self.size, n_class)]
            self.head = nn.Sequential(*layers)
        
    def forward(self,x):
        """
        Forward pass of the network. Perform pooling of tokens after transformer 
        according to global_pool argument.
        """
        x = self.model(x[self.modality])
        if self.global_pool:
            if self.global_pool == 'avg':
                x = x[:, 1:].mean(dim=1)
            elif self.global_pool == 'max':
                x ,_ = torch.max(x[:, 1:],1)
            else:
                x = x[:, 0]
        if self.n_class:
            x = self.head(x)   
        return x


if __name__ == "__main__":
    _ = Fine()