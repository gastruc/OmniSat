import torch
import torch.nn as nn

class Train_UTxT_classif(nn.Module):
    def __init__(self, 
                 encoder: torch.nn.Module,
                 output_size: dict = {},
                 inter_dim: list = [],
                 p_drop: float = 0.3,
                 num_classes: int = 15,
                 modalities: list = []
                ):
        super().__init__()

        self.modalities = modalities
        self.sizes = output_size
          
        self.encoder = encoder
        if "aerial" in modalities:
            del self.encoder.arch_vhr.segmentation_head
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.encoder.to(device)

        # set n_class to 0 if we want headless model
        self.n_class = num_classes
        self.heads = {}
        for i, m in enumerate(modalities):
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
                layers.append(nn.Linear(inter_dim[-1], self.n_class))
                setattr(self, '_'.join(['head', m]), nn.Sequential(*layers))
            else:
                layers =  nn.Linear(self.sizes[m], self.n_class)
                setattr(self, '_'.join(['head', m]), nn.Sequential(layers))
            if torch.cuda.is_available():
                getattr(self, '_'.join(['head', m])).to(device)

    def forward(self, x):
        x = self.encoder(x)
        out = {}
        for m in self.modalities:
            x_m = x[m]
            x_m = x_m.mean(dim=(2, 3))
            if self.n_class:
                x_m = getattr(self, '_'.join(['head', m]))(x_m)
                out[m] = x_m
        return out


if __name__ == "__main__":
    _ = Train_UTxT_classif()