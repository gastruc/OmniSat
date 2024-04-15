from torch import nn
from hydra.utils import instantiate

class MonoModal(nn.Module):
    """
    Initialize encoder and mlp
    Args:
        encoder (nn.Module): that encodes data
        omni (bool): If True, takes class token as encoding of data
        mlp (nn.Module): that returns prediction
    """
    def __init__(
        self, 
        encoder, 
        mlp,
        modalities: list = [],
        omni: bool = False):
        super().__init__()
        self.encoder = encoder
        self.omni = omni
        self.mlp = mlp.instance

    def forward(self, x):
        """
        Forward pass of the network
        """
        out = self.encoder(x)
        if self.omni:
            out = out[:, 0]
        out = self.mlp(out)
        return out