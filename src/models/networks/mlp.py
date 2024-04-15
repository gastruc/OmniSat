import torch
from torch import nn

class MLP(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
    ):
        """
        Initializes an MLP Classification Head
        Args:
            initial_dim (int): dimension of input layer
            hidden_dim (list): list of hidden dimensions for the MLP
            final_dim (int): dimension of output layer
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        dim = [initial_dim] + hidden_dim + [final_dim]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                args.append(norm(dim[i + 1]))
                #args.append(norm(4, dim[i + 1]))
                args.append(activation())
        return args

    def forward(self, x):
        """
        Predicts output
        Args:
            x: torch.Tensor with features
        """
        return self.mlp(x)
    
class Identity(nn.Module):
    def __init__(
        self,
    ):
        """
        Initializes a module that computes Identity
        """
        super().__init__()

    def forward(self, x):
        """
        Computes Identity
        """
        return x