import torch.nn.functional as F

import torch.nn as nn
import torch

class FullyConnectedNetwork(nn.Module):
    """Multi-layer perceptron model. Bascially just uses fully connected
    layers rather than 3x3 convs. Only can work with very small images,
    otherwise the number of weights in the model will be way too high given
    that each layer is fully connected.
    """
    def __init__(self, input_size, n_bands, p_drop = 0.3, n_class=0, modalities=[]):
        super().__init__()
        self.modality = modalities[0]
        self.fc1 = nn.Linear(input_size*input_size*n_bands, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512, 512)

        self.dropout = nn.Dropout(p_drop)
        
        # set n_class to 0 if we want headless model
        self.n_class = n_class
        if n_class:
            self.head = nn.Sequential(
                                  nn.Linear(512, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p = p_drop),
                                  nn.Linear(1024, n_class)
            )
        
        
    def forward(self,x):
        # flatten image input
        x = x[self.modality]
        _, c, h, w = x.shape
        x = x.view(-1, c*h*w) # [batch, c*h*w]
        
        x = F.relu(self.fc1(x)) # [batch, 512]
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) # [batch, 512]
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) # [batch, 512]
        
        if self.n_class:
            x = self.head(x)
        
        return x
