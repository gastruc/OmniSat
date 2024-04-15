import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Classifier(nn.Module):
    """
    Initialize a ResNet for feature extraction
    """
    def __init__(self, num_classes=15, modalities=[], reduce=False):
        super(ResNet50Classifier, self).__init__()
        # Load pre-trained ResNet-50 model
        self.modality = modalities[0]
        self.reduce = reduce
        self.resnet50 = models.resnet50(pretrained=True)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        if self.reduce:
            return self.resnet50(x[self.modality][:, 1:, :, :])
        return self.resnet50(x[self.modality])