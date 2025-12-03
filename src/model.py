import torch.nn as nn
from torchvision import models

def build_model(pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT #ImageNet weights
    else:
        weights = None

    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
