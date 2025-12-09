import torch.nn as nn
from torchvision import models
from transformers import AutoModelForImageClassification
from . import hyperparameters as hp

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = hp.NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # For an input of 224x224, after two 2x2 pooling layers, the size will be 224 / 2 / 2 = 56.
        # So the flattened size will be 32 * 56 * 56.
        self.fc1 = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

def build_model(pretrained: bool = hp.PRETRAINED, num_classes: int = hp.NUM_CLASSES) -> nn.Module:
    if hp.MODEL_ARCH == "resnet50":
        if hp.MODEL_SOURCE == "imagenet":
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT #ImageNet weights
            else:
                weights = None

            model = models.resnet50(weights=weights)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes) # Adjust for hp.NUM_CLASSES
        elif hp.MODEL_SOURCE == "hf_pretrained":
            # Load the model from Hugging Face
            model = AutoModelForImageClassification.from_pretrained(hp.HF_MODEL_PATH)
            if model.classifier.out_features != num_classes:
                raise ValueError(f"Hugging Face model output features ({model.classifier.out_features}) do not match hp.NUM_CLASSES ({num_classes})")
    elif hp.MODEL_ARCH == "simple_cnn":
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown MODEL_ARCH: {hp.MODEL_ARCH}")

    return model
