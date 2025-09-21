# backend/model_breast.py
import torch
import torch.nn as nn
import torchvision.models as models

class BreastResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        # if pretrained==True, it uses torchvision weights
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)






