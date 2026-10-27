"""Model architectures: a CNN written from scratch, plus an optional ResNet18 baseline."""

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 39


def conv_block(c_in, c_out, use_batchnorm=True):
    """Conv -> (BatchNorm) -> ReLU -> MaxPool. Halves the resolution."""
    layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(c_out))
    layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


class PlantCNN(nn.Module):
    """Simple CNN: channels double and resolution halves at every block.

    Hyperparameters worth exposing in ClearML: depth, base_channels, dropout, use_batchnorm.
    """

    def __init__(
        self, num_classes=NUM_CLASSES, base_channels=32, depth=4, dropout=0.3, use_batchnorm=True
    ):
        super().__init__()
        blocks = []
        c_in = 3
        for i in range(depth):
            c_out = base_channels * 2**i
            blocks.append(conv_block(c_in, c_out, use_batchnorm))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        # Global average pooling keeps the head small whatever the input resolution.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_in, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def resnet18_finetune(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    """Transfer-learning baseline: ImageNet ResNet18 with a new classification layer."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(arch, **kwargs):
    if arch == "cnn":
        return PlantCNN(**kwargs)
    if arch == "resnet18":
        return resnet18_finetune(
            num_classes=kwargs.get("num_classes", NUM_CLASSES),
            pretrained=kwargs.get("pretrained", True),
            freeze_backbone=kwargs.get("freeze_backbone", False),
        )
    raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
