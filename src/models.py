"""Model architectures: two CNNs written from scratch (plain and residual), plus a ResNet18 baseline."""

import torch
from torch import nn
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


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection: out = ReLU(F(x) + shortcut(x)).

    With stride 2 the block halves the resolution; a 1x1 convolution then reshapes the shortcut
    so it can be added to F(x).
    """

    def __init__(self, c_in, c_out, stride=1, use_batchnorm=True):
        super().__init__()

        def norm(c):
            return nn.BatchNorm2d(c) if use_batchnorm else nn.Identity()

        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=not use_batchnorm),
            norm(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=not use_batchnorm),
            norm(c_out),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=not use_batchnorm), norm(c_out)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.shortcut(x))


class ResidualCNN(nn.Module):
    """Small ResNet-style CNN written from scratch (second from-scratch architecture).

    A stem convolution, then `depth` stages of one residual block each; every stage after the
    first halves the resolution and doubles the channels. Skip connections let gradients flow
    straight through, so deeper versions train as easily as shallow ones.
    """

    def __init__(
        self, num_classes=NUM_CLASSES, base_channels=32, depth=4, dropout=0.3, use_batchnorm=True
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(base_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        stages, c_in = [], base_channels
        for i in range(depth):
            c_out = base_channels * 2**i
            stages.append(ResidualBlock(c_in, c_out, stride=1 if i == 0 else 2, use_batchnorm=use_batchnorm))
            c_in = c_out
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_in, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


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
    """arch: "cnn" (PlantCNN), "rescnn" (ResidualCNN) or "resnet18" (pretrained baseline)."""
    if arch == "cnn":
        return PlantCNN(**kwargs)
    if arch == "rescnn":
        return ResidualCNN(**kwargs)
    if arch == "resnet18":
        return resnet18_finetune(
            num_classes=kwargs.get("num_classes", NUM_CLASSES),
            pretrained=kwargs.get("pretrained", True),
            freeze_backbone=kwargs.get("freeze_backbone", False),
        )
    raise ValueError(f"Unknown architecture: {arch}")


def get_device():
    """Best available device: NVIDIA GPU (ClearML worker), Apple GPU (Mac), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
