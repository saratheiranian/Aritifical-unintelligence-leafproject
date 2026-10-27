import pytest
import torch

from src.models import PlantCNN, build_model, count_parameters


@pytest.mark.parametrize("img_size", [64, 128, 256])
def test_cnn_output_shape(img_size):
    model = PlantCNN(num_classes=39)
    assert model(torch.randn(2, 3, img_size, img_size)).shape == (2, 39)


def test_cnn_without_batchnorm_has_fewer_parameters():
    with_bn = count_parameters(PlantCNN(use_batchnorm=True))
    without_bn = count_parameters(PlantCNN(use_batchnorm=False))
    assert without_bn < with_bn


def test_build_model_rejects_unknown_arch():
    with pytest.raises(ValueError):
        build_model("vgg")


def test_resnet18_head_has_39_outputs():
    model = build_model("resnet18", pretrained=False)
    assert model(torch.randn(1, 3, 128, 128)).shape == (1, 39)
