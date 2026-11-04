import pytest
import torch

from src.models import PlantCNN, build_model, count_parameters, get_device


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


def test_get_device_returns_a_usable_device():
    from src.models import get_device

    device = get_device()
    assert device.type in {"cuda", "mps", "cpu"}
    assert torch.zeros(1, device=device).device.type == device.type


def test_resnet18_freeze_backbone_only_trains_the_head():
    model = build_model("resnet18", pretrained=False, freeze_backbone=True, num_classes=5)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}
    assert count_parameters(model) == 512 * 5 + 5


@pytest.mark.parametrize(
    "cuda, mps, expected",
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_get_device_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert get_device().type == expected