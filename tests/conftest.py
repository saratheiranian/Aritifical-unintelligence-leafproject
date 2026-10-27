import pytest
import torch

from src.models import PlantCNN

CLASS_NAMES = [f"Plant{i}___Disease_{i}" for i in range(39)]


@pytest.fixture
def checkpoint(tmp_path):
    """A randomly initialised small CNN saved in the same format as train.py."""
    kwargs = {"num_classes": 39, "base_channels": 8, "depth": 3, "dropout": 0.0, "use_batchnorm": True}
    path = tmp_path / "cnn.pt"
    torch.save(
        {
            "state_dict": PlantCNN(**kwargs).state_dict(),
            "arch": "cnn",
            "model_kwargs": kwargs,
            "class_names": CLASS_NAMES,
            "img_size": 64,
            "val_acc": 0.5,
            "epoch": 1,
        },
        path,
    )
    return path
