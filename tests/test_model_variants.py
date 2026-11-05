"""Tests for the residual CNN, label smoothing and class-weighted loss."""

import pytest
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import TensorDataset

from src.data import LeafDataset
from src.models import (
    PlantCNN,
    ResidualBlock,
    ResidualCNN,
    build_model,
    count_parameters,
)
from src.train import dataset_labels, make_criterion, parse_args
from tests.test_train import fast_train  # noqa: F401  (pytest fixture)


@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("use_batchnorm", [True, False])
def test_rescnn_output_shape(img_size, use_batchnorm):
    model = ResidualCNN(num_classes=39, base_channels=8, depth=3, use_batchnorm=use_batchnorm)
    assert model(torch.randn(2, 3, img_size, img_size)).shape == (2, 39)


def test_residual_block_shortcut_matches_shapes():
    same = ResidualBlock(16, 16)
    down = ResidualBlock(16, 32, stride=2)
    assert isinstance(same.shortcut, nn.Identity)
    assert not isinstance(down.shortcut, nn.Identity)
    assert down(torch.randn(1, 16, 32, 32)).shape == (1, 32, 16, 16)


def test_residual_block_passes_input_through_when_body_is_zero():
    block = ResidualBlock(4, 4, use_batchnorm=False)
    for p in block.body.parameters():
        nn.init.zeros_(p)
    x = torch.rand(1, 4, 8, 8)  # non-negative, so the final ReLU keeps it unchanged
    assert torch.allclose(block(x), x)


def test_build_model_makes_a_rescnn_and_depth_adds_parameters():
    model = build_model("rescnn", num_classes=5, base_channels=8, depth=2)
    assert isinstance(model, ResidualCNN)
    deeper = build_model("rescnn", num_classes=5, base_channels=8, depth=3)
    assert count_parameters(deeper) > count_parameters(model)
    assert count_parameters(model) != count_parameters(PlantCNN(num_classes=5, base_channels=8, depth=2))


def test_dataset_labels_reads_leaf_dataset_column_and_tensor_dataset():
    from PIL import Image

    hf = Dataset.from_dict({"image": [Image.new("RGB", (8, 8))] * 3, "label": [2, 0, 1]})
    assert dataset_labels(LeafDataset(hf, "image", "label", transform=None)) == [2, 0, 1]
    assert dataset_labels(TensorDataset(torch.zeros(3, 1), torch.tensor([1, 1, 0]))) == [1, 1, 0]


def test_default_criterion_is_plain_cross_entropy():
    criterion = make_criterion(parse_args([]), TensorDataset(torch.zeros(2, 1), torch.tensor([0, 1])), 2, "cpu")
    assert criterion.weight is None and criterion.label_smoothing == 0.0


def test_criterion_with_label_smoothing_and_class_weights():
    args = parse_args(["--label-smoothing", "0.1", "--class-weights"])
    data = TensorDataset(torch.zeros(4, 1), torch.tensor([0, 0, 0, 1]))
    criterion = make_criterion(args, data, 2, "cpu")
    assert criterion.label_smoothing == pytest.approx(0.1)
    assert criterion.weight.tolist() == pytest.approx([4 / 6, 2.0])


def test_label_smoothing_raises_the_loss_of_a_confident_correct_prediction():
    logits, target = torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([0])
    plain = nn.CrossEntropyLoss()(logits, target)
    smoothed = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
    assert smoothed > plain


@pytest.mark.parametrize(
    "extra",
    [("--arch", "rescnn"), ("--label-smoothing", "0.1"), ("--class-weights",)],
)
def test_train_main_runs_with_new_options(fast_train, tmp_path, extra):  # noqa: F811
    assert 0.0 <= fast_train(*extra) <= 1.0
    ckpt = torch.load(tmp_path / "best_model.pt", weights_only=False)
    rebuilt = build_model(ckpt["arch"], **ckpt["model_kwargs"])
    rebuilt.load_state_dict(ckpt["state_dict"])
    assert ckpt["arch"] == ("rescnn" if "rescnn" in extra else "cnn")
