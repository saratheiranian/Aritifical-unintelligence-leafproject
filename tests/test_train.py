import re
import sys
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import models, train
from src.models import PlantCNN
from src.train import EarlyStopping, make_scheduler, parse_args, run_epoch


def test_early_stopping_waits_for_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.step(0.50) is True
    assert stopper.step(0.505) is False  # below min_delta
    assert not stopper.should_stop
    assert stopper.step(0.49) is False
    assert stopper.should_stop
    assert stopper.best == 0.50


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2)
    stopper.step(0.5)
    stopper.step(0.4)
    assert stopper.step(0.6) is True
    assert stopper.epochs_without_gain == 0


def test_run_epoch_trains_and_evaluates():
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(16, 3, 32, 32), torch.randint(0, 4, (16,)))
    loader = DataLoader(data, batch_size=8)
    model = PlantCNN(num_classes=4, base_channels=4, depth=2)
    before = [p.clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss, acc = run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu", optimizer)
    assert loss > 0 and 0.0 <= acc <= 1.0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    after = [p.clone() for p in model.parameters()]
    run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu")
    assert all(torch.equal(a, b) for a, b in zip(after, model.parameters()))
    assert not model.training


def test_schedulers():
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    assert make_scheduler("none", optimizer, 10) is None
    assert make_scheduler("cosine", optimizer, 10) is not None
    assert make_scheduler("plateau", optimizer, 10) is not None


def test_default_args():
    args = parse_args([])
    assert args.arch == "cnn" and args.batch_size == 64 and args.img_size == 128


# --- train.main (fake loaders, tiny models, ClearML replaced by a mock) ---

NUM_CLASSES = 3
IMG_SIZE = 64
CLASS_NAMES = [f"Class{i}" for i in range(NUM_CLASSES)]


def tiny_loaders():
    """Random tensors shifted by class: 24 / 9 / 9 images, no dataset download needed."""
    torch.manual_seed(0)

    def one(n):
        labels = torch.arange(n) % NUM_CLASSES
        images = torch.randn(n, 3, IMG_SIZE, IMG_SIZE) + labels.view(-1, 1, 1, 1).float()
        return DataLoader(TensorDataset(images, labels), batch_size=8)

    return {"train": one(24), "val": one(9), "test": one(9)}


@pytest.fixture
def fast_train(tmp_path, monkeypatch):
    """Call train.main with fake loaders and a tiny CNN. Extra CLI args can be added per test."""
    monkeypatch.setattr("src.train.make_loaders", lambda **kwargs: (tiny_loaders(), CLASS_NAMES))

    def run(*extra, clearml=False):
        argv = [
            "--base-channels", "4",
            "--depth", "2",
            "--img-size", str(IMG_SIZE),
            "--epochs", "3",
            "--output-dir", str(tmp_path),
            *extra,
        ]
        if not clearml:
            argv.insert(0, "--no-clearml")
        return train.main(argv)

    return run


def test_cosine_scheduler_lowers_the_learning_rate(fast_train, capsys):
    fast_train("--scheduler", "cosine")
    lrs = re.findall(r"lr (\d\.\d+e[+-]\d+)", capsys.readouterr().out)
    assert len(lrs) == 3 and len(set(lrs)) > 1


def test_plateau_scheduler_runs(fast_train):
    assert 0.0 <= fast_train("--scheduler", "plateau") <= 1.0


def test_main_stops_early_when_validation_stalls(fast_train, capsys):
    # min-delta 1.0 means no epoch after the first can count as an improvement
    fast_train("--epochs", "10", "--patience", "1", "--min-delta", "1.0")
    out = capsys.readouterr().out
    assert "Early stopping" in out
    assert len(re.findall(r"^epoch \d+", out, flags=re.MULTILINE)) == 2


def test_main_trains_resnet18_with_frozen_backbone(fast_train, tmp_path, monkeypatch):
    real_build = models.build_model
    # never download the ImageNet weights in a test
    monkeypatch.setattr(
        "src.train.build_model", lambda arch, **kw: real_build(arch, **{**kw, "pretrained": False})
    )
    fast_train("--arch", "resnet18", "--freeze-backbone", "--epochs", "1")
    ckpt = torch.load(tmp_path / "best_model.pt", map_location="cpu")
    assert ckpt["arch"] == "resnet18"
    # the saved copy must never ask to re-download pretrained weights
    assert ckpt["model_kwargs"]["pretrained"] is False
    assert ckpt["model_kwargs"]["freeze_backbone"] is True


def test_main_reports_to_clearml_and_can_run_remotely(fast_train, monkeypatch):
    fake_clearml = MagicMock()
    monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

    fast_train("--queue", "gpu", "--epochs", "2", clearml=True)

    task = fake_clearml.Task.init.return_value
    task.execute_remotely.assert_called_once_with(queue_name="gpu")
    logger = task.get_logger.return_value
    assert logger.report_scalar.call_count == 6 * 2  # six scalars per epoch, two epochs
    assert logger.report_single_value.call_count == 4
    task.close.assert_called_once()