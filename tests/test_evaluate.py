import sys
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import evaluate
from src.config import WANDB_ENTITY, WANDB_PROJECT
from src.evaluate import load_checkpoint, plot_confusion_matrix, predict

CLASS_NAMES = [f"Plant{i}___Disease_{i}" for i in range(39)]
IMG_SIZE = 64  # the `checkpoint` fixture in conftest.py is saved with img_size=64


def make_loader(n=78, batch_size=16):
    """Random images, two per class, so every class appears in the 'test split'."""
    torch.manual_seed(0)
    images = torch.randn(n, 3, IMG_SIZE, IMG_SIZE)
    labels = torch.arange(n) % 39
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


@pytest.fixture
def fake_loaders(monkeypatch):
    """Replace make_loaders so evaluate.main never touches the real dataset."""
    seen = {}

    def fake_make_loaders(**kwargs):
        seen.update(kwargs)
        return {"test": make_loader()}, CLASS_NAMES

    monkeypatch.setattr(evaluate, "make_loaders", fake_make_loaders)
    return seen


def test_load_checkpoint_returns_eval_model_and_metadata(checkpoint):
    model, ckpt = load_checkpoint(checkpoint)
    assert not model.training
    assert ckpt["arch"] == "cnn" and ckpt["img_size"] == IMG_SIZE
    assert len(ckpt["class_names"]) == 39


def test_predict_returns_probabilities_and_labels(checkpoint):
    model, _ = load_checkpoint(checkpoint)
    probs, labels = predict(model, make_loader(), "cpu")
    assert probs.shape == (78, 39)
    assert labels.shape == (78,)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_plot_confusion_matrix_writes_png(tmp_path):
    path = tmp_path / "cm.png"
    fig = plot_confusion_matrix(np.eye(39, dtype=int), CLASS_NAMES, path)
    assert path.exists() and path.stat().st_size > 0
    plt.close(fig)


def test_main_writes_report_and_confusion_matrix(checkpoint, tmp_path, fake_loaders):
    result = evaluate.main(["--checkpoint", str(checkpoint), "--no-wandb", "--output-dir", str(tmp_path)])
    assert set(result) == {"accuracy", "top5_accuracy", "macro_f1"}
    assert 0.0 <= result["accuracy"] <= result["top5_accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert (tmp_path / "test_report.csv").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    # evaluation must use the image size the model was trained with, and no augmentation
    assert fake_loaders["img_size"] == IMG_SIZE
    assert fake_loaders["augment"] is False


def test_main_logs_to_wandb(checkpoint, tmp_path, fake_loaders, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    run = fake_wandb.init.return_value
    run.summary = {}  # a real dict, so we can check the exact values written to it

    result = evaluate.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path)])

    init_kwargs = fake_wandb.init.call_args.kwargs
    assert init_kwargs["entity"] == WANDB_ENTITY
    assert init_kwargs["project"] == WANDB_PROJECT
    logged = run.log.call_args.args[0]
    assert set(logged) == {"test/per_class", "test/weakest_classes", "test/confusion_matrix"}
    # the summary holds exactly the numbers main() returns
    assert run.summary == {
        "test_accuracy": result["accuracy"],
        "test_top5_accuracy": result["top5_accuracy"],
        "test_macro_f1": result["macro_f1"],
    }
    run.finish.assert_called_once_with()  # no exit code: W&B shows it as finished


@pytest.mark.parametrize("error", [RuntimeError("dataset not available"), KeyboardInterrupt()])
def test_main_marks_wandb_run_failed_when_evaluation_stops(error, checkpoint, tmp_path,
                                                           monkeypatch):
    """A crash or Ctrl+C closes the run as failed (exit code 1), not as a finished result, and
    the original error still reaches the user."""
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def broken_make_loaders(**kwargs):
        raise error

    monkeypatch.setattr(evaluate, "make_loaders", broken_make_loaders)

    with pytest.raises(type(error)):
        evaluate.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path)])
    fake_wandb.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_old_no_clearml_flag_still_turns_logging_off(checkpoint, tmp_path, fake_loaders, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    evaluate.main(["--checkpoint", str(checkpoint), "--no-clearml", "--output-dir", str(tmp_path)])

    fake_wandb.init.assert_not_called()
