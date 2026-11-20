"""Fakes shared by the robustness and Grad-CAM tests: a tiny test split and a fake wandb module,
so their main() functions run without the dataset download or a W&B account."""

import sys
import types

import pytest

from tests.test_distortions import leaf


def fake_test_split(monkeypatch, n_images=2, class_names=None):
    """Replace robustness.load_test_split (Grad-CAM loads the data through it too) with n_images
    leaf photos labelled 0, 1, 2, 0, 1, ... and 39 classes."""
    datasets = pytest.importorskip("datasets")
    from src import robustness

    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(num_classes=39),
    })
    test = datasets.Dataset.from_dict(
        {"image": [leaf() for _ in range(n_images)], "label": [i % 3 for i in range(n_images)]},
        features=features)
    names = class_names or [f"class_{i}" for i in range(39)]
    monkeypatch.setattr(robustness, "load_test_split", lambda: (test, "image", "label", names))
    return test


def fake_wandb(monkeypatch):
    """Install a fake wandb module and return (started, logged, run).

    started holds the keyword arguments of each wandb.init call, logged everything passed to
    run.log, and logged["exit_codes"] one entry per run.finish call: None when the run finished
    normally, 1 when it was marked as failed."""
    started, logged = [], {"exit_codes": []}
    run = types.SimpleNamespace(
        id="abc123", url="https://wandb.ai/run", summary={}, log=logged.update,
        finish=lambda exit_code=None: logged["exit_codes"].append(exit_code))
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(
        init=lambda **kwargs: started.append(kwargs) or run,
        Table=lambda dataframe: dataframe,
        Image=lambda path: path))
    return started, logged, run
