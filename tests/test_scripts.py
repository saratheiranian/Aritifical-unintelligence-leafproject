import numpy as np
import pytest
from datasets import ClassLabel, Dataset

from src.evaluate import top_k_accuracy
from src.register_dataset import split_indices
from src.sweep import EXPERIMENTS, to_cli, to_clearml_params
from src.train import parse_args


def test_top_k_accuracy():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    labels = np.array([1, 2, 0])
    assert top_k_accuracy(probs, labels, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy(probs, labels, k=2) == pytest.approx(2 / 3)
    assert top_k_accuracy(probs, labels, k=3) == 1.0


def test_to_cli():
    assert to_cli({"lr": 0.0003, "no_augment": True, "no_batchnorm": False}) == ["--lr", "0.0003", "--no-augment"]


def test_to_clearml_params():
    assert to_clearml_params({"batch_size": 32}) == {"Args/batch_size": 32}


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_every_sweep_experiment_is_a_valid_train_config(name):
    overrides = EXPERIMENTS[name]
    args = parse_args(to_cli(overrides))
    for key, value in overrides.items():
        assert getattr(args, key) == value


def test_split_indices_cover_dataset_once():
    ds = Dataset.from_dict({"label": [i % 3 for i in range(300)]})
    ds = ds.cast_column("label", ClassLabel(names=["a", "b", "c"]))
    idx = split_indices(ds, "label")
    all_rows = idx["train"] + idx["val"] + idx["test"]
    assert sorted(all_rows) == list(range(300))
    assert [len(idx[k]) for k in ("train", "val", "test")] == [210, 45, 45]


def test_deploy_stage(tmp_path, checkpoint):
    from src.deploy_space import stage

    out = tmp_path / "space"
    out.mkdir()
    files = stage(out, [str(checkpoint)], "4.44.0")
    assert {"app.py", "requirements.txt", "README.md", "src/inference.py", "models/cnn.pt"} <= set(files)
    assert "sdk_version: 4.44.0" in (out / "README.md").read_text()
