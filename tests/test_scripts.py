import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import ClassLabel, Dataset

from src.evaluate import top_k_accuracy
from src.register_dataset import split_indices
from src.sweep import EXPERIMENTS, to_clearml_params, to_cli
from src.train import parse_args


def test_top_k_accuracy():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    labels = np.array([1, 2, 0])
    assert top_k_accuracy(probs, labels, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy(probs, labels, k=2) == pytest.approx(1 / 3)  # rows 2 and 3 rank their label last
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
    files = {Path(f).as_posix() for f in stage(out, [str(checkpoint)], "4.44.0")}
    assert {"app.py", "requirements.txt", "README.md", "src/inference.py", "models/cnn.pt"} <= files
    assert "sdk_version: 4.44.0" in (out / "README.md").read_text(encoding="utf-8")

def test_local_sweep_carries_on_after_a_failed_variant(monkeypatch, capsys):
    from src import sweep, train

    started = []

    def fake_main(argv):
        name = argv[argv.index("--task-name") + 1]
        started.append(name)
        if name == "sweep-adam":
            raise RuntimeError("out of memory")

    monkeypatch.setattr(train, "main", fake_main)
    sweep.run_local(["baseline", "adam", "dropout-0"], {"epochs": 1})
    assert started == ["sweep-baseline", "sweep-adam", "sweep-dropout-0"]
    assert "Failed variants: adam" in capsys.readouterr().out


def test_sweep_survives_a_broken_wandb_connection(monkeypatch):
    from unittest.mock import MagicMock

    from src import sweep

    fake_wandb = MagicMock()
    fake_wandb.finish.side_effect = ConnectionAbortedError("service gone")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    sweep.close_wandb_run()  # must not raise


def test_num_workers_is_passed_to_every_run(monkeypatch):
    from src import sweep, train

    seen = []
    monkeypatch.setattr(train, "main", lambda argv: seen.append(argv))
    sweep.main(["--local", "--only", "baseline", "--num-workers", "0"])
    assert "--num-workers" in seen[0] and seen[0][seen[0].index("--num-workers") + 1] == "0"
