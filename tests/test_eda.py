"""Tests for src/eda.py. The real dataset is replaced by the small in-memory one from
test_data_quality, so nothing is downloaded."""

import sys
from unittest.mock import MagicMock

import pytest

from src import eda
from tests.test_data_quality import QUALITY_CLASS_NAMES, noisy_dataset

EDA_FILES = ["class_distribution.png", "samples.png", "split_check.png"]


@pytest.fixture
def eda_dataset(monkeypatch):
    ds = noisy_dataset()
    monkeypatch.setattr(eda, "load_full_dataset", lambda: (ds, "image", "label"))
    return ds


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_class_distribution_counts_every_class(eda_dataset):
    dist = eda.class_distribution(eda_dataset, "label")
    assert list(dist.index) == QUALITY_CLASS_NAMES
    assert dist.sum() == len(eda_dataset)


def test_no_wandb_saves_the_three_plots_without_a_run(eda_dataset, fake_wandb, tmp_path):
    eda.main(["--no-wandb", "--output-dir", str(tmp_path)])
    for name in EDA_FILES:
        assert (tmp_path / name).exists(), name
    fake_wandb.init.assert_not_called()


def test_logs_summary_values_and_the_three_plots(eda_dataset, fake_wandb, tmp_path):
    eda.main(["--output-dir", str(tmp_path)])

    assert fake_wandb.init.call_args.kwargs["name"] == "eda"
    run = fake_wandb.init.return_value
    summary = run.summary.update.call_args.args[0]
    assert summary == {"images": 90, "classes": 3,
                       "smallest_class_images": 30, "largest_class_images": 30}
    assert set(run.log.call_args.args[0]) == {"eda/class_distribution", "eda/samples",
                                             "eda/split_check"}
    run.finish.assert_called_once_with()


def test_a_crash_closes_the_wandb_run_as_failed(fake_wandb, tmp_path, monkeypatch):
    def broken_download():
        raise RuntimeError("no network")

    monkeypatch.setattr(eda, "load_full_dataset", broken_download)
    with pytest.raises(RuntimeError):
        eda.main(["--output-dir", str(tmp_path)])
    fake_wandb.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_old_no_clearml_flag_still_turns_logging_off(eda_dataset, fake_wandb, tmp_path):
    eda.main(["--no-clearml", "--output-dir", str(tmp_path)])
    fake_wandb.init.assert_not_called()
    assert (tmp_path / "samples.png").exists()
