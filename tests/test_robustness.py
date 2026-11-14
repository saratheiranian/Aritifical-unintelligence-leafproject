"""Tests for src/robustness.py. It loads the dataset through src.data, so the whole module is
skipped when the dataset library is missing. The distortions themselves are tested in
test_distortions.py, which does not need it."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

datasets = pytest.importorskip("datasets")

# imported after the skip check on purpose: src.robustness needs the dataset library
from src import robustness
from tests.test_distortions import leaf


def fake_results(rows):
    """Results table in the format of evaluate_distortions from (model, distortion, strength,
    accuracy) tuples."""
    return pd.DataFrame(
        [{"model": m, "distortion": d, "strength": s, "accuracy": a, "macro_f1": a}
         for m, d, s, a in rows]
    )


def test_levels_start_with_clean_and_cover_every_strength():
    levels = robustness.all_levels()
    assert levels[0] == ("clean", None)
    assert len(levels) == 1 + sum(len(s) for _, s in robustness.DISTORTIONS.values())


def test_sample_indices_is_fixed_and_sorted():
    assert robustness.sample_indices(10) == list(range(10))
    sample = robustness.sample_indices(100, sample=20)
    assert len(sample) == 20 and sample == sorted(sample)
    assert sample == robustness.sample_indices(100, sample=20)


def test_summary_reports_share_of_clean_accuracy_kept():
    # a starts higher but falls further in points; b keeps a larger share of what it had
    results = fake_results([
        ("a", "clean", None, 0.9), ("a", "blur", 1, 0.6),
        ("b", "clean", None, 0.6), ("b", "blur", 1, 0.45),
    ])
    summary = robustness.summarise(results)
    assert summary.loc["a", "kept"] == pytest.approx(0.6 / 0.9)
    assert summary.loc["b", "kept"] == pytest.approx(0.75)
    assert summary.loc["a", "drop"] == pytest.approx(0.3)


def test_markdown_table_bolds_only_clear_winners():
    results = fake_results([
        ("a", "clean", None, 0.99), ("b", "clean", None, 0.80),  # 19 points: bold
        ("a", "noise", 25, 0.427), ("b", "noise", 25, 0.492),    # 6.5 points: bold
        ("a", "blur", 1, 0.477), ("b", "blur", 1, 0.492),        # 1.5 points: no bold
        ("a", "zoom", 0.4, 0.30), ("b", "zoom", 0.4, 0.35),      # exactly 5 points: bold
    ])
    lines = robustness.markdown_table(results).splitlines()
    assert lines[0] == "| Distortion | Strength | a | b |"
    rows = {tuple(line.split(" | ")[:2]): line for line in lines[2:]}
    assert rows[("| clean", "-")] == "| clean | - | **99.0%** | 80.0% |"
    assert rows[("| noise", "25")] == "| noise | 25 | 42.7% | **49.2%** |"
    assert rows[("| blur", "1")] == "| blur | 1 | 47.7% | 49.2% |"
    assert "**35.0%**" in rows[("| zoom", "0.4")]
    assert len(lines) == 2 + 4  # levels that were not tested are left out


def test_evaluate_summarise_and_plot(checkpoint, tmp_path):
    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(num_classes=39),
    })
    test = datasets.Dataset.from_dict(
        {"image": [leaf() for _ in range(6)], "label": [0, 1, 2, 0, 1, 2]}, features=features
    )
    models = robustness.load_models([str(checkpoint)], "cpu")
    levels = [("clean", None), ("blur", 2), ("noise", 25)]
    results = robustness.evaluate_distortions(
        models, test, "image", "label", 39, list(range(6)), batch_size=4, levels=levels
    )
    assert len(results) == len(levels)
    assert results["accuracy"].between(0, 1).all()

    summary = robustness.summarise(results)
    assert list(summary.index) == ["cnn"]
    assert summary.loc["cnn", "worst_case"] in {"blur 2", "noise 25"}

    full = robustness.evaluate_distortions(
        models, test, "image", "label", 39, [0, 1], batch_size=2
    )
    figures_before = set(plt.get_fignums())
    path = tmp_path / "chart.png"
    assert robustness.plot_results(full, str(path)) == str(path)
    assert path.exists()
    assert set(plt.get_fignums()) == figures_before  # the figure was closed


def test_main_logs_to_wandb(checkpoint, tmp_path, monkeypatch):
    """main() with a fake dataset and a fake wandb, so no download or account is needed."""
    import sys
    import types

    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(num_classes=39),
    })
    test = datasets.Dataset.from_dict({"image": [leaf()] * 2, "label": [0, 1]}, features=features)
    monkeypatch.setattr(robustness, "load_test_split",
                        lambda: (test, "image", "label", [f"class {i}" for i in range(39)]))
    monkeypatch.setattr(robustness, "get_device", lambda: "cpu")

    logged = {}
    run = types.SimpleNamespace(id="abc123", url="https://wandb.ai/run", summary={},
                                log=logged.update, finish=lambda: logged.update(finished=True))
    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: logged.update(init=kwargs) or run,
        Table=lambda dataframe: dataframe,
        Image=lambda path: path,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    robustness.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path)])
    assert logged["init"]["project"] == robustness.WANDB_PROJECT
    assert {"robustness/results", "robustness/summary", "robustness/accuracy_chart"} <= set(logged)
    assert "cnn/kept" in run.summary
    assert logged["finished"]
    assert (tmp_path / "robustness_table.md").exists()