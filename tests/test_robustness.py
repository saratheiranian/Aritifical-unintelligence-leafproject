"""Tests for src/robustness.py. It loads the dataset through src.data, so the whole module is
skipped when the dataset library is missing. The distortions themselves are tested in
test_distortions.py, which does not need it."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

datasets = pytest.importorskip("datasets")

# imported after the skip check on purpose: src.robustness needs the dataset library
from src import robustness
from tests.eval_fakes import fake_test_split, fake_wandb
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
        models, test, "image", "label", list(range(6)), batch_size=4, levels=levels
    )
    assert len(results) == len(levels)
    assert results["accuracy"].between(0, 1).all()

    summary = robustness.summarise(results)
    assert list(summary.index) == ["cnn"]
    assert summary.loc["cnn", "worst_case"] in {"blur 2", "noise 25"}

    full = robustness.evaluate_distortions(
        models, test, "image", "label", [0, 1], batch_size=2
    )
    figures_before = set(plt.get_fignums())
    path = tmp_path / "chart.png"
    assert robustness.plot_results(full, str(path)) == str(path)
    assert path.exists()
    assert set(plt.get_fignums()) == figures_before  # the figure was closed


def test_main_logs_to_wandb(checkpoint, tmp_path, monkeypatch):
    """main() with a fake dataset and a fake wandb, so no download or account is needed."""
    fake_test_split(monkeypatch, class_names=[f"class {i}" for i in range(38)]
                    + [robustness.NO_LEAF_CLASS])
    monkeypatch.setattr(robustness, "get_device", lambda: "cpu")
    started, logged, run = fake_wandb(monkeypatch)

    robustness.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path)])
    assert started[0]["project"] == robustness.WANDB_PROJECT
    assert {"robustness/results", "robustness/summary", "robustness/accuracy_chart"} <= set(logged)
    assert "cnn/kept" in run.summary
    assert logged["exit_codes"] == [None]  # closed once, as a finished run
    assert (tmp_path / "robustness_table.md").exists()
    assert (tmp_path / "robustness_no_leaf_table.md").exists()
    assert "cnn/max_no_leaf" in run.summary


def test_preview_does_not_start_a_wandb_run(tmp_path, monkeypatch):
    """--preview saves the distortion figure only; it must not leave an empty run in WandB."""
    fake_test_split(monkeypatch, n_images=1)
    started, _, _ = fake_wandb(monkeypatch)

    assert robustness.main(["--preview", "--output-dir", str(tmp_path)]) is None
    assert started == []  # no run was started
    assert (tmp_path / "distortions_preview.png").exists()


def test_macro_f1_ignores_classes_missing_from_the_sample():
    """A sample with 2 of the 39 classes, all predicted correctly, must score 1.0, not 2/39."""
    import torch
    from torch import nn

    from src.transforms import get_transforms

    class AlwaysRight(nn.Module):
        """Predicts class 0 for the first image of a batch and class 1 for the second."""

        def forward(self, x):
            logits = torch.zeros(len(x), 39)
            logits[torch.arange(len(x)), torch.arange(len(x)) % 2] = 10.0
            return logits

    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(num_classes=39),
    })
    test = datasets.Dataset.from_dict({"image": [leaf(), leaf()], "label": [0, 1]},
                                      features=features)
    models = {"right": (AlwaysRight(), get_transforms(64))}
    results = robustness.evaluate_distortions(models, test, "image", "label", [0, 1],
                                              batch_size=2, levels=[("clean", None)])
    assert results.loc[0, "accuracy"] == 1.0
    assert results.loc[0, "macro_f1"] == pytest.approx(1.0)
    assert robustness.macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx((2 / 3 + 0.8) / 2)


def test_help_and_preview_titles_show_strengths_cleanly(tmp_path, monkeypatch):
    # the example command in --help keeps its line break instead of joining with extra spaces
    assert "outputs/cnn_augment-none.pt \\\n" in robustness.__doc__
    # the preview figure writes "bright 2", like the tables and chart, not "bright 2.0"
    import matplotlib.pyplot as plt

    figures = []
    monkeypatch.setattr(plt, "close", figures.append)
    robustness.save_preview(leaf(), str(tmp_path / "preview.png"))
    titles = {ax.get_title() for ax in figures[0].axes}
    assert "bright 2" in titles and "bright 2.0" not in titles
    monkeypatch.undo()
    plt.close(figures[0])


@pytest.mark.parametrize("bad, message", [
    (["--sample", "0"], "--sample must be at least 1"),
    (["--batch-size", "0"], "--batch-size must be at least 1"),
    (["--checkpoint", "missing.pt"], "checkpoint not found: missing.pt"),
    (["--preview-index", "5"], "--preview-index must be between 0 and 1"),
])
def test_bad_input_stops_with_a_clear_message_before_any_wandb_run(
        bad, message, checkpoint, tmp_path, monkeypatch, capsys):
    fake_test_split(monkeypatch)
    started, _, _ = fake_wandb(monkeypatch)

    with pytest.raises(SystemExit):
        robustness.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path)] + bad)
    assert message in capsys.readouterr().err
    assert started == []  # no run was started, so nothing is left behind in WandB


def test_evaluation_runs_without_gradients():
    """Robustness only needs predictions: no gradients, so it is faster and uses less memory."""
    import torch
    from torch import nn

    from src.transforms import get_transforms

    seen = []

    class Recorder(nn.Module):
        def forward(self, x):
            seen.append(torch.is_grad_enabled())
            return torch.zeros(len(x), 39)

    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(num_classes=39),
    })
    test = datasets.Dataset.from_dict({"image": [leaf()], "label": [0]}, features=features)
    robustness.evaluate_distortions({"r": (Recorder(), get_transforms(64))}, test, "image",
                                    "label", [0], levels=[("clean", None)])
    assert seen == [False]


def test_no_leaf_rate_counts_leaf_photos_read_as_background(capsys):
    """Class 38 plays "Background_without_leaves". A model that answers it for every photo reads
    all 3 leaf photos as "no leaf" (100%); the real background photo is left out of the rate."""
    import torch
    from torch import nn

    from src.transforms import get_transforms

    class SaysNoLeaf(nn.Module):
        def forward(self, x):
            logits = torch.zeros(len(x), 39)
            logits[:, 38] = 10.0
            return logits

    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(num_classes=39),
    })
    test = datasets.Dataset.from_dict({"image": [leaf()] * 4, "label": [0, 1, 2, 38]},
                                      features=features)
    models = {"nl": (SaysNoLeaf(), get_transforms(64))}
    levels = [("clean", None), ("noise", 25)]
    results = robustness.evaluate_distortions(models, test, "image", "label", [0, 1, 2, 3],
                                              levels=levels, no_leaf_class=38)
    assert list(results["no_leaf"]) == [1.0, 1.0]
    assert list(results["accuracy"]) == [0.25, 0.25]  # only the background photo is right
    assert "(no leaf 100%)" in capsys.readouterr().out

    summary = robustness.summarise(results)
    assert summary.loc["nl", "max_no_leaf"] == 1.0
    assert summary.loc["nl", "clean_no_leaf"] == 1.0
    table = robustness.markdown_table(results, min_gap=None, column="no_leaf")
    assert "| noise | 25 | 100.0% |" in table and "**" not in table

    without = robustness.evaluate_distortions(models, test, "image", "label", [0, 1, 2, 3],
                                              levels=levels)
    assert without["no_leaf"].isna().all()  # no class given: no rate
    assert "max_no_leaf" not in robustness.summarise(without)


def test_two_model_files_with_the_same_name_are_refused(checkpoint, tmp_path, capsys):
    """outputs/final_model.pt and outputs/new/final_model.pt would both be called final_model,
    and one would silently replace the other, so only one model would be tested."""
    other = tmp_path / "new"
    other.mkdir()
    copy = other / checkpoint.name
    copy.write_bytes(checkpoint.read_bytes())
    paths = [str(checkpoint), str(copy)]
    assert robustness.duplicate_names(paths) == [checkpoint.stem]
    assert robustness.duplicate_names([str(checkpoint)]) == []
    assert robustness.duplicate_names_error([str(checkpoint)]) is None
    assert robustness.duplicate_names_error(paths).startswith(
        f"two model files have the same name: {checkpoint.stem}.")
    with pytest.raises(ValueError, match="same name"):
        robustness.load_models(paths, "cpu")
    with pytest.raises(SystemExit):  # stops before loading the data or starting a WandB run
        robustness.main(["--checkpoint", *paths, "--output-dir", str(tmp_path)])
    assert "two model files have the same name" in capsys.readouterr().err


@pytest.mark.parametrize("error", [RuntimeError("out of memory"), KeyboardInterrupt()])
def test_wandb_run_is_marked_failed_when_the_test_stops_halfway(
        error, checkpoint, tmp_path, monkeypatch):
    """A crash or Ctrl+C closes the run as failed (exit code 1), not as a finished result, and
    the original error still reaches the user."""
    fake_test_split(monkeypatch)
    monkeypatch.setattr(robustness, "get_device", lambda: "cpu")
    started, logged, _ = fake_wandb(monkeypatch)

    def crash(*args, **kwargs):
        raise error

    monkeypatch.setattr(robustness, "evaluate_distortions", crash)
    with pytest.raises(type(error)):
        robustness.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path)])
    assert len(started) == 1
    assert logged["exit_codes"] == [1]  # closed once, as failed, not left running

