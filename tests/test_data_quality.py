import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from datasets import ClassLabel, Dataset
from PIL import Image

from src import data_quality
from src.data import class_counts, class_weights, make_balanced_sampler
from src.data_quality import (
    average_hash,
    conflicting_labels,
    cross_split_leaks,
    find_duplicates,
    flag_images,
    image_stats,
)


def half_image(left, right, size=64):
    """Left half one grey level, right half another."""
    image = Image.new("RGB", (size, size), (left,) * 3)
    image.paste(Image.new("RGB", (size // 2, size), (right,) * 3), (size // 2, 0))
    return image


def test_class_counts_include_empty_classes():
    assert class_counts([0, 0, 2], 4) == [2, 0, 1, 0]


def test_class_weights_are_inverse_frequency():
    weights = class_weights([0, 0, 0, 1], 2)
    assert weights == pytest.approx([4 / 6, 2.0])
    assert class_weights([0, 0], 2)[1] == 0.0


def test_balanced_sampler_draws_classes_equally():
    labels = [0] * 900 + [1] * 100
    sampler = make_balanced_sampler(labels, 2)
    drawn = [labels[i] for i in sampler]
    assert len(drawn) == 1000
    assert 0.42 < sum(drawn) / len(drawn) < 0.58  # ~50% rare class instead of 10%


def test_balanced_sampler_is_reproducible():
    labels = [0] * 50 + [1] * 10
    assert list(make_balanced_sampler(labels, 2)) == list(make_balanced_sampler(labels, 2))


def test_image_stats_of_plain_image():
    stats = image_stats(Image.new("RGB", (30, 20), (100, 100, 100)))
    assert (stats["width"], stats["height"], stats["mode"]) == (30, 20, "RGB")
    assert stats["brightness"] == pytest.approx(100)
    assert stats["contrast"] == pytest.approx(0)


def test_average_hash_matches_resized_copy_and_differs_from_other_images():
    image = half_image(20, 230)
    assert average_hash(image) == average_hash(image.resize((128, 128)))
    assert average_hash(image) != average_hash(half_image(230, 20))


def test_find_duplicates_groups_equal_hashes():
    assert find_duplicates(["a", "b", "a", "c", "b"]) == [[0, 2], [1, 4]]
    assert find_duplicates(["a", "b"]) == []


def test_cross_split_leaks_and_conflicting_labels():
    groups = [[0, 1], [2, 3]]
    split_of = {0: "train", 1: "test", 2: "train", 3: "train"}
    labels = [5, 5, 1, 2]
    assert cross_split_leaks(groups, split_of) == [[0, 1]]
    assert conflicting_labels(groups, labels) == [[2, 3]]


def test_flag_images():
    df = pd.DataFrame({"brightness": [10, 120, 250, 120], "contrast": [30, 30, 30, 5]})
    flagged = flag_images(df)
    assert list(flagged.index) == [0, 2, 3]
    assert list(flagged["reason"]) == ["dark", "bright", "low-contrast"]


# --- data_quality.main (the real dataset is replaced by a small in-memory one) ---

QUALITY_CLASS_NAMES = ["healthy", "blight", "rust"]
QUALITY_FILES = [
    "quality_sizes.csv",
    "quality_brightness_per_class.csv",
    "quality_flagged_images.csv",
    "quality_duplicates.csv",
    "quality_brightness.png",
    "quality_brightness_per_class.png",
]


def noisy_dataset(n=90, size=32):
    """Random-noise images (so hashes are unique) plus one planted duplicate and one dark flat image."""
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)) for _ in range(n)]
    images[1] = images[0].copy()  # exact duplicate, but row 1 has a different label than row 0
    images[2] = Image.new("RGB", (size, size), (5, 5, 5))  # flagged as dark and low-contrast
    labels = [i % 3 for i in range(n)]
    ds = Dataset.from_dict({"image": images, "label": labels})
    return ds.cast_column("label", ClassLabel(names=QUALITY_CLASS_NAMES))


@pytest.fixture
def quality_dataset(monkeypatch):
    """data_quality.main would otherwise download the full PlantVillage dataset."""
    ds = noisy_dataset()
    monkeypatch.setattr(data_quality, "load_full_dataset", lambda: (ds, "image", "label"))
    return ds


def test_main_writes_tables_and_plots(quality_dataset, tmp_path):
    result = data_quality.main(["--no-wandb", "--output-dir", str(tmp_path)])
    assert result["duplicate_groups"] >= 1  # the planted duplicate pair
    assert result["flagged"] >= 1  # the planted dark image
    for name in QUALITY_FILES:
        assert (tmp_path / name).exists(), name


def test_main_can_check_a_random_sample(quality_dataset, tmp_path):
    data_quality.main(["--no-wandb", "--sample", "30", "--output-dir", str(tmp_path)])
    sizes = pd.read_csv(tmp_path / "quality_sizes.csv")
    assert sizes["images"].sum() == 30


def test_main_logs_results_to_wandb(quality_dataset, tmp_path, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    data_quality.main(["--output-dir", str(tmp_path)])

    run = fake_wandb.init.return_value
    assert fake_wandb.init.call_args.kwargs["name"] == "data-quality"
    summary = run.summary.update.call_args.args[0]
    assert summary["duplicate_groups"] >= 1 and summary["images_checked"] == len(quality_dataset)
    logged = run.log.call_args.args[0]
    assert set(logged) == {"quality/image_sizes", "quality/brightness_per_class",
                           "quality/duplicates", "quality/flagged_images",
                           "quality/brightness", "quality/brightness_per_class_chart"}
    run.finish.assert_called_once()


def test_old_no_clearml_flag_still_turns_logging_off(quality_dataset, tmp_path, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    data_quality.main(["--no-clearml", "--output-dir", str(tmp_path)])
    fake_wandb.init.assert_not_called()


def test_a_crash_closes_the_wandb_run_as_failed(tmp_path, monkeypatch):
    fake_wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def broken_download():
        raise RuntimeError("no network")

    monkeypatch.setattr(data_quality, "load_full_dataset", broken_download)
    with pytest.raises(RuntimeError):
        data_quality.main(["--output-dir", str(tmp_path)])
    fake_wandb.init.return_value.finish.assert_called_once_with(exit_code=1)
