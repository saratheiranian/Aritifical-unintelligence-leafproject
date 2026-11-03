import pandas as pd
import pytest
from PIL import Image

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
