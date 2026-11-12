import numpy as np
import pytest
from PIL import Image

from src import field_test, finetune_field
from src.finetune_field import drop_test_lookalikes, replay_indices, split_per_class


def test_drop_test_lookalikes_removes_only_matching_hashes():
    train = [("a", 0), ("b", 1), ("c", 1)]
    test = [("x", 0)]
    hashes = {"a": "h1", "b": "h2", "c": "h1", "x": "h1"}
    kept, dropped = drop_test_lookalikes(train, test, hash_fn=hashes.get)
    assert kept == [("b", 1)] and dropped == 2


def test_split_per_class_keeps_every_class_in_validation():
    items = [(f"{c}{i}", c) for c in range(3) for i in range(10)] + [("lonely", 3)]
    train, val = split_per_class(items, 0.2, seed=0)
    assert sorted(train + val) == sorted(items)
    assert {label for _, label in val} == {0, 1, 2}  # class 3 has one image, so it stays in train
    assert ("lonely", 3) in train
    assert split_per_class(items, 0.2, seed=0) == (train, val)  # reproducible


def test_replay_indices_takes_up_to_n_per_class():
    labels = [0] * 5 + [1] * 2
    chosen = replay_indices(labels, 3, seed=0)
    assert len(chosen) == 5 and chosen == sorted(chosen)
    assert sum(labels[i] == 0 for i in chosen) == 3 and sum(labels[i] == 1 for i in chosen) == 2


@pytest.fixture
def plantdoc_root(tmp_path):
    """Fake PlantDoc with train/ and test/ folders for two classes of the conftest model."""
    rng = np.random.default_rng(0)
    for split, n in [("train", 6), ("test", 2)]:
        for folder in ["Leaf A", "Leaf B"]:
            (tmp_path / split / folder).mkdir(parents=True)
            for i in range(n):
                pixels = rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(tmp_path / split / folder / f"{i}.png")
    return tmp_path


def test_main_finetunes_and_reports_before_and_after(checkpoint, plantdoc_root, tmp_path, monkeypatch):
    table = {"Leaf A": "Plant0___Disease_0", "Leaf B": "Plant1___Disease_1"}
    monkeypatch.setattr(field_test, "PLANTDOC_TO_PLANTVILLAGE", table)
    out = tmp_path / "ft.pt"
    result = finetune_field.main([
        "--checkpoint", str(checkpoint), "--plantdoc-dir", str(plantdoc_root), "--epochs", "2",
        "--replay-per-class", "0", "--batch-size", "4", "--output", str(out),
    ])
    assert list(result.columns) == ["before", "after"]
    assert "field accuracy" in result.index
    assert out.exists() and (tmp_path / "ft_results.csv").exists()
    import torch
    saved = torch.load(out, map_location="cpu")
    assert saved["finetuned_on"] == "PlantDoc train" and saved["arch"] == "cnn"
