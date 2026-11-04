import pytest
import torch
from datasets import ClassLabel, Dataset
from PIL import Image
from torch.utils.data import WeightedRandomSampler

from src import data
from src.data import (
    LeafDataset,
    find_columns,
    load_full_dataset,
    make_loaders,
    split_dataset,
    stratified_subset,
)
from src.transforms import get_transforms


def fake_dataset(n=600, classes=3):
    ds = Dataset.from_dict({"label": [i % classes for i in range(n)]})
    return ds.cast_column("label", ClassLabel(names=[f"c{i}" for i in range(classes)]))


def test_split_sizes_are_70_15_15():
    splits = split_dataset(fake_dataset(), "label")
    assert [len(splits[k]) for k in ("train", "val", "test")] == [420, 90, 90]


def test_split_is_stratified():
    splits = split_dataset(fake_dataset(), "label")
    for split in splits.values():
        counts = [split["label"].count(c) for c in range(3)]
        assert max(counts) - min(counts) <= 1


def test_splits_do_not_overlap():
    ds = fake_dataset().add_column("idx", list(range(600)))
    splits = split_dataset(ds, "label")
    ids = [set(s["idx"]) for s in splits.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_stratified_subset_fraction():
    assert len(stratified_subset(fake_dataset(), "label", 0.25)) == 150
    assert len(stratified_subset(fake_dataset(), "label", 1.0)) == 600


def test_transforms_resize_and_normalise():
    image = Image.new("RGB", (256, 256), color=(120, 200, 80))
    for augment in (False, True):
        x = get_transforms(128, augment=augment)(image)
        assert x.shape == (3, 128, 128)
        assert x.dtype == torch.float32


def test_leaf_dataset_returns_tensor_and_label():
    images = [Image.new("RGB", (256, 256)) for _ in range(4)]
    ds = Dataset.from_dict({"image": images, "label": [0, 1, 0, 1]})
    image_col, label_col = find_columns(ds)
    assert image_col == "image"
    item = LeafDataset(ds, image_col, label_col, get_transforms(64))[1]
    assert item[0].shape == (3, 64, 64) and item[1] == 1


# --- load_full_dataset / make_loaders (the real dataset is replaced by a tiny in-memory one) ---

CLASS_NAMES = ["a", "b", "c"]


def image_dataset(n_per_class=40, size=16):
    """Tiny in-memory stand-in for PlantVillage: 3 classes, small solid-colour images."""
    images, labels = [], []
    for i in range(n_per_class):
        for label in range(len(CLASS_NAMES)):
            images.append(Image.new("RGB", (size, size), (60 * label + i % 5, 100, 100)))
            labels.append(label)
    ds = Dataset.from_dict({"image": images, "label": labels})
    return ds.cast_column("label", ClassLabel(names=CLASS_NAMES))


@pytest.fixture
def patched_dataset(monkeypatch):
    """make_loaders calls load_full_dataset, which would download the real dataset."""
    ds = image_dataset()
    monkeypatch.setattr(data, "load_full_dataset", lambda cache_dir=None: (ds, "image", "label"))
    return ds


def test_find_columns_requires_an_image_column():
    with pytest.raises(ValueError, match="No image column"):
        find_columns(Dataset.from_dict({"label": [0, 1]}))


def test_load_full_dataset_encodes_string_labels(monkeypatch):
    raw = Dataset.from_dict(
        {"image": [Image.new("RGB", (8, 8)) for _ in range(6)], "label": ["a", "b", "c", "a", "b", "c"]}
    )
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: raw)
    ds, image_col, label_col = load_full_dataset()
    assert (image_col, label_col) == ("image", "label")
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds.features["label"].names == CLASS_NAMES


def test_load_full_dataset_keeps_an_existing_class_label(monkeypatch):
    ds = image_dataset(n_per_class=2)
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: ds)
    out, _, label_col = load_full_dataset()
    assert out.features[label_col].names == CLASS_NAMES


def test_make_loaders_builds_three_splits(patched_dataset):
    loaders, names = make_loaders(img_size=32, batch_size=8, num_workers=0)
    assert names == CLASS_NAMES
    sizes = [len(loaders[k].dataset) for k in ("train", "val", "test")]
    assert sum(sizes) == 120 and sizes[0] == max(sizes)
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (8, 3, 32, 32) and labels.shape == (8,)


def test_make_loaders_subset_only_shrinks_training(patched_dataset):
    full, _ = make_loaders(num_workers=0)
    small, _ = make_loaders(subset_fraction=0.5, num_workers=0)
    assert len(small["train"].dataset) == pytest.approx(len(full["train"].dataset) / 2, abs=1)
    assert len(small["val"].dataset) == len(full["val"].dataset)
    assert len(small["test"].dataset) == len(full["test"].dataset)


def test_make_loaders_balanced_sampler_only_on_train(patched_dataset):
    loaders, _ = make_loaders(num_workers=0, balanced_sampler=True)
    assert isinstance(loaders["train"].sampler, WeightedRandomSampler)
    assert not isinstance(loaders["val"].sampler, WeightedRandomSampler)


def test_make_loaders_augments_only_the_training_split(patched_dataset):
    loaders, _ = make_loaders(num_workers=0, augment=True)
    train_steps = loaders["train"].dataset.transform.transforms
    val_steps = loaders["val"].dataset.transform.transforms
    assert len(train_steps) > len(val_steps)