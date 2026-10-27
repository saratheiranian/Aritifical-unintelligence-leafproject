import torch
from datasets import ClassLabel, Dataset
from PIL import Image

from src.data import LeafDataset, find_columns, split_dataset, stratified_subset
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
