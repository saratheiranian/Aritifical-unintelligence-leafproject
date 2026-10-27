"""Data loading and stratified splitting for the PlantVillage dataset."""

from datasets import ClassLabel, Image, load_dataset
from torch.utils.data import DataLoader, Dataset

from src.transforms import get_transforms

DATASET_ID = "DScomp380/plant_village"
SEED = 42


def find_columns(ds):
    """Return (image_column, label_column) by looking at the dataset features."""
    image_col = next((name for name, f in ds.features.items() if isinstance(f, Image)), None)
    label_col = next((name for name, f in ds.features.items() if isinstance(f, ClassLabel)), None)
    if image_col is None:
        raise ValueError(f"No image column found in features: {ds.features}")
    if label_col is None:
        # Labels stored as plain strings or ints: load_full_dataset turns them into a ClassLabel.
        label_col = "label" if "label" in ds.column_names else ds.column_names[-1]
    return image_col, label_col


def load_full_dataset(cache_dir=None):
    """Load the single provided split and make sure the label column is a ClassLabel."""
    ds = load_dataset(DATASET_ID, split="train", cache_dir=cache_dir)
    image_col, label_col = find_columns(ds)
    if not isinstance(ds.features[label_col], ClassLabel):
        ds = ds.class_encode_column(label_col)
    return ds, image_col, label_col


def split_dataset(ds, label_col, val_size=0.15, test_size=0.15, seed=SEED):
    """Stratified train/val/test split, so every split keeps the original class distribution."""
    first = ds.train_test_split(
        test_size=val_size + test_size, stratify_by_column=label_col, seed=seed
    )
    rest = first["test"].train_test_split(
        test_size=test_size / (val_size + test_size), stratify_by_column=label_col, seed=seed
    )
    return {"train": first["train"], "val": rest["train"], "test": rest["test"]}


def stratified_subset(ds, label_col, fraction, seed=SEED):
    """Keep a stratified fraction of a split (e.g. 0.25 for fast prototyping)."""
    if fraction >= 1.0:
        return ds
    return ds.train_test_split(train_size=fraction, stratify_by_column=label_col, seed=seed)["train"]


class LeafDataset(Dataset):
    """Wraps a Hugging Face split so it returns (image_tensor, label) pairs for PyTorch."""

    def __init__(self, hf_split, image_col, label_col, transform):
        self.ds = hf_split
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        image = row[self.image_col].convert("RGB")
        return self.transform(image), row[self.label_col]


def make_loaders(
    img_size=128, augment=True, batch_size=64, subset_fraction=1.0, num_workers=2, seed=SEED
):
    """Build train/val/test DataLoaders. Returns (loaders, class_names)."""
    ds, image_col, label_col = load_full_dataset()
    splits = split_dataset(ds, label_col, seed=seed)
    splits["train"] = stratified_subset(splits["train"], label_col, subset_fraction, seed=seed)

    loaders = {}
    for name, split in splits.items():
        transform = get_transforms(img_size, augment=(augment and name == "train"))
        loaders[name] = DataLoader(
            LeafDataset(split, image_col, label_col, transform),
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
        )
    class_names = ds.features[label_col].names
    return loaders, class_names
