"""Make the stratified 70/15/15 split (seed 42) and print class counts per split.

Every script loads data through src.data.split_dataset with the same seed, so the whole
team trains and evaluates on identical splits.

    python -m src.split_data
"""

import pandas as pd

from src.data import SEED, load_full_dataset, split_dataset


def main():
    ds, _, label_col = load_full_dataset()
    names = ds.features[label_col].names
    splits = split_dataset(ds, label_col, seed=SEED)

    counts = pd.DataFrame(
        {name: pd.Series(list(split[label_col])).value_counts() for name, split in {"full": ds, **splits}.items()}
    ).fillna(0).astype(int)
    counts.index = [names[i] for i in counts.index]
    print(counts.sort_values("full").to_string())
    print("\nSplit sizes:", {name: len(split) for name, split in splits.items()})

    shares = counts / counts.sum()
    drift = shares[["train", "val", "test"]].sub(shares["full"], axis=0).abs().max().max()
    print(f"Largest class-share difference from the full dataset: {drift:.4f}")


if __name__ == "__main__":
    main()
