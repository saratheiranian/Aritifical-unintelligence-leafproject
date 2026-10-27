"""Version the data split in ClearML (the "data" task).

Stores, as a ClearML Dataset, the row indices of the stratified 70/15/15 split (seed 42),
the class names and the class counts per split. The images themselves stay on the
Hugging Face Hub; this records exactly which images each split used.

    python -m src.register_dataset
"""

import argparse
import json
import os
import tempfile

import pandas as pd

from src.data import SEED, load_full_dataset, split_dataset
from src.train import CLEARML_PROJECT


def split_indices(ds, label_col, seed=SEED):
    """Row indices (into the original dataset) of each split."""
    indexed = ds.select_columns([label_col]).add_column("row_idx", list(range(len(ds))))
    return {name: list(split["row_idx"]) for name, split in split_dataset(indexed, label_col, seed=seed).items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-name", default="plant_village_split")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    from clearml import Dataset, Task

    task = Task.init(project_name=CLEARML_PROJECT, task_name="register-dataset", tags=["data"])
    ds, _, label_col = load_full_dataset()
    names = ds.features[label_col].names
    indices = split_indices(ds, label_col, seed=args.seed)

    labels = list(ds[label_col])
    counts = pd.DataFrame(
        {split: pd.Series([labels[i] for i in idx]).value_counts() for split, idx in indices.items()}
    ).fillna(0).astype(int)
    counts.index = [names[i] for i in counts.index]

    with tempfile.TemporaryDirectory() as tmp:
        for split, idx in indices.items():
            with open(os.path.join(tmp, f"{split}_indices.json"), "w") as f:
                json.dump(idx, f)
        with open(os.path.join(tmp, "class_names.json"), "w") as f:
            json.dump(names, f, indent=1)
        counts.to_csv(os.path.join(tmp, "class_counts.csv"))

        dataset = Dataset.create(dataset_name=args.dataset_name, dataset_project=CLEARML_PROJECT)
        dataset.add_files(tmp)
        dataset.get_logger().report_table("Class counts", "per split", iteration=0, table_plot=counts)
        dataset.upload()
        dataset.finalize()

    task.get_logger().report_table("Class counts", "per split", iteration=0, table_plot=counts)
    print(f"Registered dataset {dataset.id}: { {k: len(v) for k, v in indices.items()} }")


if __name__ == "__main__":
    main()
