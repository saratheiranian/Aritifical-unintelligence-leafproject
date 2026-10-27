"""Exploratory data analysis: class distribution, sample images, split check.

Saves the plots to outputs/eda/ (use them in the report) and logs them to ClearML.

    python -m src.eda
"""

import argparse
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_full_dataset, split_dataset
from src.train import CLEARML_PROJECT


def class_distribution(ds, label_col):
    names = ds.features[label_col].names
    counts = Counter(ds[label_col])
    return pd.Series({names[i]: counts.get(i, 0) for i in range(len(names))})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/eda")
    p.add_argument("--no-clearml", action="store_true")
    args = p.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    task = None
    if not args.no_clearml:
        from clearml import Task

        task = Task.init(project_name=CLEARML_PROJECT, task_name="eda", tags=["data"])

    ds, image_col, label_col = load_full_dataset()
    print(ds)
    dist = class_distribution(ds, label_col).sort_values()
    print(f"{len(ds)} images, {len(dist)} classes")
    print(f"smallest class: {dist.index[0]} ({dist.iloc[0]}), largest: {dist.index[-1]} ({dist.iloc[-1]})")

    # 1. images per class
    fig1, ax = plt.subplots(figsize=(8, 10))
    dist.plot.barh(ax=ax)
    ax.set_xlabel("Number of images")
    ax.set_title("Images per class")
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.output_dir, "class_distribution.png"), dpi=150)

    # 2. one sample image per class
    names = ds.features[label_col].names
    first_index = {}
    for i, label in enumerate(ds[label_col]):
        first_index.setdefault(label, i)
    fig2, axes = plt.subplots(5, 8, figsize=(16, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, label in zip(axes.flat, sorted(first_index)):
        ax.imshow(ds[first_index[label]][image_col])
        ax.set_title(names[label], fontsize=6)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "samples.png"), dpi=150)

    # 3. class proportions per split should match the full dataset
    splits = split_dataset(ds, label_col)
    proportions = pd.DataFrame(
        {"full": class_distribution(ds, label_col)}
        | {name: class_distribution(split, label_col) for name, split in splits.items()}
    )
    proportions = proportions / proportions.sum()
    fig3, ax = plt.subplots(figsize=(14, 5))
    proportions.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("Share of split")
    ax.set_title("Class proportions per split (stratified 70/15/15)")
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.output_dir, "split_check.png"), dpi=150)
    print({name: len(split) for name, split in splits.items()})

    if task:
        logger = task.get_logger()
        for title, fig in [("Class distribution", fig1), ("Samples", fig2), ("Split check", fig3)]:
            logger.report_matplotlib_figure(title, "eda", figure=fig, iteration=0)


if __name__ == "__main__":
    main()
