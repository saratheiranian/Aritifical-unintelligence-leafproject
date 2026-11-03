"""Image quality checks: sizes, colour modes, brightness, duplicates and leakage between splits.

Saves tables and plots to outputs/eda/ (use them in report section 2) and logs them to ClearML.

    python -m src.data_quality                  # every image (a few minutes)
    python -m src.data_quality --sample 5000    # quick look at a random sample

Duplicates are found with a perceptual hash: images that look the same after shrinking to
16x16 grey pixels get the same hash, even if they were saved at different sizes or qualities.
A duplicate pair split across train and test would inflate the test accuracy (leakage).
"""

import argparse
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data import SEED, load_full_dataset
from src.register_dataset import split_indices
from src.train import CLEARML_PROJECT

DARK, BRIGHT, LOW_CONTRAST = 40, 215, 15  # grey-level thresholds (0-255) for flagging images


def image_stats(image):
    """Width, height, colour mode, mean brightness and contrast (std of grey levels, 0-255)."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
    }


def average_hash(image, size=16):
    """Perceptual hash as a hex string: one bit per pixel of a size x size grey thumbnail."""
    grey = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    bits = (grey > grey.mean()).flatten()
    return np.packbits(bits).tobytes().hex()


def find_duplicates(hashes):
    """Groups of positions that share a hash; only groups with two or more images."""
    groups = defaultdict(list)
    for i, h in enumerate(hashes):
        groups[h].append(i)
    return [rows for rows in groups.values() if len(rows) > 1]


def cross_split_leaks(groups, split_of):
    """Duplicate groups whose images are in different splits (e.g. one in train, one in test)."""
    return [rows for rows in groups if len({split_of[r] for r in rows}) > 1]


def conflicting_labels(groups, labels):
    """Duplicate groups whose images have different labels (likely labelling errors)."""
    return [rows for rows in groups if len({labels[r] for r in rows}) > 1]


def flag_images(df):
    """Rows of the stats table that are very dark, very bright or have very low contrast."""
    reasons = pd.Series("", index=df.index)
    reasons[df["brightness"] < DARK] += "dark "
    reasons[df["brightness"] > BRIGHT] += "bright "
    reasons[df["contrast"] < LOW_CONTRAST] += "low-contrast "
    flagged = df[reasons != ""].copy()
    flagged["reason"] = reasons[reasons != ""].str.strip()
    return flagged


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sample", type=int, default=None, help="check only this many random images")
    p.add_argument("--output-dir", default="outputs/eda")
    p.add_argument("--no-clearml", action="store_true")
    args = p.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    task = None
    if not args.no_clearml:
        from clearml import Task

        task = Task.init(project_name=CLEARML_PROJECT, task_name="data-quality", tags=["data"])

    ds, image_col, label_col = load_full_dataset()
    names = ds.features[label_col].names
    labels = list(ds[label_col])
    split_of = {i: split for split, idx in split_indices(ds, label_col).items() for i in idx}

    rows = list(range(len(ds)))
    if args.sample and args.sample < len(ds):
        rows = sorted(np.random.default_rng(SEED).choice(len(ds), args.sample, replace=False).tolist())

    records, hashes = [], []
    for n, i in enumerate(rows):
        image = ds[i][image_col]
        records.append({"row": i, "label": names[labels[i]], "split": split_of[i], **image_stats(image)})
        hashes.append(average_hash(image))
        if n % 5000 == 0:
            print(f"checked {n}/{len(rows)} images")
    df = pd.DataFrame(records)

    # Sizes and colour modes
    sizes = df.groupby(["width", "height"]).size().rename("images").reset_index()
    modes = df["mode"].value_counts().rename("images")
    print("Image sizes:\n", sizes.to_string(index=False))
    print("Colour modes:\n", modes.to_string())

    # Brightness and contrast per class
    per_class = df.groupby("label")[["brightness", "contrast"]].mean().round(1).sort_values("brightness")
    flagged = flag_images(df)
    print(f"{len(flagged)} images flagged as dark, bright or low contrast")

    # Duplicates (positions in `rows` mapped back to dataset rows)
    groups = [[rows[j] for j in g] for g in find_duplicates(hashes)]
    leaks = cross_split_leaks(groups, split_of)
    conflicts = conflicting_labels(groups, labels)
    print(
        f"{len(groups)} duplicate groups ({sum(len(g) for g in groups)} images); "
        f"{len(leaks)} span more than one split; {len(conflicts)} have conflicting labels"
    )
    dup_table = pd.DataFrame(
        [
            {"group": g, "row": r, "label": names[labels[r]], "split": split_of[r]}
            for g, members in enumerate(groups)
            for r in members
        ],
        columns=["group", "row", "label", "split"],
    )

    sizes.to_csv(os.path.join(args.output_dir, "quality_sizes.csv"), index=False)
    per_class.to_csv(os.path.join(args.output_dir, "quality_brightness_per_class.csv"))
    flagged.to_csv(os.path.join(args.output_dir, "quality_flagged_images.csv"), index=False)
    dup_table.to_csv(os.path.join(args.output_dir, "quality_duplicates.csv"), index=False)

    fig1, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["brightness"], bins=50)
    ax.axvline(DARK, color="grey", linestyle="--")
    ax.axvline(BRIGHT, color="grey", linestyle="--")
    ax.set_xlabel("Mean brightness (0-255)")
    ax.set_ylabel("Images")
    ax.set_title("Image brightness")
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.output_dir, "quality_brightness.png"), dpi=150)

    fig2, ax = plt.subplots(figsize=(8, 10))
    per_class["brightness"].plot.barh(ax=ax)
    ax.set_xlabel("Mean brightness (0-255)")
    ax.set_title("Mean brightness per class")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, "quality_brightness_per_class.png"), dpi=150)

    if task:
        logger = task.get_logger()
        logger.report_single_value("images_checked", len(df))
        logger.report_single_value("duplicate_groups", len(groups))
        logger.report_single_value("cross_split_duplicate_groups", len(leaks))
        logger.report_single_value("conflicting_label_groups", len(conflicts))
        logger.report_single_value("flagged_images", len(flagged))
        logger.report_table("Image sizes", "quality", iteration=0, table_plot=sizes)
        logger.report_table("Brightness per class", "quality", iteration=0, table_plot=per_class)
        logger.report_matplotlib_figure("Brightness", "quality", figure=fig1, iteration=0)
        logger.report_matplotlib_figure("Brightness per class", "quality", figure=fig2, iteration=0)
    return {"duplicate_groups": len(groups), "leaks": len(leaks), "flagged": len(flagged)}


if __name__ == "__main__":
    main()
