"""Save example images for the Gradio app.

  app/examples/            one test-split image for each of N classes (the example gallery)
  app/class_examples/<c>/  a few training images per class (shown next to a prediction)

    python -m src.make_app_assets --n-examples 12 --per-class 3

These folders are git-ignored (no dataset files in the repo); deploy_space.py uploads them.
"""

import argparse
import os
import random

from src.data import SEED, load_full_dataset, split_dataset

APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")


def first_rows_per_class(split, label_col, per_class):
    rows = {}
    for i, label in enumerate(split[label_col]):
        bucket = rows.setdefault(label, [])
        if len(bucket) < per_class:
            bucket.append(i)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-examples", type=int, default=12)
    p.add_argument("--per-class", type=int, default=3)
    p.add_argument("--out", default=APP_DIR)
    args = p.parse_args(argv)

    ds, image_col, label_col = load_full_dataset()
    names = ds.features[label_col].names
    splits = split_dataset(ds, label_col, seed=SEED)

    # gallery examples come from the test split, so they were never seen in training
    test_rows = first_rows_per_class(splits["test"], label_col, 1)
    chosen = random.Random(SEED).sample(sorted(test_rows), min(args.n_examples, len(test_rows)))
    examples_dir = os.path.join(args.out, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    for label in chosen:
        image = splits["test"][test_rows[label][0]][image_col].convert("RGB")
        image.save(os.path.join(examples_dir, f"{names[label]}.jpg"), quality=90)

    for label, rows in first_rows_per_class(splits["train"], label_col, args.per_class).items():
        class_dir = os.path.join(args.out, "class_examples", names[label])
        os.makedirs(class_dir, exist_ok=True)
        for k, row in enumerate(rows):
            image = splits["train"][row][image_col].convert("RGB")
            image.save(os.path.join(class_dir, f"{k}.jpg"), quality=85)

    print(f"Saved {len(chosen)} gallery examples and {args.per_class} images per class to {args.out}")


if __name__ == "__main__":
    main()
