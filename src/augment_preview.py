"""Figure of what each augmentation level does to a few training images (report section 2).

    python -m src.augment_preview                     # outputs/eda/augmentation_levels.png
    python -m src.augment_preview --images 4 --copies 3
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from src.transforms import augmentation_steps


def augmented_copies(image, level, copies, img_size=128, seed=0):
    """`copies` randomly augmented versions of a PIL image, as PIL images of img_size x img_size."""
    pipeline = transforms.Compose([transforms.Resize((img_size, img_size)), *augmentation_steps(level, img_size)])
    torch.manual_seed(seed)
    return [pipeline(image) for _ in range(copies)]


def preview_figure(images, titles=None, levels=("none", "basic", "strong"), copies=2, img_size=128):
    """One row per image; for each level, `copies` columns of augmented versions."""
    cols = len(levels) * copies
    fig, axes = plt.subplots(len(images), cols, figsize=(1.6 * cols, 1.7 * len(images)), squeeze=False)
    for r, image in enumerate(images):
        for li, level in enumerate(levels):
            for c, out in enumerate(augmented_copies(image, level, copies, img_size, seed=r * 100 + li)):
                ax = axes[r][li * copies + c]
                ax.imshow(np.asarray(out))
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(level, fontsize=9)
        if titles:
            axes[r][0].set_ylabel(titles[r], fontsize=7)
    fig.tight_layout()
    return fig


def main(argv=None):  # pragma: no cover - needs the real dataset
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images", type=int, default=3)
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--output-dir", default="outputs/eda")
    args = p.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    from src.data import SEED, load_full_dataset

    ds, image_col, label_col = load_full_dataset()
    names = ds.features[label_col].names
    rows = np.random.default_rng(SEED).choice(len(ds), args.images, replace=False).tolist()
    images = [ds[i][image_col].convert("RGB") for i in rows]
    titles = [names[ds[i][label_col]].replace("___", "\n").replace("_", " ") for i in rows]
    fig = preview_figure(images, titles, copies=args.copies)
    path = os.path.join(args.output_dir, "augmentation_levels.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
