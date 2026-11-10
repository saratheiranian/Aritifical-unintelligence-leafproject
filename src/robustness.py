"""Robustness test: how much accuracy drops when test photos are distorted.

PlantVillage photos were taken in a lab: one leaf, a plain background and even light. Phone photos
taken in a field are often blurry, dark, noisy, tilted, zoomed in or heavily compressed. This
script applies one distortion at a time, at three strengths, to the test images only (never to
training images) and measures how each model copes.

    python -m src.robustness        # figure of every distortion on one test leaf
"""

import argparse
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.data import SEED, load_full_dataset, split_dataset

OUTPUT_DIR = os.path.join("outputs", "robustness")


# ---------------------------------------------------------------------------------------------
# Distortions. Each one takes a PIL image and a strength, and returns a new PIL image of the
# same size. Real-life problem each one imitates is in its docstring.
# ---------------------------------------------------------------------------------------------


def gaussian_blur(image, sigma):
    """Out-of-focus or shaky photo. sigma = blur radius in pixels."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_noise(image, std, seed=SEED):
    """Grainy photo taken in low light. std = noise strength on the 0-255 pixel scale."""
    rng = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.float32)
    noisy = pixels + rng.normal(0.0, std, size=pixels.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def brightness(image, factor):
    """Shade (factor below 1) or strong sunlight (factor above 1). 1 = unchanged."""
    return ImageEnhance.Brightness(image).enhance(factor)


def border_colour(image):
    """Average colour of the image's outer edge (its background)."""
    pixels = np.asarray(image)
    edges = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    return tuple(int(v) for v in edges.mean(axis=0))


def rotation(image, degrees):
    """Photo taken at an angle. The empty corners are filled with the background colour rather
    than black, so the model cannot use black corners as a clue."""
    return image.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=border_colour(image))


def jpeg_compression(image, quality):
    """Photo sent through a messaging app. quality 100 = best, 1 = worst."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def zoom(image, keep):
    """Close-up showing only the middle of the leaf. keep = share of the width and height kept."""
    width, height = image.size
    w, h = int(width * keep), int(height * keep)
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h)).resize(
        (width, height), Image.Resampling.BILINEAR
    )


# name: (function, three strengths from mild to severe)
DISTORTIONS = {
    "blur": (gaussian_blur, [1, 2, 4]),
    "noise": (gaussian_noise, [10, 25, 50]),
    "dark": (brightness, [0.7, 0.5, 0.3]),
    "bright": (brightness, [1.3, 1.6, 2.0]),
    "rotation": (rotation, [30, 60, 90]),
    "jpeg": (jpeg_compression, [30, 15, 5]),
    "zoom": (zoom, [0.8, 0.6, 0.4]),
}


def apply_distortion(image, name, strength, seed=SEED):
    """Apply one distortion by name. The name "clean" returns the image unchanged."""
    if name == "clean":
        return image
    function, _ = DISTORTIONS[name]
    if function is gaussian_noise:
        return function(image, strength, seed=seed)
    return function(image, strength)


def save_preview(image, path):
    """Figure with one row per distortion: the clean image, then its three strengths."""
    fig, axes = plt.subplots(len(DISTORTIONS), 4, figsize=(8, 2 * len(DISTORTIONS)))
    for row, (name, (_, strengths)) in enumerate(DISTORTIONS.items()):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("clean", fontsize=8)
        axes[row, 0].set_ylabel(name)
        for col, strength in enumerate(strengths, start=1):
            axes[row, col].imshow(apply_distortion(image, name, strength))
            axes[row, col].set_title(f"{name} {strength}", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_test_split():
    """The same test split as train.py and evaluate.py (stratified, seed 42)."""
    ds, image_col, label_col = load_full_dataset()
    test = split_dataset(ds, label_col, seed=SEED)["test"]
    return test, image_col, label_col, ds.features[label_col].names


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--preview-index", type=int, default=0, help="which test image to show")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    args = p.parse_args(argv)

    test, image_col, label_col, class_names = load_test_split()
    row = test[args.preview_index]
    path = save_preview(row[image_col].convert("RGB"),
                        os.path.join(args.output_dir, "distortions_preview.png"))
    print(f"Saved {path} (true class: {class_names[row[label_col]]})")


if __name__ == "__main__":
    main()
