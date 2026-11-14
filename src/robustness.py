"""Robustness test: how much accuracy drops when test photos are distorted.

PlantVillage photos were taken in a lab: one leaf, a plain background and even light. Phone photos
taken in a field are often blurry, dark, noisy, tilted, zoomed in or heavily compressed. This
script applies one distortion at a time, at three strengths, to the test images only (never to
training images) and measures how each model copes.

    python -m src.robustness --preview --no-clearml          # figure of every distortion only
    python -m src.robustness --checkpoint outputs/final_model.pt --sample 500 --no-clearml
    python -m src.robustness --checkpoint outputs/final_model.pt outputs/cnn_augment-none.pt \
        outputs/cnn_augment-strong.pt

Every model sees exactly the same distorted images (the noise is seeded), so the comparison is
fair. Outputs go to outputs/robustness/: robustness_results.csv (every model, distortion and
strength), robustness_summary.csv (one row per model) and robustness_accuracy.png.
"""

import argparse
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score

from src.data import SEED, load_full_dataset, split_dataset
from src.evaluate import load_checkpoint
from src.models import get_device
from src.train import CLEARML_PROJECT
from src.transforms import get_transforms

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


# ---------------------------------------------------------------------------------------------
# Measuring accuracy on distorted images
# ---------------------------------------------------------------------------------------------


def strength_label(strength):
    """Strength as short text for tables and charts: 2.0 -> "2", 0.5 -> "0.5", None -> ""."""
    return "" if pd.isna(strength) else f"{strength:g}"


def all_levels():
    """Every (distortion, strength) pair to test, starting with the clean images."""
    levels = [("clean", None)]
    for name, (_, strengths) in DISTORTIONS.items():
        levels += [(name, strength) for strength in strengths]
    return levels


def sample_indices(n_images, sample=None, seed=SEED):
    """Positions of the test images to use: all of them, or a fixed random sample."""
    if sample is None or sample >= n_images:
        return list(range(n_images))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_images, size=sample, replace=False))


def load_models(checkpoint_paths, device):
    """{model name: (model, test-time transform)}; the name is the file name without .pt."""
    models = {}
    for path in checkpoint_paths:
        model, ckpt = load_checkpoint(path, device)
        models[Path(path).stem] = (model, get_transforms(ckpt["img_size"], augment=False))
    return models


@torch.no_grad()
def evaluate_distortions(models, test, image_col, label_col, num_classes, indices,
                         batch_size=64, device="cpu", levels=None):
    """Accuracy and macro F1 of every model at every distortion level.

    Each batch of images is distorted once and then given to every model, so all models are
    tested on exactly the same images. Returns a table with one row per model and level.
    """
    rows = []
    for name, strength in levels or all_levels():
        predictions = {model_name: [] for model_name in models}
        labels = []
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            data = test[batch]
            # seed = image position, so every image gets its own (but repeatable) noise
            images = [
                apply_distortion(image.convert("RGB"), name, strength, seed=SEED + position)
                for image, position in zip(data[image_col], batch)
            ]
            labels += list(data[label_col])
            for model_name, (model, transform) in models.items():
                inputs = torch.stack([transform(image) for image in images]).to(device)
                predictions[model_name].append(model(inputs).argmax(dim=1).cpu().numpy())
        labels = np.array(labels)
        for model_name in models:
            preds = np.concatenate(predictions[model_name])
            rows.append({
                "model": model_name,
                "distortion": name,
                "strength": strength,
                "accuracy": float(np.mean(preds == labels)),
                "macro_f1": float(f1_score(labels, preds, labels=range(num_classes),
                                           average="macro", zero_division=0)),
            })
        summary = ", ".join(f"{r['model']} {r['accuracy']:.1%}" for r in rows[-len(models):])
        print(f"{name} {strength_label(strength)}: {summary}")
    return pd.DataFrame(rows)


def summarise(results):
    """One row per model: clean accuracy, mean accuracy over all distortions, and the drop."""
    clean = results[results["distortion"] == "clean"].set_index("model")["accuracy"]
    distorted = results[results["distortion"] != "clean"].groupby("model")["accuracy"]
    summary = pd.DataFrame({
        "clean_accuracy": clean,
        "mean_distorted_accuracy": distorted.mean(),
        "worst_accuracy": distorted.min(),
    })
    summary["drop"] = summary["clean_accuracy"] - summary["mean_distorted_accuracy"]
    distorted_rows = results[results["distortion"] != "clean"]
    worst = distorted_rows.loc[distorted_rows.groupby("model")["accuracy"].idxmin()]
    summary["worst_case"] = pd.Series(
        [f"{d} {strength_label(s)}" for d, s in zip(worst["distortion"], worst["strength"])],
        index=worst["model"],
    )
    return summary.sort_values("mean_distorted_accuracy", ascending=False)


def plot_results(results, path, num_classes=39):
    """One small chart per distortion: accuracy against strength, one line per model.

    The dashed line is the accuracy of random guessing (1 in 39), where a model has stopped
    working altogether."""
    names = list(DISTORTIONS)
    clean = results[results["distortion"] == "clean"].set_index("model")["accuracy"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=True)
    for ax, name in zip(axes.flat, names):
        strengths = DISTORTIONS[name][1]
        for model_name in results["model"].unique():
            part = results[(results["model"] == model_name) & (results["distortion"] == name)]
            values = [clean[model_name]] + [
                part.loc[part["strength"] == s, "accuracy"].iloc[0] for s in strengths
            ]
            ax.plot(range(len(values)), [100 * v for v in values], marker="o", label=model_name)
        ax.axhline(100 / num_classes, color="grey", linestyle="--", linewidth=1,
                   label=f"random guess ({100 / num_classes:.1f}%)")
        ax.set_title(name)
        ax.set_xticks(range(len(strengths) + 1), ["clean"] + [strength_label(s) for s in strengths])
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test accuracy (%)")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(*axes.flat[0].get_legend_handles_labels(), loc="center", title="Model")
    fig.suptitle("Accuracy on distorted test images (mild to severe)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", default=["outputs/final_model.pt"],
                   help="one or more model files to compare")
    p.add_argument("--sample", type=int, default=None,
                   help="use a fixed random sample of this many test images (default: all 8,318)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--preview", action="store_true", help="only save the distortion figure")
    p.add_argument("--preview-index", type=int, default=0, help="which test image to show")
    p.add_argument("--task-name", default="robustness")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--no-clearml", action="store_true")
    args = p.parse_args(argv)

    task = None
    if not args.no_clearml:
        from clearml import Task

        task = Task.init(project_name=CLEARML_PROJECT, task_name=args.task_name,
                         tags=["eval", "robustness"])

    test, image_col, label_col, class_names = load_test_split()
    row = test[args.preview_index]
    preview = save_preview(row[image_col].convert("RGB"),
                           os.path.join(args.output_dir, "distortions_preview.png"))
    print(f"Saved {preview} (true class: {class_names[row[label_col]]})")
    if args.preview:
        return None

    device = get_device()
    models = load_models(args.checkpoint, device)
    indices = sample_indices(len(test), args.sample)
    print(f"Testing {len(models)} model(s) on {len(indices)} test images, "
          f"{len(all_levels())} levels, on {device}")
    results = evaluate_distortions(models, test, image_col, label_col, len(class_names),
                                   indices, args.batch_size, device)
    summary = summarise(results)
    print(summary.round(3).to_string())

    results.to_csv(os.path.join(args.output_dir, "robustness_results.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "robustness_summary.csv"))
    fig = plot_results(results, os.path.join(args.output_dir, "robustness_accuracy.png"),
                       num_classes=len(class_names))
    print(f"Saved results, summary and chart to {args.output_dir}")

    if task:
        logger = task.get_logger()
        logger.report_table("Robustness results", "test", iteration=0, table_plot=results)
        logger.report_table("Robustness summary", "test", iteration=0, table_plot=summary)
        logger.report_matplotlib_figure("Accuracy on distorted images", "test", figure=fig,
                                        iteration=0)
        logger.report_image("Distortions", "preview", iteration=0, local_path=preview)
    plt.close(fig)
    return results


if __name__ == "__main__":
    main()
