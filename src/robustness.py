r"""Robustness test: how much accuracy drops when test photos are distorted.

PlantVillage photos were taken in a lab: one leaf, a plain background and even light. Phone photos
taken in a field are often blurry, dark, noisy, tilted, zoomed in or heavily compressed. This
script applies one distortion at a time, at three strengths, to the test images only (never to
training images) and measures how each model copes.

    python -m src.robustness --preview                       # figure of every distortion only
    python -m src.robustness --checkpoint outputs/final_model.pt --sample 500 --no-wandb
    python -m src.robustness --checkpoint outputs/final_model.pt outputs/cnn_augment-none.pt \
        outputs/cnn_augment-strong.pt

Every model sees exactly the same distorted images (the noise is seeded), so the comparison is
fair. Outputs go to outputs/robustness/: robustness_results.csv (every model, distortion and
strength), robustness_summary.csv (one row per model), robustness_table.md (accuracy at every
level, ready for docs/results.md), robustness_no_leaf_table.md (share of leaf photos read as
"Background_without_leaves" at every level) and robustness_accuracy.png. Without --no-wandb the
results, summary and charts are also logged to the team's Weights & Biases project
(src/config.py).
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.config import WANDB_ENTITY, WANDB_PROJECT
from src.data import SEED, load_full_dataset, split_dataset
from src.distortions import DISTORTIONS, apply_distortion
from src.evaluate import load_checkpoint
from src.models import get_device
from src.transforms import get_transforms

OUTPUT_DIR = os.path.join("outputs", "robustness")
NO_LEAF_CLASS = "Background_without_leaves"  # the class for photos with no leaf at all

# Each model was trained once, so small gaps between models may be down to the training run
# rather than the model. The results table only marks a winner that leads by at least this much.
MIN_GAP = 0.05


def save_preview(image, path):
    """Figure with one row per distortion: the clean image, then its three strengths."""
    fig, axes = plt.subplots(len(DISTORTIONS), 4, figsize=(8, 2 * len(DISTORTIONS)))
    for row, (name, (_, strengths)) in enumerate(DISTORTIONS.items()):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("clean", fontsize=8)
        axes[row, 0].set_ylabel(name)
        for col, strength in enumerate(strengths, start=1):
            axes[row, col].imshow(apply_distortion(image, name, strength, seed=SEED))
            axes[row, col].set_title(f"{name} {strength_label(strength)}", fontsize=8)
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


def macro_f1(labels, preds):
    """Macro F1 over the classes that occur in the tested images.

    Classes missing from a --sample run are left out instead of counting as F1 = 0, which would
    pull the score down (a perfect run on 25 of the 39 classes would score 25/39 = 0.64). On the
    full test set every class occurs, so the result is the usual macro F1."""
    return float(f1_score(labels, preds, labels=np.unique(labels), average="macro",
                          zero_division=0))


@torch.no_grad()
def evaluate_distortions(models, test, image_col, label_col, indices, batch_size=64,
                         device="cpu", levels=None, no_leaf_class=None):
    """Accuracy and macro F1 of every model at every distortion level.

    Each batch of images is distorted once and then given to every model, so all models are
    tested on exactly the same images. Returns a table with one row per model and level.

    no_leaf_class: index of the "Background_without_leaves" class. When given, the column
    "no_leaf" is the share of photos that do show a leaf but were read as "no leaf", which is
    what noise seems to cause.
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
        leaf_photos = labels != no_leaf_class
        for model_name in models:
            preds = np.concatenate(predictions[model_name])
            no_leaf = (float(np.mean(preds[leaf_photos] == no_leaf_class))
                       if no_leaf_class is not None and leaf_photos.any() else np.nan)
            rows.append({
                "model": model_name,
                "distortion": name,
                "strength": strength,
                "accuracy": float(np.mean(preds == labels)),
                "macro_f1": macro_f1(labels, preds),
                "no_leaf": no_leaf,
            })
        summary = ", ".join(
            f"{r['model']} {r['accuracy']:.1%}"
            + ("" if np.isnan(r["no_leaf"]) else f" (no leaf {r['no_leaf']:.0%})")
            for r in rows[-len(models):])
        print(f"{name} {strength_label(strength)}: {summary}")
    return pd.DataFrame(rows)


def summarise(results):
    """One row per model: clean accuracy, mean accuracy over all distortions, and the drop.

    "kept" is the share of clean accuracy kept on distorted images (mean / clean). It is fairer
    than "drop" when models start from different clean accuracies: a model that starts lower
    has less room to fall, so a small drop alone can flatter it."""
    clean = results[results["distortion"] == "clean"].set_index("model")["accuracy"]
    distorted = results[results["distortion"] != "clean"].groupby("model")["accuracy"]
    summary = pd.DataFrame({
        "clean_accuracy": clean,
        "mean_distorted_accuracy": distorted.mean(),
        "worst_accuracy": distorted.min(),
    })
    summary["drop"] = summary["clean_accuracy"] - summary["mean_distorted_accuracy"]
    summary["kept"] = summary["mean_distorted_accuracy"] / summary["clean_accuracy"]
    distorted_rows = results[results["distortion"] != "clean"]
    worst = distorted_rows.loc[distorted_rows.groupby("model")["accuracy"].idxmin()]
    summary["worst_case"] = pd.Series(
        [f"{d} {strength_label(s)}" for d, s in zip(worst["distortion"], worst["strength"])],
        index=worst["model"],
    )
    if "no_leaf" in results and results["no_leaf"].notna().any():
        rows = results.dropna(subset=["no_leaf"])
        most = rows.loc[rows.groupby("model")["no_leaf"].idxmax()]
        summary["clean_no_leaf"] = rows[rows["distortion"] == "clean"].set_index("model")["no_leaf"]
        summary["max_no_leaf"] = pd.Series(list(most["no_leaf"]), index=most["model"])
        summary["max_no_leaf_level"] = pd.Series(
            [f"{d} {strength_label(s)}" for d, s in zip(most["distortion"], most["strength"])],
            index=most["model"])
    return summary.sort_values("mean_distorted_accuracy", ascending=False)


def markdown_table(results, min_gap=MIN_GAP, column="accuracy"):
    """One value per model and level as a Markdown table, for docs/results.md.

    By default the accuracy: the best model in a row is in bold only if it leads the runner-up by
    at least min_gap (5 points by default), since with one training run per model smaller gaps may
    be noise. min_gap=None leaves out the bold (used for the "no leaf" table, where lower is
    better)."""
    model_names = list(results["model"].unique())
    lines = [
        "| Distortion | Strength | " + " | ".join(model_names) + " |",
        "| --- | --- |" + " --- |" * len(model_names),
    ]
    for name, strength in all_levels():
        if name == "clean":
            level = results[results["distortion"] == "clean"]
        else:
            level = results[(results["distortion"] == name) & (results["strength"] == strength)]
        if level.empty:
            continue
        accuracy = level.set_index("model")[column]
        ranked = accuracy.sort_values(ascending=False)
        clear_winner = (min_gap is not None and len(ranked) > 1
                        and ranked.iloc[0] - ranked.iloc[1] >= min_gap - 1e-9)
        cells = []
        for model_name in model_names:
            cell = f"{accuracy[model_name]:.1%}"
            if clear_winner and model_name == ranked.index[0]:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {name} | {strength_label(strength) or '-'} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_results(results, path, num_classes=39):
    """One small chart per distortion: accuracy against strength, one line per model.

    The dashed line is the accuracy of random guessing (1 in 39), where a model has stopped
    working altogether. Saves the chart, closes the figure and returns the file path."""
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
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", default=["outputs/final_model.pt"],
                   help="one or more model files to compare")
    p.add_argument("--sample", type=int, default=None,
                   help="use a fixed random sample of this many test images (default: all 8,318)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--preview", action="store_true",
                   help="only save the distortion figure (no WandB run is started)")
    p.add_argument("--preview-index", type=int, default=0, help="which test image to show")
    p.add_argument("--run-name", default="robustness", help="name of the WandB run")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--no-wandb", action="store_true", help="do not log to Weights & Biases")
    args = p.parse_args(argv)
    if args.sample is not None and args.sample < 1:
        p.error("--sample must be at least 1")
    if args.batch_size < 1:
        p.error("--batch-size must be at least 1")
    if not args.preview:
        missing = [path for path in args.checkpoint if not os.path.exists(path)]
        if missing:
            p.error(f"checkpoint not found: {', '.join(missing)}")

    test, image_col, label_col, class_names = load_test_split()
    if not 0 <= args.preview_index < len(test):
        p.error(f"--preview-index must be between 0 and {len(test) - 1}")
    row = test[args.preview_index]
    preview = save_preview(row[image_col].convert("RGB"),
                           os.path.join(args.output_dir, "distortions_preview.png"))
    print(f"Saved {preview} (true class: {class_names[row[label_col]]})")
    if args.preview:  # a quick look at the distortions, not a result: no WandB run
        return None

    device = get_device()
    models = load_models(args.checkpoint, device)

    # Started only once the inputs are checked and the data and models load, so a mistake in the
    # command does not leave a crashed or empty run in the team's project
    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=args.run_name,
                         job_type="eval", tags=["eval", "robustness"], config=vars(args))
        print(f"WandB run ID: {run.id}")

    indices = sample_indices(len(test), args.sample)
    print(f"Testing {len(models)} model(s) on {len(indices)} test images, "
          f"{len(all_levels())} levels, on {device}")
    no_leaf_class = (class_names.index(NO_LEAF_CLASS) if NO_LEAF_CLASS in class_names else None)
    results = evaluate_distortions(models, test, image_col, label_col, indices,
                                   args.batch_size, device, no_leaf_class=no_leaf_class)
    summary = summarise(results)
    print(summary.round(3).to_string())

    results.to_csv(os.path.join(args.output_dir, "robustness_results.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "robustness_summary.csv"))
    with open(os.path.join(args.output_dir, "robustness_table.md"), "w") as f:
        f.write(markdown_table(results))
    if no_leaf_class is not None:
        with open(os.path.join(args.output_dir, "robustness_no_leaf_table.md"), "w") as f:
            f.write(markdown_table(results, min_gap=None, column="no_leaf"))
    chart = plot_results(results, os.path.join(args.output_dir, "robustness_accuracy.png"),
                         num_classes=len(class_names))
    print(f"Saved results, summary and chart to {args.output_dir}")

    if run:
        run.log({
            "robustness/results": wandb.Table(dataframe=results),
            "robustness/summary": wandb.Table(dataframe=summary.reset_index()),
            "robustness/accuracy_chart": wandb.Image(chart),
            "robustness/distortions_preview": wandb.Image(preview),
        })
        for model_name, values in summary.iterrows():
            for column in ["clean_accuracy", "mean_distorted_accuracy", "drop", "kept"]:
                run.summary[f"{model_name}/{column}"] = float(values[column])
            for column in ["clean_no_leaf", "max_no_leaf"]:  # only when the class exists
                if column in values and pd.notna(values[column]):
                    run.summary[f"{model_name}/{column}"] = float(values[column])
        print(f"Logged to WandB run {run.id}: {run.url}")
        run.finish()
    return results


if __name__ == "__main__":
    main()