"""Evaluate a trained checkpoint on the held-out test split.

Reports accuracy, top-5 accuracy, per-class precision/recall/F1 and a confusion matrix,
and logs them to the team's Weights & Biases project (src/config.py).

    python -m src.evaluate --checkpoint outputs/best_model.pt
    python -m src.evaluate --checkpoint outputs/best_model.pt --no-wandb   # nothing logged online
"""

import argparse
import os
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import WANDB_ENTITY, WANDB_PROJECT
from src.data import make_loaders
from src.models import build_model, get_device


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, device):
    all_probs, all_labels = [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def top_k_accuracy(probs, labels, k=5):
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([label in row for row, label in zip(top_k, labels)]))


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="outputs/best_model.pt")
    p.add_argument("--run-name", "--task-name", dest="run_name", default="evaluate",
                   help="name of the W&B run")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--no-wandb", action="store_true", help="do not log to Weights & Biases")
    # old name, kept so existing commands still work
    p.add_argument("--no-clearml", dest="no_wandb", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=args.run_name,
                         job_type="eval", tags=["eval", "test"], config=vars(args))
        print(f"W&B run ID: {run.id}")

    with closing_wandb_runs([run] if run is not None else []):  # no run with --no-wandb
        return run_evaluation(args, run)


@contextmanager
def closing_wandb_runs(runs):
    """Close every W&B run in `runs` when the block ends, so none is left marked as running.

    A block that completes closes them normally. A block that raises an error, or is stopped with
    Ctrl+C, closes them as failed (exit code 1), so a crashed run is not mistaken for a finished
    result; the error still reaches the user. Runs added to the list inside the block are closed
    too. Shared by evaluate.py, robustness.py and gradcam.py."""
    try:
        yield runs
    except BaseException:  # also Ctrl+C: the run did not finish either
        for run in runs:
            run.finish(exit_code=1)
        raise
    for run in runs:
        run.finish()


def run_evaluation(args, run=None):
    """Evaluate args.checkpoint on the test split; log to `run` if one is given."""
    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    loaders, class_names = make_loaders(
        img_size=ckpt["img_size"],
        augment=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    probs, labels = predict(model, loaders["test"], device)
    preds = probs.argmax(axis=1)

    acc = float(np.mean(preds == labels))
    top5 = top_k_accuracy(probs, labels, k=5)
    print(f"Test accuracy {acc:.4f} | top-5 accuracy {top5:.4f}")

    report = classification_report(
        labels, preds, labels=range(len(class_names)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    print(report_df.round(3).to_string())

    os.makedirs(args.output_dir, exist_ok=True)
    report_df.to_csv(os.path.join(args.output_dir, "test_report.csv"))
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    fig = plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png"))

    macro_f1 = float(report_df.loc["macro avg", "f1-score"])

    if run:
        import wandb

        worst = report_df.iloc[: len(class_names)].sort_values("f1-score").head(5)
        run.log({
            "test/per_class": wandb.Table(dataframe=report_df.reset_index(names="class")),
            "test/weakest_classes": wandb.Table(dataframe=worst.reset_index(names="class")),
            "test/confusion_matrix": wandb.Image(fig),
        })
        run.summary["test_accuracy"] = acc
        run.summary["test_top5_accuracy"] = top5
        run.summary["test_macro_f1"] = macro_f1
        print(f"Logged to W&B run {run.id}: {run.url}")
    return {"accuracy": acc, "top5_accuracy": top5, "macro_f1": macro_f1}


if __name__ == "__main__":
    main()
