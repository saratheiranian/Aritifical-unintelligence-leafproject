"""Evaluate a trained checkpoint on the held-out test split.

Reports accuracy, top-5 accuracy, per-class precision/recall/F1 and a confusion matrix,
and logs them to ClearML.

    python -m src.evaluate --checkpoint outputs/best_model.pt
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data import make_loaders
from src.models import build_model
from src.train import CLEARML_PROJECT


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
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--no-clearml", action="store_true")
    args = p.parse_args(argv)

    task = None
    if not args.no_clearml:
        from clearml import Task

        task = Task.init(project_name=CLEARML_PROJECT, task_name="evaluate", tags=["eval"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    if task:
        logger = task.get_logger()
        logger.report_single_value("test_accuracy", acc)
        logger.report_single_value("test_top5_accuracy", top5)
        logger.report_table("Per-class metrics", "test", iteration=0, table_plot=report_df)
        logger.report_matplotlib_figure("Confusion matrix", "test", figure=fig, iteration=0)


if __name__ == "__main__":
    main()
