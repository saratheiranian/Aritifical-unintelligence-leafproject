"""Train a plant disease classifier and track the run in ClearML.

Examples:
    # quick prototype on 25% of the training split, logged to ClearML
    python -m src.train --subset 0.25 --epochs 10 --tags cnn subset

    # same run, but executed on the KCL GPU worker (use your team's queue name)
    python -m src.train --subset 0.25 --queue <gpu-queue-name>

    # run locally without ClearML
    python -m src.train --subset 0.05 --epochs 1 --no-clearml
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from src.data import make_loaders
from src.models import build_model, count_parameters

CLEARML_PROJECT = "PlantDisease"  # change to the project name your ClearML guide asks for


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # data
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--subset", type=float, default=1.0, help="fraction of the training split")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    # model
    p.add_argument("--arch", choices=["cnn", "resnet18"], default="cnn")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-batchnorm", action="store_true")
    # optimisation
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3, help="early stopping patience (epochs)")
    p.add_argument("--min-delta", type=float, default=0.002, help="min val accuracy gain to count")
    # tracking
    p.add_argument("--task-name", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--queue", default=None, help="ClearML queue for remote execution")
    p.add_argument("--no-clearml", action="store_true")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    task = None
    if not args.no_clearml:
        from clearml import Task

        task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name=args.task_name or f"train-{args.arch}",
            tags=args.tags,
            output_uri=True,  # upload saved checkpoints to the ClearML server
        )
        if args.queue:
            task.execute_remotely(queue_name=args.queue)  # stops here locally, reruns on the worker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_names = make_loaders(
        img_size=args.img_size,
        augment=not args.no_augment,
        batch_size=args.batch_size,
        subset_fraction=args.subset,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model_kwargs = {
        "num_classes": len(class_names),
        "base_channels": args.base_channels,
        "depth": args.depth,
        "dropout": args.dropout,
        "use_batchnorm": not args.no_batchnorm,
    }
    model = build_model(args.arch, **model_kwargs).to(device)
    print(f"{args.arch}: {count_parameters(model):,} trainable parameters on {device}")

    criterion = nn.CrossEntropyLoss()
    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model.pt")
    best_val_acc, epochs_without_gain = 0.0, 0
    logger = task.get_logger() if task else None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        print(
            f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if logger:
            logger.report_scalar("loss", "train", value=train_loss, iteration=epoch)
            logger.report_scalar("loss", "val", value=val_loss, iteration=epoch)
            logger.report_scalar("accuracy", "train", value=train_acc, iteration=epoch)
            logger.report_scalar("accuracy", "val", value=val_acc, iteration=epoch)

        if val_acc > best_val_acc + args.min_delta:
            best_val_acc, epochs_without_gain = val_acc, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": args.arch,
                    "model_kwargs": model_kwargs,
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"Early stopping: no val accuracy gain for {args.patience} epochs")
                break

    print(f"Best val accuracy {best_val_acc:.4f}, checkpoint saved to {best_path}")
    if logger:
        logger.report_single_value("best_val_accuracy", best_val_acc)


if __name__ == "__main__":
    main()
