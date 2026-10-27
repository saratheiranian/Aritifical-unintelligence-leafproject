"""Train a plant disease classifier and track the run in ClearML.

Examples:
    # quick prototype on 25% of the training split, logged to ClearML
    python -m src.train --subset 0.25 --epochs 10 --tags cnn subset

    # same run, but executed on the KCL GPU worker (queue name from Workers & Queues)
    python -m src.train --subset 0.25 --queue <gpu-queue-name>

    # run locally without ClearML
    python -m src.train --subset 0.05 --epochs 1 --no-clearml

Remote runs: the ClearML agent clones the repository at your last pushed commit, so commit
and push before using --queue. Batch size 64 at 128x128 (or 32 at 256x256) fits easily in the
8 GB GPU slice.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.data import make_loaders
from src.models import build_model, count_parameters, get_device

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
    p.add_argument("--freeze-backbone", action="store_true", help="resnet18 only: train the head only")
    # optimisation
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
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


class EarlyStopping:
    """Stop when validation accuracy has not improved by min_delta for `patience` epochs."""

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.epochs_without_gain = 0

    def step(self, value):
        """Record a new value. Returns True if it is the best so far."""
        if value > self.best + self.min_delta:
            self.best = value
            self.epochs_without_gain = 0
            return True
        self.epochs_without_gain += 1
        return False

    @property
    def should_stop(self):
        return self.epochs_without_gain >= self.patience


def make_scheduler(name, optimizer, epochs):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    return None


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


def save_checkpoint(path, model, args, model_kwargs, class_names, val_acc, epoch):
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
        path,
    )


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

    device = get_device()
    loaders, class_names = make_loaders(
        img_size=args.img_size,
        augment=not args.no_augment,
        batch_size=args.batch_size,
        subset_fraction=args.subset,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model_kwargs = {"num_classes": len(class_names)}
    if args.arch == "cnn":
        model_kwargs |= {
            "base_channels": args.base_channels,
            "depth": args.depth,
            "dropout": args.dropout,
            "use_batchnorm": not args.no_batchnorm,
        }
    else:
        model_kwargs |= {"pretrained": True, "freeze_backbone": args.freeze_backbone}
    model = build_model(args.arch, **model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"{args.arch}: {n_params:,} trainable parameters on {device}")
    # Saved copies are rebuilt from model_kwargs, so a fine-tuned ResNet must not re-download weights.
    saved_kwargs = {**model_kwargs, "pretrained": False} if args.arch == "resnet18" else model_kwargs

    criterion = nn.CrossEntropyLoss()
    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = opt_cls(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args.scheduler, optimizer, args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model.pt")
    stopper = EarlyStopping(args.patience, args.min_delta)
    logger = task.get_logger() if task else None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr:.2e} | {epoch_time:.0f}s"
        )
        if logger:
            logger.report_scalar("loss", "train", value=train_loss, iteration=epoch)
            logger.report_scalar("loss", "val", value=val_loss, iteration=epoch)
            logger.report_scalar("accuracy", "train", value=train_acc, iteration=epoch)
            logger.report_scalar("accuracy", "val", value=val_acc, iteration=epoch)
            logger.report_scalar("learning rate", "lr", value=lr, iteration=epoch)
            logger.report_scalar("epoch time (s)", "time", value=epoch_time, iteration=epoch)

        if stopper.step(val_acc):
            save_checkpoint(best_path, model, args, saved_kwargs, class_names, val_acc, epoch)
        if scheduler is not None:
            scheduler.step(val_acc) if args.scheduler == "plateau" else scheduler.step()
        if stopper.should_stop:
            print(f"Early stopping: no val accuracy gain for {args.patience} epochs")
            break

    total_minutes = (time.time() - start) / 60
    print(f"Best val accuracy {stopper.best:.4f} in {total_minutes:.1f} min; saved {best_path}")
    if logger:
        logger.report_single_value("best_val_accuracy", stopper.best)
        logger.report_single_value("epochs_run", epoch)
        logger.report_single_value("train_minutes", total_minutes)  # for the sustainability section
        logger.report_single_value("parameters", n_params)
    if task:
        task.close()  # lets sweep.py start the next run in the same process
    return stopper.best


if __name__ == "__main__":
    main()
