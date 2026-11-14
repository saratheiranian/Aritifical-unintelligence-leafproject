"""Train a plant disease classifier and track the run in Weights & Biases.

Examples:
    # quick prototype on 25% of the training split, logged to W&B
    python -m src.train --subset 0.25 --epochs 10 --tags cnn subset

    # run without W&B (nothing is logged online)
    python -m src.train --subset 0.05 --epochs 1 --no-wandb

Log in once with `wandb login` before the first tracked run.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from torch import nn

from src.config import WANDB_ENTITY, WANDB_PROJECT
from src.data import class_weights, make_loaders
from src.models import build_model, count_parameters, get_device

# still imported by the evaluation and data scripts, which have not moved to W&B yet
CLEARML_PROJECT = "PlantDisease"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # data
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--subset", type=float, default=1.0, help="fraction of the training split")
    p.add_argument("--augment", choices=["none", "basic", "strong"], default="basic",
                   help="training augmentation strength")
    p.add_argument("--no-augment", action="store_true", help="same as --augment none")
    p.add_argument("--balanced-sampler", action="store_true", help="draw every class equally often in training")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    # model
    p.add_argument("--arch", choices=["cnn", "rescnn", "resnet18"], default="cnn")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-batchnorm", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true", help="resnet18 only: train the head only")
    # optimisation
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="spread this much of each target's probability over the other classes")
    p.add_argument("--class-weights", action="store_true",
                   help="weight the training loss by inverse class frequency")
    p.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=3, help="early stopping patience (epochs)")
    p.add_argument("--min-delta", type=float, default=0.002, help="min val accuracy gain to count")
    # tracking
    p.add_argument("--task-name", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--no-wandb", action="store_true", help="do not log the run to W&B")
    # old name, kept so existing commands still work
    p.add_argument("--no-clearml", dest="no_wandb", action="store_true", help=argparse.SUPPRESS)
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


def dataset_labels(dataset):
    """Training labels without loading images (LeafDataset), or by iterating any other dataset."""
    if hasattr(dataset, "ds") and hasattr(dataset, "label_col"):
        return list(dataset.ds[dataset.label_col])
    return [int(dataset[i][1]) for i in range(len(dataset))]


def make_criterion(args, train_dataset, num_classes, device):
    """Training loss: cross-entropy, optionally with label smoothing and inverse-frequency weights."""
    weight = None
    if args.class_weights:
        labels = dataset_labels(train_dataset)
        weight = torch.tensor(class_weights(labels, num_classes), dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)


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

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=args.task_name or f"train-{args.arch}",
            tags=args.tags,
            config=vars(args),
            reinit=True,  # sweep.py starts several runs in one process
        )

    device = get_device()
    loaders, class_names = make_loaders(
        img_size=args.img_size,
        augment="none" if args.no_augment else args.augment,
        batch_size=args.batch_size,
        subset_fraction=args.subset,
        num_workers=args.num_workers,
        seed=args.seed,
        balanced_sampler=args.balanced_sampler,
    )

    model_kwargs = {"num_classes": len(class_names)}
    if args.arch in ("cnn", "rescnn"):
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

    criterion = make_criterion(args, loaders["train"].dataset, len(class_names), device)
    eval_criterion = nn.CrossEntropyLoss()  # validation loss stays comparable across runs
    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = opt_cls(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(args.scheduler, optimizer, args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_model.pt")
    stopper = EarlyStopping(args.patience, args.min_delta)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], eval_criterion, device)
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr:.2e} | {epoch_time:.0f}s"
        )
        if run:
            # key names must match what compare_runs.py reads
            run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": lr,
                "epoch_time": epoch_time,
            }, step=epoch)

        if stopper.step(val_acc):
            save_checkpoint(best_path, model, args, saved_kwargs, class_names, val_acc, epoch)
        if scheduler is not None:
            scheduler.step(val_acc) if args.scheduler == "plateau" else scheduler.step()
        if stopper.should_stop:
            print(f"Early stopping: no val accuracy gain for {args.patience} epochs")
            break

    total_minutes = (time.time() - start) / 60
    print(f"Best val accuracy {stopper.best:.4f} in {total_minutes:.1f} min; saved {best_path}")
    if run:
        run.summary.update({
            "best_val_accuracy": stopper.best,
            "epochs_run": epoch,
            "train_minutes": total_minutes,  # for the sustainability section
            "parameters": n_params,
        })
        run.finish()  # lets sweep.py start the next run in the same process
    return stopper.best


if __name__ == "__main__":
    main()
