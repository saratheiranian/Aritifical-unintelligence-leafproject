"""Fine-tune the final model on real field photos (PlantDoc train) and measure what it gains.

The final model scores 99% on PlantVillage lab photos but 14% on PlantDoc field photos
(src/field_test.py). This script checks how much of that gap a little real field data closes:

  1. takes the PlantDoc *train* photos of the 27 classes we share (the test photos stay unseen),
     dropping any train photo that looks identical to a test photo (perceptual hash), so the
     field test stays clean;
  2. keeps 15% of them per class as a field validation set;
  3. fine-tunes the model on the rest, mixed with a few PlantVillage images per class
     ("replay"), so it does not forget the lab photos;
  4. reports accuracy on the PlantDoc test photos and the PlantVillage test split, before and after.

    python -m src.finetune_field --checkpoint outputs/final_model.pt --plantdoc-dir data/PlantDoc
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.data_quality import average_hash
from src.evaluate import load_checkpoint, predict
from src.field_test import ImageListDataset, field_report, list_images, resolve_mapping
from src.models import get_device
from src.train import run_epoch, set_seed
from src.transforms import get_transforms


def mapped_items(root, class_names):
    """[(path, class index)] for the PlantDoc folders under root that match a model class."""
    folders = sorted(d.name for d in Path(root).iterdir() if d.is_dir())
    mapping, _ = resolve_mapping(folders, class_names)
    return list_images(root, mapping)


def image_hash(path):
    with Image.open(path) as img:
        return average_hash(img)


def drop_test_lookalikes(train_items, test_items, hash_fn=image_hash):
    """Remove train photos whose perceptual hash equals that of any test photo."""
    test_hashes = {hash_fn(path) for path, _ in test_items}
    kept = [(path, label) for path, label in train_items if hash_fn(path) not in test_hashes]
    return kept, len(train_items) - len(kept)


def split_per_class(items, val_fraction, seed):
    """Split [(path, label)] into train and validation, keeping val_fraction of every class
    (at least one image when a class has two or more)."""
    by_class = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        group = sorted(by_class[label])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val += group[:n_val]
        train += group[n_val:]
    return train, val


def replay_indices(labels, per_class, seed):
    """Up to per_class random positions for every label value, sorted."""
    by_class = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)
    rng = random.Random(seed)
    chosen = []
    for label in sorted(by_class):
        positions = by_class[label]
        chosen += rng.sample(positions, min(per_class, len(positions)))
    return sorted(chosen)


def load_plantvillage(img_size, augment):  # pragma: no cover - downloads the real dataset
    """(train split as a Dataset, test split as a Dataset, train labels) from PlantVillage."""
    from src.data import LeafDataset, load_full_dataset, split_dataset

    ds, image_col, label_col = load_full_dataset()
    splits = split_dataset(ds, label_col)
    train = LeafDataset(splits["train"], image_col, label_col, get_transforms(img_size, augment=augment))
    test = LeafDataset(splits["test"], image_col, label_col, get_transforms(img_size, augment=False))
    return train, test, list(splits["train"][label_col])


def field_scores(model, items, class_names, img_size, device, batch_size):
    loader = DataLoader(ImageListDataset(items, get_transforms(img_size, augment=False)), batch_size=batch_size)
    probs, labels = predict(model, loader, device)
    summary, _ = field_report(probs, labels, class_names)
    return summary


def lab_accuracy(model, dataset, device, batch_size):
    probs, labels = predict(model, DataLoader(dataset, batch_size=batch_size), device)
    return float(np.mean(probs.argmax(axis=1) == labels))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", default="outputs/final_model.pt")
    p.add_argument("--plantdoc-dir", default="data/PlantDoc", help="folder with train/ and test/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--augment", default="strong", choices=["none", "basic", "strong"])
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--replay-per-class", type=int, default=30,
                   help="PlantVillage training images per class mixed in (0 = none, no lab test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/field_finetuned.pt")
    p.add_argument("--clearml", action="store_true", help="log the run to ClearML")
    args = p.parse_args(argv)
    set_seed(args.seed)

    task = None
    if args.clearml:  # pragma: no cover - needs a ClearML server
        from clearml import Task

        from src.train import CLEARML_PROJECT

        task = Task.init(project_name=CLEARML_PROJECT, task_name="finetune-field",
                         tags=["plantdoc", "finetune"], output_uri=True)

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    class_names, img_size = ckpt["class_names"], ckpt["img_size"]
    root = Path(args.plantdoc_dir)

    test_items = mapped_items(root / "test", class_names)
    train_items, dropped = drop_test_lookalikes(mapped_items(root / "train", class_names), test_items)
    ft_train, ft_val = split_per_class(train_items, args.val_fraction, args.seed)
    print(f"PlantDoc: {len(ft_train)} train, {len(ft_val)} val, {len(test_items)} test photos "
          f"({dropped} train photos dropped as copies of test photos)")

    train_parts = [ImageListDataset(ft_train, get_transforms(img_size, augment=args.augment))]
    lab_test = None
    if args.replay_per_class > 0:  # pragma: no cover - needs the real dataset
        pv_train, lab_test, pv_labels = load_plantvillage(img_size, args.augment)
        replay = replay_indices(pv_labels, args.replay_per_class, args.seed)
        train_parts.append(Subset(pv_train, replay))
        print(f"PlantVillage replay: {len(replay)} lab images")
    train_loader = DataLoader(ConcatDataset(train_parts), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ImageListDataset(ft_val, get_transforms(img_size, augment=False)),
                            batch_size=args.batch_size)

    before = field_scores(model, test_items, class_names, img_size, device, args.batch_size)
    lab_before = lab_accuracy(model, lab_test, device, args.batch_size) if lab_test else None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val, best_epoch = -1.0, 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| field val loss {val_loss:.4f} acc {val_acc:.4f}")
        if task:  # pragma: no cover
            logger = task.get_logger()
            logger.report_scalar("accuracy", "train", value=train_acc, iteration=epoch)
            logger.report_scalar("accuracy", "field val", value=val_acc, iteration=epoch)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save({**ckpt, "state_dict": model.state_dict(), "val_acc": val_acc, "epoch": epoch,
                        "finetuned_on": "PlantDoc train"}, args.output)

    model, _ = load_checkpoint(args.output, device)
    after = field_scores(model, test_items, class_names, img_size, device, args.batch_size)
    lab_after = lab_accuracy(model, lab_test, device, args.batch_size) if lab_test else None

    rows = {
        "field accuracy": (before["accuracy_all_classes"], after["accuracy_all_classes"]),
        "field accuracy, shared classes": (before["accuracy_shared_classes"], after["accuracy_shared_classes"]),
        "field top-5 accuracy": (before["top5_accuracy"], after["top5_accuracy"]),
        "field macro F1": (before["macro_f1"], after["macro_f1"]),
    }
    if lab_test is not None:  # pragma: no cover
        rows["lab (PlantVillage test) accuracy"] = (lab_before, lab_after)
    table = pd.DataFrame(rows, index=["before", "after"]).T
    print(f"\nBest field validation accuracy {best_val:.4f} at epoch {best_epoch}; saved {args.output}")
    print(table.round(4).to_string())
    table.to_csv(os.path.splitext(args.output)[0] + "_results.csv")

    if task:  # pragma: no cover
        for name, (b, a) in rows.items():
            task.get_logger().report_single_value(f"{name} after", a)
    return table


if __name__ == "__main__":
    main()
