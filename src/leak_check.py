"""Does duplicate leakage inflate our test accuracy?

data_quality.py found near-duplicate images that sit in more than one split. If a test image has
a copy in the training set, the model may simply remember it. This script scores a trained model
on the test images WITH a duplicate in train/val ("leaked") and on the rest ("clean"), so the report
can say how much (if at all) the leak changes the headline accuracy.

    python -m src.data_quality --no-clearml          # once: writes outputs/eda/quality_duplicates.csv
    python -m src.leak_check --clearml-model-id c7b1c372f02e41ba9ec5f3b321e7183d
    python -m src.leak_check --checkpoint outputs/best_model.pt
"""

import argparse

import numpy as np
import pandas as pd


def leaked_test_positions(duplicates, test_rows):
    """Positions (0, 1, 2, ...) in the test split whose image has a duplicate in another split.

    duplicates: table with columns group, row, split (outputs/eda/quality_duplicates.csv).
    test_rows: dataset row index of each test image, in test-split order.
    """
    leaked_rows = set()
    for _, members in duplicates.groupby("group"):
        if "test" in set(members["split"]) and members["split"].nunique() > 1:
            leaked_rows |= set(members.loc[members["split"] == "test", "row"])
    return [pos for pos, row in enumerate(test_rows) if row in leaked_rows]


def leak_report(preds, labels, leaked_positions):
    """Accuracy on all test images, on the leaked ones and on the clean ones."""
    preds, labels = np.asarray(preds), np.asarray(labels)
    leaked = np.zeros(len(labels), dtype=bool)
    leaked[list(leaked_positions)] = True

    def acc(mask):
        return float(np.mean(preds[mask] == labels[mask])) if mask.any() else float("nan")

    return {
        "test_images": len(labels),
        "leaked_images": int(leaked.sum()),
        "accuracy_all": acc(np.ones(len(labels), dtype=bool)),
        "accuracy_leaked": acc(leaked),
        "accuracy_clean": acc(~leaked),
    }


def main(argv=None):  # pragma: no cover - needs the real dataset and a trained model
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", default="outputs/best_model.pt")
    p.add_argument("--clearml-model-id", default=None, help="score a model stored in ClearML instead")
    p.add_argument("--duplicates", default="outputs/eda/quality_duplicates.csv")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args(argv)

    from src.data import load_full_dataset, make_loaders
    from src.evaluate import load_checkpoint, predict
    from src.models import get_device
    from src.register_dataset import split_indices

    checkpoint = args.checkpoint
    if args.clearml_model_id:
        from clearml import InputModel

        checkpoint = InputModel(model_id=args.clearml_model_id).get_local_copy()

    device = get_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    loaders, _ = make_loaders(img_size=ckpt["img_size"], augment=False, batch_size=args.batch_size)
    probs, labels = predict(model, loaders["test"], device)

    ds, _, label_col = load_full_dataset()
    test_rows = split_indices(ds, label_col)["test"]
    positions = leaked_test_positions(pd.read_csv(args.duplicates), test_rows)
    report = leak_report(probs.argmax(axis=1), labels, positions)
    for key, value in report.items():
        print(f"{key:>16}: {value:.4f}" if isinstance(value, float) else f"{key:>16}: {value}")


if __name__ == "__main__":
    main()
