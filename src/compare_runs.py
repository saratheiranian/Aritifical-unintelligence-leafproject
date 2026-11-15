"""Collect finished W&B training runs into one results table for the report.

    python -m src.compare_runs --tag sweep

Writes outputs/results.csv and outputs/results.md (a Markdown table to paste into the report).
Only runs W&B marks as finished are included, so crashed or stopped runs stay out of the table.
"""

import argparse
import os

import pandas as pd
import wandb

from src.config import WANDB_ENTITY, WANDB_PROJECT

PARAMS = ["arch", "lr", "optimizer", "batch_size", "dropout", "no_batchnorm", "no_augment",
          "depth", "base_channels", "scheduler", "img_size", "subset"]


def column(history, key):
    if key not in history.columns:
        return pd.Series(dtype=float)
    return history[key].dropna()


def summarise(run):
    history = run.history(pandas=True)
    val = column(history, "val_acc")
    train = column(history, "train_acc")
    times = column(history, "epoch_time")

    row = {"name": run.name, "id": run.id}
    row |= {key: run.config.get(key) for key in PARAMS}
    row["best_val_acc"] = val.max() if len(val) else None
    row["best_train_acc"] = train.max() if len(train) else None
    row["epochs"] = len(val)
    row["mean_epoch_seconds"] = times.mean() if len(times) else None
    row["url"] = run.url
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="sweep")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args(argv)

    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    filters = {"tags": {"$in": [args.tag]}, "state": "finished"}
    try:
        runs = wandb.Api().runs(path, filters=filters)
        rows = [summarise(run) for run in runs]
    except (wandb.errors.CommError, ValueError) as err:
        raise SystemExit(f"Could not read runs from W&B project {path}: {err}")

    if not rows:
        print(f"No finished runs tagged '{args.tag}' in {path}")
        return

    df = pd.DataFrame(rows).sort_values("best_val_acc", ascending=False, na_position="last")
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    table = df.drop(columns=["id", "url"])
    with open(os.path.join(args.output_dir, "results.md"), "w") as f:
        f.write(table.to_markdown(index=False, floatfmt=".4f"))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()