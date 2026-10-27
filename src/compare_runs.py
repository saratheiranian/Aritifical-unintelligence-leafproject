"""Collect finished ClearML training runs into one results table for the report.

    python -m src.compare_runs --tag sweep

Writes outputs/results.csv and outputs/results.md (a Markdown table to paste into the report).
"""

import argparse
import os

import pandas as pd

from src.train import CLEARML_PROJECT

PARAMS = ["arch", "lr", "optimizer", "batch_size", "dropout", "no_batchnorm", "no_augment",
          "depth", "base_channels", "scheduler", "img_size", "subset"]


def summarise(task):
    params = task.get_parameters_as_dict().get("Args", {})
    scalars = task.get_reported_scalars()
    val = scalars.get("accuracy", {}).get("val", {}).get("y", [])
    train = scalars.get("accuracy", {}).get("train", {}).get("y", [])
    times = scalars.get("epoch time (s)", {}).get("time", {}).get("y", [])
    row = {"name": task.name, "id": task.id}
    row |= {key: params.get(key) for key in PARAMS}
    row["best_val_acc"] = max(val) if val else None
    row["best_train_acc"] = max(train) if train else None
    row["epochs"] = len(val)
    row["mean_epoch_seconds"] = sum(times) / len(times) if times else None
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="sweep")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args(argv)

    from clearml import Task

    tasks = Task.get_tasks(
        project_name=CLEARML_PROJECT,
        tags=[args.tag],
        task_filter={"status": ["completed", "stopped"]},
    )
    if not tasks:
        print(f"No finished tasks tagged '{args.tag}' in {CLEARML_PROJECT}")
        return
    df = pd.DataFrame([summarise(t) for t in tasks]).sort_values("best_val_acc", ascending=False)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    with open(os.path.join(args.output_dir, "results.md"), "w") as f:
        f.write(df.drop(columns=["id"]).to_markdown(index=False, floatfmt=".4f"))
    print(df.drop(columns=["id"]).to_string(index=False))


if __name__ == "__main__":
    main()
