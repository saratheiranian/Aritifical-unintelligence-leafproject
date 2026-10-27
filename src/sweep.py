"""Run a set of training experiments, changing one hyperparameter at a time from a baseline.

Two modes:
  remote (recommended): clone an existing ClearML training task and enqueue each variant on
  the GPU queue. Get the base task ID from the ClearML web app (a finished `train-cnn` run).

      python -m src.sweep --base-task-id <task id> --queue <gpu-queue-name>

  local: run each variant here, one after the other (slow without a GPU).

      python -m src.sweep --local --subset 0.25 --epochs 5

All variants are tagged "sweep" plus their name, so compare_runs.py can collect them.
"""

import argparse

# One-factor-at-a-time variants of the baseline in train.py. Values are train.py arguments.
EXPERIMENTS = {
    "baseline": {},
    "lr-3e-4": {"lr": 3e-4},
    "lr-3e-3": {"lr": 3e-3},
    "adam": {"optimizer": "adam"},
    "batch-32": {"batch_size": 32},
    "batch-128": {"batch_size": 128},
    "dropout-0": {"dropout": 0.0},
    "dropout-0.5": {"dropout": 0.5},
    "no-batchnorm": {"no_batchnorm": True},
    "no-augment": {"no_augment": True},
    "depth-3": {"depth": 3},
    "depth-5": {"depth": 5},
    "channels-16": {"base_channels": 16},
    "channels-64": {"base_channels": 64},
    "cosine-lr": {"scheduler": "cosine"},
    "img-256": {"img_size": 256, "batch_size": 32},
    "resnet18": {"arch": "resnet18", "lr": 3e-4},
}


def to_cli(overrides):
    """{'lr': 3e-4, 'no_augment': True} -> ['--lr', '0.0003', '--no-augment']"""
    args = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif value is not False:
            args += [flag, str(value)]
    return args


def to_clearml_params(overrides):
    """train.py uses argparse, which ClearML stores under the 'Args/' section by dest name."""
    return {f"Args/{key}": value for key, value in overrides.items()}


def run_remote(names, base_task_id, queue, shared):
    from clearml import Task

    base = Task.get_task(task_id=base_task_id)
    for name in names:
        overrides = {**shared, **EXPERIMENTS[name]}
        task = Task.clone(source_task=base, name=f"sweep-{name}")
        task.set_parameters(to_clearml_params(overrides))
        task.set_tags(["sweep", name])
        Task.enqueue(task, queue_name=queue)
        print(f"queued {name}: {task.id}")


def run_local(names, shared):
    from src import train

    for name in names:
        overrides = {**shared, **EXPERIMENTS[name]}
        argv = to_cli(overrides) + ["--task-name", f"sweep-{name}", "--tags", "sweep", name]
        print(f"\n=== {name}: {' '.join(argv)}")
        train.main(argv)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--only", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    p.add_argument("--base-task-id", help="remote mode: ClearML task to clone")
    p.add_argument("--queue", help="remote mode: ClearML queue")
    p.add_argument("--local", action="store_true")
    p.add_argument("--subset", type=float, default=None, help="override the training subset")
    p.add_argument("--epochs", type=int, default=None, help="override the number of epochs")
    args = p.parse_args(argv)

    shared = {k: v for k, v in {"subset": args.subset, "epochs": args.epochs}.items() if v is not None}
    if args.local:
        run_local(args.only, shared)
    elif args.base_task_id and args.queue:
        run_remote(args.only, args.base_task_id, args.queue, shared)
    else:
        p.error("use --local, or give both --base-task-id and --queue")


if __name__ == "__main__":
    main()
