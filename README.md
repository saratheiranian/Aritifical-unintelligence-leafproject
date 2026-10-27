# Plant disease classifier

KCL 5CCSAGAP small group project: classify plant diseases from leaf photos (39 classes,
[PlantVillage](https://huggingface.co/datasets/DScomp380/plant_village)) with a CNN built from
scratch in PyTorch, tracked in ClearML, served with Gradio on Hugging Face Spaces.

## Team

Team number: _TBD_

| Name | Role |
| --- | --- |
| _TBD_ | Project manager and report editor |
| _TBD_ | Data lead |
| _TBD_ | Model lead |
| _TBD_ | Experiments lead |
| _TBD_ | Evaluation and quality lead |
| _TBD_ | App and deployment lead |

## Links

- Repository: https://github.com/saratheiranian/Aritifical-unintelligence-leafproject
- Deployed app (HF Space): _TBD_
- Final model (ClearML task ID): `1bee769f0c32413e8b0ad57fdcc7d626` (output model `best_model.pt`, 99.0% test accuracy). Results log: [docs/results.md](docs/results.md)

## Status and results

| Stage | Status |
| --- | --- |
| Data split and EDA | Done: 38,812 train / 8,317 val / 8,318 test, stratified, seed 42 |
| CNN from scratch, full data | Done: **99.0% test accuracy**, 99.99% top-5, macro F1 0.986 (20 epochs, 64 min on a MacBook GPU) |
| Gradio app (local) | Done |
| Hyperparameter sweep | To do (waiting for the KCL ClearML GPU queue) |
| ResNet18 baseline | To do |
| Hugging Face Space | To do |
| Report, demo video, AI use statement | To do (deadline 3 Dec 2026, 4pm) |

Weakest classes: Corn Cercospora leaf spot (F1 0.89) and Tomato Early blight (F1 0.93).
Known limitation: photos of plants outside the 39 classes still get a confident answer
(a maple leaf came out as "Background without leaves" at 97%). Full details, run IDs and
per-epoch numbers are in [docs/results.md](docs/results.md).

## Project layout

| Path | Contents |
| --- | --- |
| `src/data.py`, `src/transforms.py` | Loading PlantVillage, the 70/15/15 split, data loaders, resizing and augmentation |
| `src/models.py` | `PlantCNN` (from scratch) and the ResNet18 baseline |
| `src/train.py`, `src/evaluate.py` | Training with early stopping, test-set evaluation |
| `src/split_data.py`, `src/eda.py`, `src/register_dataset.py` | Data checks, plots and ClearML dataset versioning |
| `src/sweep.py`, `src/compare_runs.py` | Hyperparameter experiments and the results table |
| `src/inference.py`, `app/app.py` | Prediction helper and the Gradio app |
| `src/make_app_assets.py`, `src/publish_model.py`, `src/deploy_space.py` | App images, Hugging Face model upload and Space deployment |
| `tests/` | Unit tests, run by GitHub Actions on every push and pull request |
| `docs/results.md` | Results log for the report |

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
clearml-init                       # paste credentials from the KCL ClearML server (Settings > Workspace)
```

Activate the environment again in every new terminal (`source .venv/bin/activate`); a
"No module named torch" error usually means it is not active. While the KCL ClearML server
is down, credentials from a free [app.clear.ml](https://app.clear.ml) account work the same way.
Training picks the best device automatically: an NVIDIA GPU on the ClearML worker, the Apple
GPU (MPS) on a Mac, otherwise the CPU.

Never commit `clearml.conf`, the dataset or model files (they are in `.gitignore`).

## Usage

All commands run from the repo root. Each script is its own ClearML task (data, train, eval),
as the brief recommends. Add `--no-clearml` to the train, evaluate and EDA scripts to run without tracking.

### 1. Data

```bash
python -m src.split_data                   # class counts per split, stratification check
python -m src.eda                          # plots to outputs/eda/ and ClearML
python -m src.register_dataset             # version the split (row indices) as a ClearML Dataset
```

### 2. Train

```bash
python -m src.train --subset 0.25 --epochs 10 --tags cnn subset            # local
python -m src.train --scheduler cosine --patience 4 --tags cnn full        # full data (the 99% model)
python -m src.train --subset 0.25 --queue <gpu-queue-name> --tags cnn      # KCL GPU worker
python -m src.train --arch resnet18 --lr 3e-4 --queue <gpu-queue-name>     # transfer-learning baseline
```

Remote runs: the ClearML agent clones the repository at your last pushed commit, so commit and
push first. The queue name is on the Workers & Queues page of the KCL ClearML server. The GPU
slice has 8 GB, which is plenty for batch size 64 at 128x128 or 32 at 256x256.

### 3. Experiments

```bash
python -m src.sweep --base-task-id <id of a finished train task> --queue <gpu-queue-name> --subset 0.25
python -m src.compare_runs --tag sweep     # outputs/results.md: table for the report
```

`src/sweep.py` lists the one-change-at-a-time variants (learning rate, optimiser, batch size,
dropout, batchnorm, augmentation, depth, channels, scheduler, image size, ResNet18).
Compare them in the ClearML web app too: select the tasks, then Compare.

### 4. Evaluate

```bash
python -m src.evaluate --checkpoint outputs/best_model.pt
python -m src.evaluate --clearml-model-id <model id> --queue <gpu-queue-name>
```

Logs test accuracy, top-5 accuracy, per-class precision/recall/F1, the weakest classes and a
confusion matrix.

### 5. App and deployment

```bash
python -m src.make_app_assets                                    # example gallery + class examples
python app/app.py --checkpoint outputs/best_model.pt             # local app at http://127.0.0.1:7860
huggingface-cli login                                            # token with write access
python -m src.publish_model --repo-id <hf-user>/plant-disease-cnn --checkpoint outputs/best_model.pt
python -m src.deploy_space --space-id <hf-user>/plant-disease-classifier --hf-model-repo <hf-user>/plant-disease-cnn
```

App features: top-5 predictions as bars and a table, an example gallery captioned with each
image's true class, a model selector (when several checkpoints are given), a low-confidence
warning, training images of the predicted class, flagging wrong predictions (saved to
`app/flagged/`), batch upload, and an About tab on limitations.

### Tests

```bash
pytest --cov=src
```

GitHub Actions runs the same tests on every push and pull request; keep them green before merging.

## Contributing

Work on a branch (`role/task`, e.g. `data/eda`), open a pull request, and get one review before merging into `main`.

## AI use statement

Parts of this codebase were written with the help of an AI assistant (Claude) and then reviewed,
tested and adapted by the team. See the AI use statement submitted with the project for details.
