# Plant disease classifier

KCL 5CCSAGAP small group project: classify plant diseases from leaf photos (39 classes,
[PlantVillage](https://huggingface.co/datasets/DScomp380/plant_village)) with a CNN built from
scratch in PyTorch, tracked in W&B, served with Gradio on Render.

## Team

Team number: 7

| Role | Name | Owns |
| --- | --- | --- |
| 1. Data lead | Sara | Data loading, split, EDA, data quality checks, balanced sampler, data card |
| 2. Model lead | Precious | CNN architectures, training loop, ResNet18 baseline |
| 3. Experiments lead | Amrit | W&B tracking, hyperparameter sweep and search, results table |
| 4. Evaluation lead | Maryam and Media | Test metrics, robustness tests, Grad-CAM, Responsible AI |
| 5. App and deployment lead | Maryam and Media | Gradio app, deployment (Render, Colab), demo video |
| 6. Project manager and quality lead | Nischal | Planning, Issues, minutes, CI and tests, report assembly, submission |

Partners review each other's pull requests: 1 and 2, 3 and 4, 5 and 6.

## Links

- Repository: https://github.com/saratheiranian/Aritifical-unintelligence-leafproject
- Deployed app (Render): https://plant-disease-classifier-7w10.onrender.com
- Final model (used by the app): ResNet18 pretrained on ImageNet and fine-tuned on PlantVillage, W&B run `rbccs1x4`, on the Hugging Face Hub as `resnet18.pt` in [Media1234/plant-disease-cnn](https://huggingface.co/Media1234/plant-disease-cnn) (99.13% test accuracy)
- Best CNN from scratch (ClearML task ID): `1bee769f0c32413e8b0ad57fdcc7d626` (output model `best_model.pt`, 99.0% test accuracy). Results log: [docs/results.md](docs/results.md)

## Status and results

| Stage | Status |
| --- | --- |
| Data split, EDA and quality checks | Done: 38,812 train / 8,317 val / 8,318 test, stratified, seed 42; ClearML Dataset `7d1a35d50f2d4fab994f3761743d0c57`; see [docs/data_card.md](docs/data_card.md) |
| Balanced sampler | Done: no accuracy gain on the 25% subset (88.9% vs 89.4% baseline) |
| CNN from scratch, full data | Done: **99.0% test accuracy**, 99.99% top-5, macro F1 0.986 (20 epochs, 64 min on a MacBook GPU) |
| Robustness tests | Done: on 7 distortions the 99% model keeps 73.2% on average; noise is the main weakness; strong augmentation keeps the largest share of its clean accuracy (77%, against 74% for the 99% model and 59% without augmentation) (see [docs/results.md](docs/results.md)) |
| Grad-CAM | Done: on all 8,318 test images, 71% of the 83 mistakes are between diseases of the same plant; the 99% model puts 64% of its heatmap on the leaf, more than the same model with random weights (51% on average); it only partly passes the sanity check, so the heatmaps are a rough guide (see [docs/results.md](docs/results.md)) |
| Gradio app (local) | Done |
| Hyperparameter sweep | To do (waiting for the ClearML GPU queue) |
| ResNet18 | Done: **99.13% test accuracy**, 100% top-5, macro F1 0.989; best model on field photos too (16.9% against 14.4% for the 99% CNN); now the final model (see [docs/results.md](docs/results.md)) |
| Deployment | Done: Render (permanent link) and a Colab notebook for demo day, both running ResNet18 |
| Report, demo video, AI use statement | To do (deadline 3 Dec 2026, 4pm) |

Weakest classes of the 99% CNN: Corn Cercospora leaf spot (F1 0.89) and Tomato Early blight (F1 0.93).
Known limitation: photos of plants outside the 39 classes still get a confident answer
(a maple leaf came out as "Background without leaves" at 97%). Full details, run IDs and
per-epoch numbers are in [docs/results.md](docs/results.md).

## Project layout

| Path | Contents |
| --- | --- |
| `src/data.py`, `src/transforms.py` | Loading PlantVillage, the 70/15/15 split, data loaders, resizing and augmentation |
| `src/models.py` | `PlantCNN` (from scratch) and the ResNet18 baseline |
| `src/train.py`, `src/evaluate.py` | Training with early stopping, test-set evaluation |
| `src/robustness.py` | Accuracy on distorted test images (blur, noise, lighting, rotation, JPEG, zoom) |
| `src/distortions.py` | The 7 image distortions used by the robustness test |
| `src/gradcam.py` | Grad-CAM heatmaps: which parts of a leaf photo each model used for its prediction |
| `src/split_data.py`, `src/eda.py`, `src/data_quality.py`, `src/register_dataset.py` | Split check, EDA plots, image quality and duplicate checks, ClearML dataset versioning |
| `src/sweep.py`, `src/compare_runs.py` | Hyperparameter experiments and the results table |
| `src/inference.py`, `app/app.py` | Prediction helper and the Gradio app |
| `src/make_app_assets.py`, `src/publish_model.py`, `src/deploy_space.py` | App images, Hugging Face model upload and Space deployment |
| `tests/` | Unit tests, run by GitHub Actions on every push and pull request |
| `docs/results.md`, `docs/data_card.md` | Results log and dataset documentation for the report |

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
wandb login                        # paste your API key from your W&B account
```

Activate the environment again in every new terminal (`source .venv/bin/activate`); a
"No module named torch" error usually means it is not active.
Training picks the best device automatically: an NVIDIA GPU on the ClearML worker, the Apple GPU (MPS) on a Mac, otherwise the CPU.

Never commit API keys, the dataset or model files (the dataset and model files are in `.gitignore`). `wandb login` keeps your W&B API key outside the repo.

## Usage

All commands run from the repo root. Training, evaluation, robustness and Grad-CAM log to the
team's Weights & Biases project (`src/config.py`); add `--no-wandb` to run them without tracking.
`eda` and `data_quality` log to W&B too (add `--no-wandb` to skip it). Only `register_dataset` still uses ClearML, because the report cites the ClearML Dataset ID it created.

### 1. Data

```bash
python -m src.split_data                   # class counts per split, stratification check
python -m src.eda                          # plots to outputs/eda/ and W&B
python -m src.register_dataset             # version the split (row indices) as a ClearML Dataset
python -m src.data_quality                 # image sizes, brightness, duplicates and split leakage
python -m src.augment_preview              # figure of the none/basic/strong augmentation levels
```

Dataset details, class balance and known limitations are in [docs/data_card.md](docs/data_card.md).
Train with `--balanced-sampler` to draw every class equally often (the classes are 36x imbalanced).
Choose the training augmentation with `--augment none|basic|strong` (default `basic`).

### 2. Train

```bash
python -m src.train --subset 0.25 --epochs 10 --tags cnn subset            # local
python -m src.train --scheduler cosine --patience 4 --tags cnn full        # full data (the 99% model)
python -m src.train --subset 0.25 --queue <gpu-queue-name> --tags cnn      # KCL GPU worker
python -m src.train --arch resnet18 --lr 3e-4 --queue <gpu-queue-name>     # transfer-learning baseline
python -m src.train --arch rescnn --subset 0.25 --epochs 10 --tags rescnn    # residual CNN from scratch
python -m src.train --label-smoothing 0.1 --class-weights --subset 0.25      # loss options
```

Architectures (`--arch`): `cnn` is a plain CNN (conv, batch norm, ReLU, max pool blocks);
`rescnn` is a small ResNet-style CNN written from scratch, with skip connections; `resnet18` is
the pretrained ImageNet ResNet18 with a new final layer. `--depth`, `--base-channels`,
`--dropout` and `--no-batchnorm` apply to both from-scratch models.

Remote runs: the ClearML agent clones the repository at your last pushed commit, so commit and push first. The queue name is on the Workers & Queues page of the ClearML server. The GPU slice has 8 GB, which is plenty for batch size 64 at 128x128 or 32 at 256x256.

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
python -m src.evaluate --checkpoint outputs/best_model.pt              # logs to W&B
python -m src.evaluate --checkpoint outputs/best_model.pt --no-wandb   # nothing logged online
python -m src.robustness --checkpoint outputs/final_model.pt outputs/cnn_augment-none.pt outputs/cnn_augment-strong.pt
python -m src.gradcam --checkpoint outputs/final_model.pt outputs/cnn_augment-none.pt outputs/cnn_augment-strong.pt
```

`evaluate` logs test accuracy, top-5 accuracy, per-class precision/recall/F1, the weakest classes
and a confusion matrix. `robustness` measures accuracy on test images with 7 distortions (blur,
noise, dark, bright, rotation, JPEG, zoom) at 3 strengths, for one or more models, and saves a
results table, a summary, a chart and how often leaf photos are read as "no leaf" (Background
without leaves) to `outputs/robustness/` (`--sample 500` for a quick run,
`--preview` for the distortion figure only). It also logs them to the team's Weights & Biases
project (`src/config.py`); add `--no-wandb` to skip that.

`gradcam` shows which parts of a photo each model used for its prediction: heatmaps for correct
predictions from 8 classes and the 4 most confident mistakes (each also explained for its true
class), random photos explained by every model, and a diseased and a healthy leaf under the
distortions of the robustness test. It also saves every mistake with its confidence and the most
confused pairs of classes, measures how much of each model's heatmap falls on the leaf rather than
the background (on random photos, with a leaf mask you can check in `gradcam_leaf_masks.png`), and
runs a sanity check against the same model with random weights. Figures and tables go to
`outputs/gradcam/` and to W&B (`--no-wandb` to skip).

### 5. App and deployment

```bash
python -m src.make_app_assets                                    # example gallery + class examples
python app/app.py --checkpoint outputs/best_model.pt             # local app at http://127.0.0.1:7860
hf auth login                                           # token with write access
python -m src.publish_model --repo-id <hf-user>/plant-disease-cnn --checkpoint outputs/best_model.pt
```

The app is deployed on Render (https://plant-disease-classifier-7w10.onrender.com) and can be started
in Colab for demo day with `notebooks/colab_demo.ipynb`; both download the model from the Hugging Face
Hub. Settings and test results are in [docs/deployment.md](docs/deployment.md). (`src/deploy_space.py`
deploys to a Hugging Face Space instead, which now needs a paid plan for Gradio.)

App features: top-5 predictions as bars and a table, an example gallery captioned with each
image's true class, a model selector (when several checkpoints are given), a low-confidence
warning, training images of the predicted class, flagging wrong predictions (saved to
`app/flagged/`; on Render's free plan this folder is wiped when the app sleeps or restarts),
batch upload, and an About tab on limitations.

### Tests

```bash
pytest --cov=src
```

Coverage target: 80% (omits: make_app_assets.py, publish_model.py, sweep.py, compare_runs.py, register_dataset.py, split_data.py).

GitHub Actions runs the same tests on every push and pull request; keep them green before merging.

## Contributing

Work on a branch (`role/task`, e.g. `data/eda`), open a pull request, and get one review before merging into `main`.

## AI use statement

Parts of this codebase were written with the help of an AI assistant (Claude) and then reviewed,
tested and adapted by the team. See the AI use statement submitted with the project for details.
