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
- Final model (ClearML model ID / task ID): _TBD_

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
clearml-init                       # paste credentials from the KCL ClearML server (Settings > Workspace)
```

Never commit `clearml.conf`, the dataset or model files (they are in `.gitignore`).

## Usage

All commands run from the repo root.

```bash
# data: class counts and the stratified 70/15/15 split (seed 42)
python -m src.split_data
python -m src.eda                          # plots to outputs/eda/ and ClearML

# train: quick prototype on 25% of the training split
python -m src.train --subset 0.25 --epochs 10 --tags cnn subset
python -m src.train --subset 0.25 --queue <gpu-queue-name>    # on the KCL GPU worker
python -m src.train --subset 0.05 --epochs 1 --no-clearml     # local smoke test

# evaluate the best checkpoint on the test split
python -m src.evaluate --checkpoint outputs/best_model.pt

# app
python app/app.py --checkpoint outputs/best_model.pt

# tests
pytest --cov=src
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/data.py` | Load the HF dataset, stratified split, subset, PyTorch `Dataset` and loaders |
| `src/transforms.py` | Resize, normalisation and train-only augmentation |
| `src/split_data.py` | Prints class counts for full/train/val/test and checks stratification |
| `src/eda.py` | Class distribution, sample grid and split-check plots |
| `src/models.py` | `PlantCNN` (from scratch) and ResNet18 transfer-learning baseline |
| `src/train.py` | Training loop with early stopping and ClearML logging |
| `src/evaluate.py` | Test accuracy, top-5 accuracy, per-class metrics, confusion matrix |
| `app/app.py` | Gradio UI (upload a leaf, see top-5 predictions) |
| `tests/` | Pytest unit tests, run by GitHub Actions on every pull request |

## Deploying to Hugging Face Spaces

Create a public Gradio Space, then copy into it: `app/app.py` as `app.py`, the `src/` folder,
`app/requirements.txt` as `requirements.txt`, and the trained checkpoint as `model.pt`.

## Contributing

Work on a branch (`role/task`, e.g. `data/eda`), open a pull request, and get one review before merging into `main`.
