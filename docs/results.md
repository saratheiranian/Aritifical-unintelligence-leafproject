# Results log

Running record of experiments, for the report's data, evaluation and project-management
sections. Newest first. Runs below were tracked on app.clear.ml (project `PlantDisease`)
while the KCL ClearML server was down.

## 27 Oct 2026: first full training run (Sara, MacBook, Apple GPU)

### Data split (`python -m src.split_data`)

| Split | Images | Share |
| --- | --- | --- |
| Train | 38,812 | 70% |
| Validation | 8,317 | 15% |
| Test | 8,318 | 15% |
| **Total** | **55,447** | |

- Stratified by class with seed 42. The largest difference between a class's share in any
  split and in the full dataset is 0.0001, so every split keeps the original distribution.
- Strong class imbalance: from 152 images (Potato healthy) to 5,507 (Orange Haunglongbing),
  about 36x. Other small classes: Apple cedar rust (275), Peach healthy (360),
  Raspberry healthy (371), Tomato mosaic virus (373).

### Training runs

| Run | ClearML task | Data | Settings | Epochs | Best val acc | Test acc | Test top-5 | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Smoke test | (no ClearML) | 5% of train | defaults | 1 | 30.5% | – | – | 0.9 min (CPU) |
| Subset baseline | `c5f26a161be6488da116cf6e0545b898` | 25% of train | defaults, no scheduler | 10 | – | 65.9% | 92.3% | – |
| **Full data** | `1bee769f0c32413e8b0ad57fdcc7d626` | 100% of train | cosine LR, patience 4 | 20 | **99.0%** | **99.0%** | **99.99%** | 64 min (MPS) |

Shared settings: `PlantCNN` (4 conv blocks, 32 base channels, batchnorm, dropout 0.3,
399,399 parameters), 128x128 input, AdamW lr 1e-3, weight decay 1e-4, batch size 64,
augmentation (flip, rotation 15°, colour jitter) on training images only.

Full-data run, validation accuracy per epoch (from the ClearML log):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 19 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Val acc | 80.5% | 87.5% | 91.0% | 93.7% | 93.3% | 94.4% | 95.3% | 96.7% | 97.2% | 96.5% | 98.2% | 98.1% | 99.0% | 99.1% |

Train accuracy stayed at or below validation accuracy (98.3% at epoch 20), so no overfitting;
augmentation and dropout make training batches harder than validation ones.

The jump from 65.9% to 99.0% test accuracy comes from using 4x more training data, training
for 20 epochs, and the cosine learning-rate schedule. The sweep should separate these effects.

### Test-set evaluation of the full-data model (`python -m src.evaluate`)

- Accuracy 99.0%, top-5 accuracy 99.99%, macro F1 0.986, weighted F1 0.990.
- Every class has F1 ≥ 0.89, including the smallest classes, which scored 0% after the
  subset run (Potato healthy 0.978, Apple cedar rust 0.988, Grape healthy 1.000).
- Weakest classes:

| Class | Precision | Recall | F1 | Test images |
| --- | --- | --- | --- | --- |
| Corn: Cercospora leaf spot / Gray leaf spot | 0.918 | 0.870 | 0.893 | 77 |
| Tomato: Early blight | 0.978 | 0.887 | 0.930 | 150 |
| Corn: Northern leaf blight | 0.933 | 0.946 | 0.939 | 147 |

- Error analysis: the two corn diseases are mostly confused with each other; both show grey-brown
  elongated lesions. See `outputs/confusion_matrix.png` (not committed) or the ClearML eval task.

### App check (`python app/app.py`)

- Runs locally with the full-data model; gallery examples are classified correctly
  (e.g. Cherry healthy at 100%).
- **Out-of-distribution failure:** a photo of an autumn maple leaf held in a hand (not one of
  the 39 classes) was classified as "Background without leaves" with 97% confidence. The
  low-confidence warning does not trigger because the model is confident. Use this in the
  Responsible AI / limitations section and the demo video.

## Next

- Hyperparameter sweep (`src/sweep.py`) on the KCL GPU queue once the server is back,
  then `src/compare_runs.py` for the results table.
- ResNet18 transfer-learning baseline.
- Deploy to a Hugging Face Space and record the link in the README.
