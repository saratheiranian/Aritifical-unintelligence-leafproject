# Results log

Running record of experiments, for the report's data, evaluation and project-management
sections. Newest first. Runs below were tracked on app.clear.ml (project `PlantDisease`)
while the KCL ClearML server was down.

## 10 Nov 2026: robustness test on distorted test images (Maryam, MacBook, Apple GPU)

```bash
python -m src.robustness --checkpoint outputs/final_model.pt outputs/cnn_augment-none.pt \
    outputs/cnn_augment-strong.pt --no-clearml
```

All 8,318 test images, 7 distortions at 3 strengths each (22 levels including clean). Each batch
is distorted once and given to all three models, so they see identical images; the noise is seeded,
so the run is repeatable. Distortions are applied to the original image before the usual resize to
128x128, as they would be in a phone photo. Not logged to ClearML yet (no workspace access); re-run
without `--no-clearml` for a task ID. Figures (not committed): `outputs/robustness/robustness_accuracy.png`
and `outputs/robustness/distortions_preview.png`.

Models: `final_model` is the 99% model (all training data, 20 epochs, basic augmentation, task
`1bee769f0c32413e8b0ad57fdcc7d626`); `augment-none` and `augment-strong` are the 4 Nov subset models
(25% of the data, 10 epochs, tasks `967dd4f0b1ee494994490febdb1fc157` and
`0cc43dfc41094935a7dd024058f32ce1`). The final model differs in data and epochs too, so only none
vs strong isolates the effect of augmentation.

| Model | Clean | Mean over the 21 distorted levels | Drop (points) | Worst level |
| --- | --- | --- | --- | --- |
| final_model | 99.0% | **73.2%** | 25.8 | noise 50 (6.0%) |
| augment-strong | 80.6% | 62.2% | **18.5** | noise 50 (13.0%) |
| augment-none | 91.4% | 54.4% | 37.1 | noise 50 (5.5%) |

Test accuracy at every level (best model per row in bold; random guessing is 2.6%). Strengths:
blur radius in pixels, noise standard deviation on the 0-255 scale, share of light kept (dark),
brightness factor (bright), degrees (rotation), JPEG quality out of 100, share of the width and
height kept (zoom).

| Distortion | Strength | final_model | augment-none | augment-strong |
| --- | --- | --- | --- | --- |
| clean | - | **99.0%** | 91.4% | 80.6% |
| blur | 1 | **95.9%** | 76.1% | 82.8% |
| blur | 2 | 72.6% | 54.9% | **82.9%** |
| blur | 4 | 41.0% | 37.6% | **71.2%** |
| noise | 10 | **97.8%** | 89.0% | 77.1% |
| noise | 25 | 42.7% | **49.2%** | 47.7% |
| noise | 50 | 6.0% | 5.5% | **13.0%** |
| dark | 0.7 | **98.5%** | 41.7% | 75.1% |
| dark | 0.5 | **84.2%** | 18.4% | 55.0% |
| dark | 0.3 | **32.7%** | 13.0% | 17.4% |
| bright | 1.3 | **98.7%** | 65.9% | 79.4% |
| bright | 1.6 | **95.9%** | 38.5% | 72.1% |
| bright | 2 | **79.3%** | 18.7% | 52.3% |
| rotation | 30 | **98.2%** | 88.5% | 79.8% |
| rotation | 60 | **96.1%** | 85.8% | 78.8% |
| rotation | 90 | **96.5%** | 87.1% | 78.0% |
| jpeg | 30 | **98.7%** | 90.5% | 77.1% |
| jpeg | 15 | **91.3%** | 83.8% | 69.8% |
| jpeg | 5 | 24.5% | **35.0%** | 28.0% |
| zoom | 0.8 | **95.9%** | 82.7% | 77.8% |
| zoom | 0.6 | **64.7%** | 51.1% | 55.5% |
| zoom | 0.4 | 26.3% | 28.5% | **34.3%** |

- **Rotation does no harm.** The final model keeps 96-98% at 30-90 degrees although it was trained
  with rotations of 15 degrees at most, and the model trained without any augmentation loses at
  most 6 points. Leaves have no fixed "up", so the models never relied on orientation.
- **Noise is the main weakness of all three models.** At noise 25 they fall to 43-49%, at noise 50 to
  near random guessing. No model was trained with noise, so adding Gaussian noise to the training
  augmentation is the clearest next step.
- **Strong augmentation protects against what it trains on.** With blur 4 it keeps 71.2%, against
  41.0% for the 99% model and 37.6% without augmentation, although its clean accuracy is 18 points
  lower. It is also best at zoom 0.4 (34.3%) and noise 50 (13.0%). Strong augmentation includes
  random blur and random crops.
- **Without augmentation the model depends on lighting.** 30% less light takes it from 91.4% to
  41.7%, and doubling the brightness to 18.7%; with strong augmentation the same changes give 75.1%
  and 52.3%. This supports the data card: mean brightness differs by class (91 to 148), so a model
  trained without colour changes can use lighting as a shortcut.
- **The 99% model is the most robust on average (73.2%) but still loses 26 points**, and heavy JPEG
  compression (quality 5), zoom 0.4 and dark 0.3 take it below 33%.
- **Consistent with the field test (PlantDoc, further down):** strong augmentation also does better on real
  field photos (10.2% vs 6.4%). Field photos combine several of these changes at once, which may help
  explain the drop from 99% to 14%. Training the final model with strong augmentation plus noise,
  for more epochs, is the natural follow-up.
- A first run on a fixed sample of 500 test images gave the same picture within about 2 points.

## 9 Nov 2026: leak check (Sara)

`python -m src.leak_check --checkpoint outputs/final_model.pt` on the final 99% model
(ClearML train task `1bee769f0c32413e8b0ad57fdcc7d626`).

| Test images | With a copy in train/val | Accuracy all | Accuracy leaked | Accuracy clean |
| --- | --- | --- | --- | --- |
| 8,318 | 7 (0.08%) | 99.00% | 100% | 99.00% |

The duplicate leak does not change the headline accuracy, so the split stays as it is.
## 4 Nov 2026: augmentation study (Sara, MacBook, Apple GPU)

25% training subset, 10 epochs, default CNN, evaluated on the full test set (8,318 images).
`python -m src.train --subset 0.25 --epochs 10 --augment <level>` then `python -m src.evaluate`.

| Augmentation | Test acc | Top-5 | Macro F1 | Weighted F1 | Best val acc | ClearML train task |
| --- | --- | --- | --- | --- | --- | --- |
| none | 91.4% | 99.5% | 0.881 | 0.912 | 91.0% (epoch 9) | `967dd4f0b1ee494994490febdb1fc157` |
| basic (default) | 89.4% | | 0.867 | 0.893 | | baseline re-run, 3 Nov |
| strong | 80.6% | 98.0% | 0.760 | 0.801 | 80.7% (epoch 10) | `0cc43dfc41094935a7dd024058f32ce1` |

- Without augmentation, clean test accuracy is 2 points higher after 10 epochs: augmentation
  makes training harder and needs more epochs to pay off. Its expected benefit is on distorted
  or field-like images, so compare the checkpoints in the robustness tests too.
- Strong augmentation is clearly worse after 10 epochs (80.6%) but is underfitting, not
  failing: its training accuracy (75%) is below its validation accuracy (81%), and validation
  accuracy was still rising at epoch 10. It needs a longer run before it can be judged; its
  weakest classes are Tomato target spot (F1 0.24), Pepper bacterial spot (0.34) and Potato
  healthy (0.50). (A first strong run, task `189f4a6be0a44b24a822aaeaf8900ff4`, hung on a
  ClearML upload at epoch 8 and was discarded.)
- Weakest classes without augmentation: Grape healthy (F1 0.49, recall 0.33), Tomato early
  blight (0.67), Apple scab (0.73), Potato healthy (0.74).
- Figure of the three levels: `outputs/eda/augmentation_levels.png` (`python -m src.augment_preview`).

## 3 Nov 2026: data quality checks and balanced sampler (Sara, MacBook, Apple GPU)

### Data quality (`python -m src.data_quality`, ClearML task `98283c9e959b446384d3236bb65fd0cf`)

- All 55,447 images are RGB; 54,304 are 256x256 and 1,143 are 256x192.
- 405 images (0.7%) flagged as very dark, very bright or low contrast.
- 30 near-duplicate groups (68 images); 14 groups span more than one split, under 0.1% of the
  test set, so they were kept. No duplicates have conflicting labels.
- Mean brightness per class ranges from 91.5 (Corn Common rust) to 148.1 (Corn healthy).
- EDA task `c0dc1ce16d824f50a2213e49d9211a48`; split registered as ClearML Dataset
  `7d1a35d50f2d4fab994f3761743d0c57`.

### Balanced sampler

| Run | ClearML task | Data | Settings | Epochs | Best val acc | Test acc | Macro F1 | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Subset baseline (27 Oct) | `c5f26a161be6488da116cf6e0545b898` | 25% of train | defaults | 10 | – | 65.9% | – | – |
| Subset baseline, re-run (3 Nov) | `5166a0838c3d4484a46ca2ad6638e609` (eval) | 25% of train | defaults | 10 | 89.5% | 89.4% | 0.867 | 5.4 min (MPS) |
| Subset + balanced sampler | `e26eb9580186490c87bc562083ef4bd2` | 25% of train | `--balanced-sampler` | 10 | 88.9% | 88.9% | [from eval-balanced] | 5.5 min (MPS) |

- Re-running the baseline on the current code gives 89.4%, not 65.9%, so the 27 Oct subset result
  is not reproducible and should not be used. [Check that ClearML task's epochs and settings to find out why.]
- With the same code and data, the balanced sampler gives no accuracy gain (88.9% vs 89.4%;
  weighted F1 0.891 vs 0.893). Its effect on the smallest classes still has to be checked with the
  per-class F1 of the eval-balanced task. In the baseline: Potato healthy F1 0.800 (recall 0.696),
  Apple cedar rust 0.883, Grape healthy 0.676.
- Weakest baseline classes on the 25% subset: Tomato Early blight (F1 0.626), Grape healthy (0.676),
  Apple scab (0.715), Grape Esca (0.727), Corn Cercospora (0.732); all improve with full data.

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

## 10 Nov 2026: field-photo test (PlantDoc)

`python -m src.field_test --checkpoint outputs/final_model.pt --plantdoc-dir data/PlantDoc/test`:
the 99% model on the PlantDoc test split (Singh et al., 2020, CC BY 4.0), real field photos of the
27 classes PlantDoc shares with PlantVillage.

| Test set | Images | Classes | Accuracy | Top-5 | Macro F1 |
|---|---|---|---|---|---|
| PlantVillage test (lab photos) | 8,318 | 39 | 99.0% | | |
| PlantDoc, model picks from all 39 classes | 236 | 27 | 14.4% | 40.7% | 0.128 |
| PlantDoc, prediction limited to the 27 shared classes | 236 | 27 | 17.8% | | |

- Accuracy falls from 99% to 14% on field photos (chance is 2.6%). The model has learnt the
  PlantVillage setting (one leaf, plain background, even lighting) as much as the disease itself.
- Limiting predictions to the shared classes only adds 3 points, so the errors are not just
  confusion with classes missing from PlantDoc.
- Best field classes: Corn common rust (F1 0.44), Squash powdery mildew (0.43), Tomato late
  blight (0.37), Potato early blight (0.35): large, distinctive lesions.
- 13 of 27 classes score F1 0, including all healthy Tomato, Cherry, Peach, Soybean and Strawberry
  leaves and most Tomato diseases.
- Consequence: the app must warn users that it is trained on lab photos of single leaves and is not
  reliable on field photos. Fine-tuning on field images (e.g. PlantDoc train) is future work.

### Augmentation on field photos (10 Nov)

`python -m src.field_test --checkpoint outputs/shared_models/cnn_augment-<level>.pt` (4 Nov checkpoints, 25% subset, 10 epochs).

| Augmentation | PlantVillage test | PlantDoc field test | Field top-5 | Field macro F1 |
|---|---|---|---|---|
| none | 91.4% | 6.4% | 31.4% | 0.062 |
| strong | 80.6% | 10.2% | 39.4% | 0.093 |

Strong augmentation costs 11 points on lab photos but gains 4 points on field photos, which suggests it
reduces reliance on the PlantVillage setting. With 236 field images the gap is about 9 photos, so it is
suggestive, not conclusive. Worth trying strong augmentation with more epochs for the final model.
