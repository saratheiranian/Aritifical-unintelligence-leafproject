# Results log

Running record of experiments, for the report's data, evaluation and project-management
sections. Newest first. Runs below were tracked on app.clear.ml (project `PlantDisease`)
while the KCL ClearML server was down.

## 15 Nov 2026: Grad-CAM, where the models look (Maryam, MacBook, Apple GPU)

```bash
python -m src.gradcam --checkpoint outputs/final_model.pt outputs/cnn_augment-none.pt \
    outputs/cnn_augment-strong.pt --attention-sample 1000 --run-name maryam-gradcam
```

W&B run: `7hnz7t1s` (project Aritifical-unintelligence-leafproject). Grad-CAM (Selvaraju et al.,
2017) shows how much each region of a photo pushed the prediction up (bright yellow = most, dark =
hardly at all; colour-blind-friendly "inferno" map). The final model scanned all 8,318 test images
(99.0% correct, 83 mistakes, the same accuracy as `src.evaluate`). Figures (not committed) in
`outputs/gradcam/`: examples (correct predictions from 8 classes, fixed random choice, seed 42, and
the 4 most confident mistakes), mistakes (each confident mistake explained for the chosen and the
true class), models (4 random photos chosen without any model, explained by all three models) and
distortions (a diseased and a healthy leaf at the robustness levels). Tables:
`gradcam_mistakes.csv` (all 83 mistakes), `gradcam_confused_pairs.csv`, `gradcam_attention.csv`
(heat on the leaf per photo and model), `gradcam_attention_summary.csv` and
`gradcam_sanity_summary.csv`. The last feature maps
of the CNN are 8x8 for a 128x128 input, so the heatmaps show rough regions, not exact outlines.

**How sure the model is when it is wrong.** Only 9 of the 83 mistakes (11%) were at least 90%
confident, and 20 at least 80%. The median confidence is 66.4% on mistakes against 100.0% on
correct predictions, so low confidence is a useful warning sign, but not a complete one: some
mistakes are confidently wrong.

| Most confused pairs (true, predicted) | Mistakes |
| --- | --- |
| Corn: Cercospora leaf spot Gray leaf spot, read as Corn: Northern Leaf Blight | 10 |
| Tomato: Early blight, read as Tomato: Septoria leaf spot | 7 |
| Corn: Northern Leaf Blight, read as Corn: Cercospora leaf spot Gray leaf spot | 6 |
| Tomato: Target Spot, read as Tomato: Spider mites Two-spotted spider mite | 5 |
| Tomato: Early blight, read as Tomato: Late blight | 4 |

- **Mistakes are mostly between diseases of the same plant.** 59 of the 83 (71%) stay within one
  plant. The two corn leaf-spot diseases account for 16 mistakes (19%) in both directions, and
  Tomato Early blight for 17, mostly read as Septoria leaf spot or Late blight; both match the
  weakest classes in the test metrics.
- **8 diseased leaves were read as healthy**, the costliest kind of mistake for a grower, for
  example Tomato Late blight read as Tomato healthy at 88%.
- **The corn confusions look at the leaf.** On both confident corn mistakes the heatmaps for the
  chosen and for the true class lie on the leaf around its lesions, in partly different places, not
  on the background: the model looks at the lesions but cannot tell the two diseases apart.
- **One confident mistake comes from the background.** For Apple healthy read as Blueberry healthy
  (98%), the evidence for the chosen class lies on the background on either side of the leaf, while
  the evidence for the true class lies on the leaf.
- **On random photos all three models look at the lesions.** All three are correct on the 4 photos
  and put the heat on the Esca and Septoria spots; on the healthy strawberry all three spread some
  heat onto the background next to the leaf. Four photos cannot show a difference between the
  models, so it is measured over 1,000 photos below.
- **Distortions.** The diseased tomato leaf stays correct under blur 2, dark 0.5 and zoom 0.6, but
  at noise 25 it is read as Background without leaves (50%), with the heat spread over the
  background. The healthy strawberry fails under blur 2 (Orange Haunglongbing, 76%) and zoom 0.6
  (Corn healthy, 100%).

**Heat on the leaf, over 1,000 random photos.** For each photo a leaf mask separates the leaf from
the background and its shadow by colour (`gradcam_leaf_masks.png` shows 12 of them), and the share
of the Grad-CAM heat that falls on the leaf is measured for every model. The photos are drawn at
random from the test set, without the Background without leaves class; 7 more were skipped because
their mask covered less than 15% or more than 95% of the photo. 95% intervals come from 2,000
bootstrap resamples of the photos; the differences are paired (same photos, same masks).

| Model | Heat on the leaf | 95% interval | Difference from final_model (points) |
| --- | --- | --- | --- |
| final_model | 64.4% | 63.4 to 65.4% | |
| augment-strong | 61.4% | 60.1 to 62.8% | -3.0 (-3.9 to -2.1) |
| augment-none | 59.0% | 57.7 to 60.3% | -5.4 (-6.1 to -4.7) |
| final_model with random weights, 5 initialisations | 37.6 to 55.6% (mean 51.2%) | | -8.7 to -26.8 |

The leaf covers 49.1% of these photos on average. The trained-model numbers are reproducible: on a
300-photo trial run, Media got exactly the same heat on the leaf for all three models on a Windows
CPU as on the MacBook GPU.

- **All three trained models put more heat on the leaf than any random model.** The same network
  with random weights puts 54 to 56% of its heat on the leaf for four of the five initialisations
  (37.6% for the fifth), a little above the leaf's share of the photo (49.1%), probably because
  even random filters react more to the leaf's texture and contrast than to the plain background.
  The fair baseline is therefore the random model, not the leaf's area. The trained final model
  puts 64.4% on the leaf, 8.7 to 26.8 points more than the random models, and augment-none (59.0%)
  and augment-strong (61.4%) are also above all five.
- **About a third of the heat still falls outside the mask.** Part of it is the method: the 8x8
  heatmap is blurred when enlarged to 128x128, so heat on the leaf edge spills onto the background.
  Part is the mask (see the limits below), and part may be real use of the background, as in the
  Apple healthy mistake above. This measure cannot separate the three.
- **Without augmentation the model puts the least heat on the leaf** (5.4 points below the final
  model), and strong augmentation is between the two (3.0 points below). This fits the robustness
  test, where the model without augmentation depends most on lighting, but the gaps are small: all
  three models look mostly at the leaf, and each was trained only once. The final model also had
  more data and epochs, so only augment-none against augment-strong (2.4 points; no interval was
  computed for this pair) isolates the effect of augmentation.

**Sanity check: does the heatmap depend on what the model learned?** Some explanation methods
give nearly the same picture for a trained and an untrained model, because they mostly trace edges
in the photo (Adebayo et al., 2018). Here the final model's heatmaps were compared with those of the
same network with random weights, on the same 1,000 photos, for 5 random initialisations (one row
each in `gradcam_sanity_summary.csv`; `gradcam_sanity.png` shows 4 photos for the first). The
random weights are drawn on the CPU, so every machine gets the same random models.

| Random initialisation (seed) | 42 | 43 | 44 | 45 | 46 |
| --- | --- | --- | --- | --- | --- |
| Rank correlation with the trained heatmap | 0.31 | 0.27 | -0.21 | 0.22 | 0.23 |
| Heat on the leaf | 54.5% | 55.6% | 37.6% | 54.1% | 54.4% |

The rank correlation is 0.17 on average, from -0.21 to 0.31 (each with a 95% interval of about
+-0.02). **Grad-CAM only partly passes this check.** A correlation of 0.3 or less means most of the
heatmap's ordering changes when the weights are random, so the heatmaps do reflect what the model
learned. But for four of the five initialisations the two are weakly related (0.22 to 0.31): part
of the heatmap follows the photo itself, probably the leaf's texture and contrast, as the random
models' heat on the leaf also suggests. The heatmaps should therefore be read as a rough guide to
where the model looks, not as proof of what it uses. An earlier version of this check used a
single random initialisation drawn on the MacBook GPU and gave 0.05 ("almost unrelated"); in her
review Media got 0.32 for the same seed on a CPU. The cause was that the GPU draws other random
numbers than the CPU, and the gap showed that one draw is not enough, so the check now uses 5.

**Limits of the leaf mask.** The mask uses colour only. In a visual check of 12 masks from a trial
run on 300 photos, 8 were clean; a brightly lit patch of background and a yellowish shadow were
counted as leaf in two photos, and parts of a dark, glossy leaf and of a pale corn leaf were missed
in two others. The mask is built
to keep pale and dry lesions rather than drop them, since lesions are where the model should look.
All models are measured with the same masks, so the differences between models are less affected
than the absolute percentages.
  
## 12 Nov 2026: fine-tuning on field photos (PlantDoc train)

`python -m src.finetune_field --checkpoint outputs/final_model.pt --plantdoc-dir data/PlantDoc`:
10 epochs, lr 1e-4, strong augmentation, on 1,972 PlantDoc train photos (349 kept for validation,
15 dropped as look-alikes of test photos) mixed with 1,170 PlantVillage photos (30 per class).

| Metric | Before | After |
|---|---|---|
| Field accuracy (PlantDoc test, 236 photos) | 14.4% | 22.9% |
| Field accuracy, shared classes | 17.8% | 23.3% |
| Field top-5 accuracy | 40.7% | 59.8% |
| Field macro F1 | 0.128 | 0.216 |
| Lab accuracy (PlantVillage test) | 99.0% | 85.7% |

- A little real field data lifts field accuracy by 8.5 points and top-5 by 19 points.
- The price is 13 points of lab accuracy: one small CNN cannot serve both settings well.
- Field validation accuracy was still rising at epoch 10, so more epochs or more field data would help.
- The deployed model stays the 99% lab model; the app warns it is for lab-style photos.

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

## 10 Nov 2026: robustness test on distorted test images (Maryam, MacBook, Apple GPU)

```bash
python -m src.robustness --checkpoint outputs/final_model.pt outputs/cnn_augment-none.pt \
    outputs/cnn_augment-strong.pt --run-name maryam-robustness
```

All 8,318 test images, 7 distortions at 3 strengths each (22 levels including clean). Each batch
is distorted once and given to all three models, so they see identical images; the noise is seeded,
so the run is repeatable. Distortions are applied to the original image before the usual resize to
128x128, as they would be in a phone photo. Re-run on 15 Nov with Weights & Biases logging and the
no-leaf rate (W&B run `mye5k1mi`, project Aritifical-unintelligence-leafproject): every accuracy
was identical to the first run, so the test is fully repeatable. Figures (not committed):
`outputs/robustness/robustness_accuracy.png` and `outputs/robustness/distortions_preview.png`.

Models: `final_model` is the 99% model (all training data, 20 epochs, basic augmentation, task
`1bee769f0c32413e8b0ad57fdcc7d626`); `augment-none` and `augment-strong` are the 4 Nov subset models
(25% of the data, 10 epochs, tasks `967dd4f0b1ee494994490febdb1fc157` and
`0cc43dfc41094935a7dd024058f32ce1`). The final model differs in data and epochs too, so only none
vs strong isolates the effect of augmentation.

| Model | Clean | Mean over the 21 distorted levels | Drop (points) | Share of clean accuracy kept | Worst level |
| --- | --- | --- | --- | --- | --- |
| final_model | 99.0% | **73.2%** | 25.8 | 74% | noise 50 (6.0%) |
| augment-strong | 80.6% | 62.2% | 18.5 | 77% | noise 50 (13.0%) |
| augment-none | 91.4% | 54.4% | 37.1 | 59% | noise 50 (5.5%) |

The drop in points partly reflects where each model starts: augment-strong has the lowest clean
accuracy, so it has the least room to fall. The share of clean accuracy kept (mean on distorted
images divided by clean accuracy) corrects for that. Strong augmentation still keeps the most (77%,
against 74% for the final model and 59% without augmentation), but its lead over the final model is
only about 3 points from one training run each, so it is not a clear win. The clear gap is between
augmentation (strong or basic) and none.

Test accuracy at every level (random guessing is 2.6%). The best model in a row is in bold only
where it leads the runner-up by at least 5 points: each model was trained once, so smaller gaps may
come from the training run rather than the model (noise 25 has no clear winner). The table is
written by `src.robustness` to `outputs/robustness/robustness_table.md`. Strengths: blur radius in
pixels, noise standard deviation on the 0-255 scale, share of light kept (dark), brightness factor
(bright), degrees (rotation), JPEG quality out of 100, share of the width and height kept (zoom).

| Distortion | Strength | final_model | augment-none | augment-strong |
| --- | --- | --- | --- | --- |
| clean | - | **99.0%** | 91.4% | 80.6% |
| blur | 1 | **95.9%** | 76.1% | 82.8% |
| blur | 2 | 72.6% | 54.9% | **82.9%** |
| blur | 4 | 41.0% | 37.6% | **71.2%** |
| noise | 10 | **97.8%** | 89.0% | 77.1% |
| noise | 25 | 42.7% | 49.2% | 47.7% |
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
- **Noise is the main weakness of all three models.** At noise 25 they fall to 43-49% with no clear
  winner, at noise 50 to near random guessing. No model was trained with noise, so adding Gaussian
  noise to the training augmentation is the clearest next step.
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
- **Consistent with the field test (PlantDoc, further down):** strong augmentation also does better
  on real field photos (10.2% vs 6.4%). Field photos combine several of these changes at once, which
  may help explain the drop from 99% to 14%. Training the final model with strong augmentation plus
  noise, for more epochs, is the natural follow-up.
- A first run on a fixed sample of 500 test images gave the same picture: the mean over the
  distorted levels was within 2 points for every model, though single levels differed by up to
  4.1 points (augment-strong at bright 1.6), so single levels from a sample are less reliable.

**When a leaf is read as "no leaf".** The dataset has a class Background_without_leaves (photos
with no leaf). Reading a leaf photo as that class is a distinct kind of failure: the model does not
give a wrong disease, it says there is no leaf. The table shows the share of leaf photos read as
Background_without_leaves (from `outputs/robustness/robustness_no_leaf_table.md`, rounded to whole
percent). Clean photos and every level not listed stay at 1% or less for all three models.

| Distortion | Strength | final_model | augment-none | augment-strong |
| --- | --- | --- | --- | --- |
| blur | 4 | 3% | 1% | 0% |
| noise | 25 | 29% | 8% | 3% |
| noise | 50 | 62% | 62% | 30% |
| dark | 0.5 | 0% | 10% | 1% |
| dark | 0.3 | 11% | 19% | 11% |
| bright | 1.3 | 0% | 4% | 0% |
| bright | 1.6 | 0% | 16% | 0% |
| bright | 2 | 1% | 27% | 1% |
| jpeg | 5 | 13% | 6% | 11% |

- **Noise is the main cause.** At noise 25 the final model reads 29% of leaf
  photos as "no leaf", about half of its mistakes at that level, and at noise 50 the final model
  and augment-none do so for 62%. This matches the Grad-CAM results (15 Nov, from the Grad-CAM
  pull request): the noisy tomato leaf there is read as Background without leaves, with the heat
  spread over the background.
- **Strong augmentation halves it at noise 50** (30% against 62%) and keeps it at 3% at noise 25,
  although it was never trained with noise either.
- **Without augmentation, lighting changes also hide the leaf**: 27% at double brightness and 19%
  at 30% of the light, against 1% and 11% for the final model. This is the same lighting shortcut
  as in the accuracy table.
- **For the app this is the safer failure**: "no leaf found, please take a sharper photo" is better
  than a confident wrong disease. It is still a failure, since the leaf was there. At the mildest
  strength of every distortion the final model and augment-strong stay below 0.5%.

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

Re-run on 15 Nov with Weights & Biases logging (run `1hp9lxh9`, project
Aritifical-unintelligence-leafproject): same results as the 27 Oct run.

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
  elongated lesions. See the confusion matrix in W&B run `1hp9lxh9`.

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
