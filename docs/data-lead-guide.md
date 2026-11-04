# Role 1 (data lead): checklist

Everything the data lead owns, in order. Each step says which command to run and where its
output goes in the report. Tick items off in the matching GitHub Issue.

## 1. Run the existing data scripts (this week)

```bash
source .venv/bin/activate
python -m src.split_data          # class counts per split; check the max drift is ~0
python -m src.eda                 # outputs/eda/class_distribution.png, samples.png, split_check.png
python -m src.register_dataset    # ClearML Dataset "plant_village_split" with the split indices
```

- [ ] Copy the three EDA plots into the Overleaf project (`figures/`).
- [ ] Note the ClearML Dataset ID in `docs/results.md`.

## 2. Image quality checks (new, `src/data_quality.py`)

```bash
python -m src.data_quality --sample 5000 --no-clearml   # quick test, about a minute
python -m src.data_quality                              # all images, logged to ClearML
```

Outputs in `outputs/eda/`:

- `quality_sizes.csv`: image sizes
- `quality_brightness.png`, `quality_brightness_per_class.png/.csv`: lighting per class
- `quality_flagged_images.csv`: very dark, bright or low-contrast images
- `quality_duplicates.csv`: possible duplicates, and which split each copy is in

- [ ] Fill the quality table in `docs/data_card.md`.
- [ ] If duplicates span train and test, report how many. If there are many, discuss with the
      team whether to remove them from the split (that changes every result, so decide early).
- [ ] Open a few flagged images to see whether the thresholds make sense.

## 3. Balanced sampler (new, `--balanced-sampler`)

The dataset is 36x imbalanced. With `--balanced-sampler`, every class is drawn equally often
during training (validation and test are unchanged).

```bash
python -m src.train --subset 0.25 --epochs 10 --tags cnn subset baseline
python -m src.train --subset 0.25 --epochs 10 --balanced-sampler --tags cnn subset balanced
python -m src.evaluate --checkpoint outputs/best_model.pt      # after each run
```

- [ ] Compare macro F1 and the scores of the smallest classes (Potato healthy, Apple cedar rust)
      between the two runs. Run the evaluation straight after each training run, because both
      runs save to `outputs/best_model.pt`.
- [ ] Give the experiments lead the result. The variant is also in the sweep as `balanced-sampler`.

## 3b. Augmentation study (new, `--augment none|basic|strong`)

Class brightness ranges from 91 to 148 (0-255), so a model could learn lighting instead of
disease. `strong` adds random crops, vertical flips, larger rotations, stronger colour jitter
and occasional blur. Compare the three levels on the 25% subset, evaluating straight after each run:

```bash
python -m src.augment_preview          # outputs/eda/augmentation_levels.png, for report section 2
python -m src.train --subset 0.25 --epochs 10 --augment none   --tags cnn subset aug-none
python -m src.evaluate --checkpoint outputs/best_model.pt
python -m src.train --subset 0.25 --epochs 10 --augment strong --tags cnn subset aug-strong
python -m src.evaluate --checkpoint outputs/best_model.pt
```

The `basic` run is the baseline re-run you already have (89.4%, macro F1 0.867).

- [ ] Add test accuracy and macro F1 for all three levels to `docs/results.md`.
- [ ] Give the robustness lead (role 4) the strong-augmentation checkpoint: it should do better
      on the brightness and blur tests even if clean accuracy is similar.

## 4. Data card and report section 2 (by 15 Nov)

- [ ] Finish `docs/data_card.md` (fill every [bracket]).
- [ ] Write report section 2, about one page:
  - **Dataset overview:** source, size, classes, a sample-image figure.
  - **Exploratory analysis:** a class distribution figure, the imbalance, and the quality checks.
  - **Preprocessing and augmentation:** what is applied and why augmentation is used on training only.
  - **Split:** stratified 70/15/15 with seed 42, the sizes, the drift check, and the ClearML Dataset.
  - **Imbalance handling:** the balanced sampler result.

## 5. Later

- [ ] Section 6 Conclusions (27 Nov): collect one lesson learned from each teammate.
- [ ] Clean-clone test (1 Dec): on a fresh machine, follow only the README.
- [ ] Review your partner's (role 2) pull requests.
