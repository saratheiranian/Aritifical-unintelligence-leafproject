# Data card: PlantVillage (39 classes)

Fill in the [brackets] once `python -m src.data_quality` has run. This card feeds report
section 2 and the Responsible AI paragraph.

## Source

- Hugging Face dataset: [`DScomp380/plant_village`](https://huggingface.co/datasets/DScomp380/plant_village),
  loaded with `datasets.load_dataset(..., split="train")` (the only split provided).
- Original collection: the PlantVillage project. Cite Hughes, D.P., Salathé, M.: An open access
  repository of images on plant health to enable the development of mobile disease diagnostics.
  arXiv:1511.08060 (2015).
- Licence: [check the Hugging Face dataset page and the original PlantVillage terms].
- We do not redistribute the images: the repository only stores code and split indices.

## Contents

- 55,447 RGB leaf photos, 39 classes: 38 crop/disease combinations across 14 crops, plus
  "Background without leaves".
- One label per image (`label` column, a ClassLabel).
- Image sizes: [from `outputs/eda/quality_sizes.csv`, e.g. "all 256x256"].
- Colour modes: [from the data-quality output].

## Class balance

- Strongly imbalanced: from 152 images (Potato healthy) to 5,507 (Orange Haunglongbing), about 36x.
- Other small classes: Apple cedar rust (275), Peach healthy (360), Raspberry healthy (371),
  Tomato mosaic virus (373).
- Mitigations available: the stratified split keeps every class in every split, macro F1 is reported
  next to accuracy, and `--balanced-sampler` trains with every class drawn equally often.

## Split

- Stratified 70/15/15 train/validation/test with seed 42: 38,812 / 8,317 / 8,318 images.
- Largest difference between a class's share in any split and in the full dataset: 0.0001.
- The exact row indices of each split are stored as the ClearML Dataset `plant_village_split`
  (`python -m src.register_dataset`).

## Quality checks (`python -m src.data_quality`)

| Check | Result |
| --- | --- |
| Duplicate groups (same perceptual hash) | [number] groups, [number] images |
| Duplicates spread across train/val/test (leakage) | [number] groups |
| Duplicates with conflicting labels | [number] groups |
| Very dark (< 40), very bright (> 215) or low-contrast (< 15) images | [number] |
| Class with the lowest / highest mean brightness | [class, value] / [class, value] |

## Preprocessing

- Resize to 128x128 (configurable with `--img-size`) and normalise with the ImageNet mean and std.
- Training only: random horizontal flip, rotation up to 15°, colour jitter. Validation and test
  images are only resized and normalised.

## Known limitations and biases

- Lab conditions: mostly single leaves on plain backgrounds, photographed in a controlled setting.
  Field photos (several leaves, soil, shadows, other plants) look different, so accuracy on them
  will be lower.
- Closed set: any photo is mapped to one of the 39 classes, even a plant that is not in the dataset
  (a maple leaf was classified as "Background without leaves" with 97% confidence).
- Imbalance: rare classes have fewer training examples, so their scores are less reliable.
- Geography and varieties: [who collected the images, and where; not every variety or region is covered].
- [Add anything the quality checks turn up, e.g. duplicates or classes with different lighting.]
