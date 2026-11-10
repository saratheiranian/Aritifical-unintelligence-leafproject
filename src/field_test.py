"""Test the model on real field photos from PlantDoc instead of lab photos from PlantVillage.

PlantVillage photos are single leaves on a plain background, so a 99% test accuracy says little
about photos taken in a field. PlantDoc (Singh et al., CoDS-COMAD 2020, CC BY 4.0) has cropped
leaves photographed in real conditions for 27 classes that also exist in PlantVillage.

    git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/PlantDoc
    python -m src.field_test --checkpoint outputs/final_model.pt --plantdoc-dir data/PlantDoc/test

Reports two accuracies:
  all classes     the model may predict any of its 39 classes (what the app does);
  shared classes  the prediction is limited to classes that exist in PlantDoc, a fairer
                  comparison with the PlantVillage test accuracy on the same classes.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from src.evaluate import load_checkpoint, predict, top_k_accuracy
from src.models import get_device
from src.transforms import get_transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# PlantDoc folder name -> PlantVillage class name. Names are matched after normalize(), so
# differences in case, spaces, underscores and punctuation do not matter.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf": "Potato___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def normalize(name, drop_brackets=False):
    """Lower case, letters and digits only; optionally drop text in brackets first."""
    if drop_brackets:
        name = re.sub(r"\(.*?\)", "", name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_class(target, class_names):
    """Index of target in class_names, ignoring case and punctuation, then also text in brackets."""
    for drop in (False, True):
        hits = [i for i, c in enumerate(class_names) if normalize(c, drop) == normalize(target, drop)]
        if len(hits) == 1:
            return hits[0]
    return None


def resolve_mapping(folders, class_names, table=None):
    """Match PlantDoc folders to model class indices.

    Returns (mapping {folder: class index}, unmatched folders). A folder is unmatched when it is
    not in the table, or its PlantVillage name is not one of the model's classes.
    """
    table = PLANTDOC_TO_PLANTVILLAGE if table is None else table
    by_folder = {normalize(k): v for k, v in table.items()}
    mapping, unmatched = {}, []
    for folder in folders:
        target = by_folder.get(normalize(folder))
        index = find_class(target, class_names) if target else None
        if index is None:
            unmatched.append(folder)
        else:
            mapping[folder] = index
    return mapping, unmatched


def list_images(root, mapping):
    """[(image path, class index)] for every image in the mapped class folders, sorted."""
    items = []
    for folder, index in sorted(mapping.items()):
        for path in sorted((Path(root) / folder).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, index))
    return items


class ImageListDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as img:
            return self.transform(img.convert("RGB")), label


def restricted_predictions(probs, allowed):
    """Argmax over the allowed class indices only."""
    allowed = np.array(sorted(allowed))
    return allowed[probs[:, allowed].argmax(axis=1)]


def field_report(probs, labels, class_names):
    """Accuracies on the field photos and a per-class table (classes present in the photos)."""
    present = sorted(set(labels.tolist()))
    preds = probs.argmax(axis=1)
    shared_preds = restricted_predictions(probs, present)
    summary = {
        "images": len(labels),
        "classes": len(present),
        "accuracy_all_classes": float(np.mean(preds == labels)),
        "accuracy_shared_classes": float(np.mean(shared_preds == labels)),
        "top5_accuracy": top_k_accuracy(probs, labels, k=5),
    }
    report = classification_report(
        labels, preds, labels=present, target_names=[class_names[i] for i in present],
        output_dict=True, zero_division=0,
    )
    summary["macro_f1"] = float(report["macro avg"]["f1-score"])
    return summary, pd.DataFrame(report).transpose()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", default="outputs/final_model.pt")
    p.add_argument("--plantdoc-dir", default="data/PlantDoc/test")
    p.add_argument("--output-dir", default="outputs/field_test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--clearml", action="store_true", help="log the results to ClearML")
    args = p.parse_args(argv)

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    class_names = ckpt["class_names"]

    folders = sorted(d.name for d in Path(args.plantdoc_dir).iterdir() if d.is_dir())
    mapping, unmatched = resolve_mapping(folders, class_names)
    for folder, index in sorted(mapping.items()):
        print(f"  {folder:40s} -> {class_names[index]}")
    if unmatched:
        print(f"Not used (no matching class): {', '.join(unmatched)}")
    if not mapping:
        raise SystemExit("No PlantDoc folder matched a model class; check --plantdoc-dir.")

    items = list_images(args.plantdoc_dir, mapping)
    dataset = ImageListDataset(items, get_transforms(ckpt["img_size"], augment=False))
    probs, labels = predict(model, DataLoader(dataset, batch_size=args.batch_size), device)
    summary, report_df = field_report(probs, labels, class_names)

    print(f"\n{summary['images']} field photos, {summary['classes']} classes")
    print(f"Accuracy (all 39 classes)    {summary['accuracy_all_classes']:.4f}")
    print(f"Accuracy (shared classes)    {summary['accuracy_shared_classes']:.4f}")
    print(f"Top-5 accuracy               {summary['top5_accuracy']:.4f}")
    print(f"Macro F1                     {summary['macro_f1']:.4f}")
    print(report_df.round(3).to_string())

    os.makedirs(args.output_dir, exist_ok=True)
    report_df.to_csv(os.path.join(args.output_dir, "field_report.csv"))
    pd.Series(summary).to_csv(os.path.join(args.output_dir, "field_summary.csv"), header=False)

    if args.clearml:  # pragma: no cover - needs a ClearML server
        from clearml import Task

        from src.train import CLEARML_PROJECT

        task = Task.init(project_name=CLEARML_PROJECT, task_name="field-test", tags=["eval", "plantdoc"])
        logger = task.get_logger()
        for key, value in summary.items():
            logger.report_single_value(key, value)
        logger.report_table("Per-class metrics", "plantdoc", iteration=0, table_plot=report_df)
    return summary


if __name__ == "__main__":
    main()
