"""Upload a trained checkpoint to the Hugging Face Hub, with a model card.

    huggingface-cli login            # once, with a token that has write access
    python -m src.publish_model --repo-id <hf-user>/plant-disease-cnn --checkpoint outputs/best_model.pt
    python -m src.publish_model --repo-id <hf-user>/plant-disease-cnn --clearml-model-id <id> --filename cnn.pt

The app can then load every .pt file in that repo (set HF_MODEL_REPO on the Space).
"""

import argparse

import torch

from src.inference import pretty_name

CARD = """---
license: mit
library_name: pytorch
tags:
- image-classification
- plant-disease
datasets:
- DScomp380/plant_village
---

# Plant disease classifier ({arch})

Predicts one of {n} PlantVillage classes (plant + disease, or healthy) from a leaf photo.
Built for the KCL 5CCSAGAP small group project.

- Architecture: `{arch}` ({kwargs})
- Input: RGB image resized to {img}x{img}, normalised with ImageNet mean/std
- Best validation accuracy: {val}
- Trained on a stratified 70% split of [DScomp380/plant_village](https://huggingface.co/datasets/DScomp380/plant_village)

## Usage

```python
from huggingface_hub import hf_hub_download
from src.inference import Predictor   # from the project repository
predictor = Predictor(hf_hub_download("{repo}", "{file}"))
print(predictor.top_k(pil_image, k=5))
```

## Limitations

PlantVillage photos are single leaves on plain backgrounds taken in controlled conditions.
Accuracy on field photos (several leaves, cluttered backgrounds, other lighting) will be lower.
Only the {n} listed classes are known: any other plant or disease is forced into one of them.
This is a student project, not a diagnostic tool.

## Classes

{classes}
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--checkpoint", default="outputs/best_model.pt")
    p.add_argument("--clearml-model-id", default=None)
    p.add_argument("--filename", default=None, help="name in the repo (default: <arch>.pt)")
    p.add_argument("--private", action="store_true")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi

    checkpoint = args.checkpoint
    if args.clearml_model_id:
        from clearml import InputModel

        checkpoint = InputModel(model_id=args.clearml_model_id).get_local_copy()

    ckpt = torch.load(checkpoint, map_location="cpu")
    filename = args.filename or f"{ckpt['arch']}.pt"
    val = f"{ckpt['val_acc']:.4f}" if ckpt.get("val_acc") is not None else "n/a"
    card = CARD.format(
        arch=ckpt["arch"],
        kwargs=", ".join(f"{k}={v}" for k, v in ckpt["model_kwargs"].items()),
        n=len(ckpt["class_names"]),
        img=ckpt["img_size"],
        val=val,
        repo=args.repo_id,
        file=filename,
        classes="\n".join(f"- {pretty_name(c)}" for c in ckpt["class_names"]),
    )

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=checkpoint, path_in_repo=filename, repo_id=args.repo_id)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.repo_id)
    print(f"Uploaded {filename} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
