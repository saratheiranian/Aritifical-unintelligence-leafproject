"""Gradio app: upload a leaf photo, get the top-5 predicted diseases.

Run locally from the repo root:
    python app/app.py --checkpoint outputs/best_model.pt

For the Hugging Face Space, copy this file to the Space root as app.py, together with the
src/ folder, app/requirements.txt (as requirements.txt) and the checkpoint as model.pt.
"""

import argparse
import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

# Works both from the repo (app/app.py) and from a Space where src/ sits next to app.py.
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from src.models import build_model  # noqa: E402
from src.transforms import get_transforms  # noqa: E402

TOP_K = 5


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["arch"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["class_names"], get_transforms(ckpt["img_size"], augment=False)


def pretty(name):
    return name.replace("___", ": ").replace("_", " ")


def make_predict(model, class_names, transform):
    @torch.no_grad()
    def predict(image):
        if image is None:
            return {}, pd.DataFrame(columns=["Disease", "Probability"])
        x = transform(image.convert("RGB")).unsqueeze(0)  # resizes to the model's input size
        probs = torch.softmax(model(x), dim=1)[0]
        top = torch.topk(probs, TOP_K)
        rows = [(pretty(class_names[i]), round(p.item(), 4)) for p, i in zip(top.values, top.indices)]
        return dict(rows), pd.DataFrame(rows, columns=["Disease", "Probability"])

    return predict


def build_ui(predict):
    with gr.Blocks(title="Plant disease classifier") as demo:
        gr.Markdown(
            "# Plant disease classifier\n"
            "Upload a photo of a single leaf to see the 5 most likely diseases. "
            "This is a student project trained on lab photos (PlantVillage), not a diagnosis."
        )
        with gr.Row():
            image = gr.Image(type="pil", label="Leaf photo")
            with gr.Column():
                label = gr.Label(num_top_classes=TOP_K, label="Top 5 predictions")
                table = gr.Dataframe(headers=["Disease", "Probability"], label="Probabilities")
        button = gr.Button("Classify", variant="primary")
        button.click(predict, inputs=image, outputs=[label, table])
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.environ.get("MODEL_PATH", "model.pt"))
    args, _ = p.parse_known_args()
    model, class_names, transform = load_model(args.checkpoint)
    build_ui(make_predict(model, class_names, transform)).launch()


if __name__ == "__main__":
    main()
