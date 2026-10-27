"""Gradio app: upload leaf photos and get the most likely diseases.

Features: top-5 predictions (bars + table), example gallery, model selector, low-confidence
warning, training examples of the predicted class, flagging wrong predictions, batch upload.

Run locally from the repo root:
    python app/app.py --checkpoint outputs/best_model.pt
    python app/app.py --checkpoint outputs/cnn.pt outputs/resnet18.pt     # model selector

Where models come from (first match wins):
    --checkpoint paths, else every .pt in ./models next to app.py, else every .pt in the
    Hugging Face model repo named by the HF_MODEL_REPO environment variable.
"""

import argparse
import csv
import os
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

# Works both from the repo (app/app.py) and from a Space where src/ sits next to app.py.
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from src.inference import Predictor, pretty_name  # noqa: E402

TOP_K = 5
LOW_CONFIDENCE = 0.5
EXAMPLES_DIR = HERE / "examples"
CLASS_EXAMPLES_DIR = HERE / "class_examples"
FLAG_DIR = HERE / "flagged"


def find_checkpoints(cli_paths):
    """Return {display name: local path} for every available model."""
    if cli_paths:
        return {Path(p).stem: p for p in cli_paths}
    local = sorted((HERE / "models").glob("*.pt"))
    if local:
        return {p.stem: str(p) for p in local}
    repo = os.environ.get("HF_MODEL_REPO")
    if repo:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = [f for f in list_repo_files(repo) if f.endswith(".pt")]
        return {Path(f).stem: hf_hub_download(repo, f) for f in files}
    raise SystemExit("No model found: pass --checkpoint, add .pt files to models/, or set HF_MODEL_REPO")


def build_app(checkpoints):
    @lru_cache(maxsize=None)
    def get_predictor(name):
        return Predictor(checkpoints[name])

    model_names = list(checkpoints)
    class_names = get_predictor(model_names[0]).class_names

    def describe(name):
        p = get_predictor(name)
        acc = f"{p.val_acc:.1%}" if p.val_acc is not None else "unknown"
        return f"**{name}**: `{p.arch}`, input {p.img_size}x{p.img_size}, validation accuracy {acc}"

    def class_gallery(class_name):
        folder = CLASS_EXAMPLES_DIR / class_name
        return sorted(str(p) for p in folder.glob("*.jpg")) if folder.is_dir() else []

    def classify(image, model_name):
        if image is None:
            return {}, pd.DataFrame(columns=["Disease", "Probability"]), "Upload a leaf photo first.", [], None
        top = get_predictor(model_name).top_k(image, k=TOP_K)
        label = {pretty_name(c): p for c, p in top}
        table = pd.DataFrame(
            [(pretty_name(c), f"{p:.1%}") for c, p in top], columns=["Disease", "Probability"]
        )
        best_class, best_p = top[0]
        if best_p < LOW_CONFIDENCE:
            note = (
                f"⚠️ Low confidence ({best_p:.0%}). The photo may be unclear, show several leaves, "
                "or show a plant or disease the model was not trained on."
            )
        else:
            note = f"Most likely: **{pretty_name(best_class)}** ({best_p:.0%})."
        state = {"model": model_name, "predicted": best_class, "probability": best_p}
        return label, table, note, class_gallery(best_class), state

    def flag(image, state, correct_label):
        if image is None or not state:
            return "Classify an image before flagging it."
        FLAG_DIR.mkdir(exist_ok=True)
        image_path = FLAG_DIR / f"{uuid.uuid4().hex}.png"
        image.save(image_path)
        log = FLAG_DIR / "flags.csv"
        new_file = not log.exists()
        with open(log, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["time", "image", "model", "predicted", "probability", "user_label"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"), image_path.name,
                state["model"], state["predicted"], f"{state['probability']:.4f}", correct_label or "",
            ])
        return "Thanks, this prediction was flagged for review."

    def classify_batch(files, model_name):
        if not files:
            return pd.DataFrame()
        rows = []
        for f in files:
            path = f if isinstance(f, str) else f.name
            top = get_predictor(model_name).top_k(Image.open(path), k=3)
            row = {"File": Path(path).name}
            for rank, (c, p) in enumerate(top, start=1):
                row[f"#{rank}"] = pretty_name(c)
                row[f"#{rank} prob"] = f"{p:.1%}"
            row["Low confidence"] = "yes" if top[0][1] < LOW_CONFIDENCE else ""
            rows.append(row)
        return pd.DataFrame(rows)

    with gr.Blocks(title="Plant disease classifier") as demo:
        gr.Markdown(
            "# 🌿 Plant disease classifier\n"
            f"Upload a photo of a single leaf to see the {TOP_K} most likely diseases "
            f"out of {len(class_names)} PlantVillage classes. "
            "This is a student project trained on lab photos, **not a diagnosis**."
        )
        model = gr.Dropdown(model_names, value=model_names[0], label="Model",
                            visible=len(model_names) > 1)
        model_info = gr.Markdown(describe(model_names[0]))
        model.change(describe, model, model_info)

        with gr.Tab("Single image"):
            state = gr.State()
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Leaf photo", height=320)
                    button = gr.Button("Classify", variant="primary")
                    examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg"))
                    if examples:
                        gr.Examples(examples=examples, inputs=image, label="Or pick an example")
                with gr.Column():
                    note = gr.Markdown()
                    label = gr.Label(num_top_classes=TOP_K, label=f"Top {TOP_K} predictions")
                    table = gr.Dataframe(headers=["Disease", "Probability"], label="Probabilities")
            similar = gr.Gallery(label="Training images of the predicted class (does yours look similar?)",
                                 columns=6, height=200)
            with gr.Accordion("Is this prediction wrong?", open=False):
                correct = gr.Dropdown([pretty_name(c) for c in class_names],
                                      label="Correct disease, if you know it")
                flag_button = gr.Button("Flag this prediction")
                flag_status = gr.Markdown()
            outputs = [label, table, note, similar, state]
            button.click(classify, [image, model], outputs)
            image.upload(classify, [image, model], outputs)
            flag_button.click(flag, [image, state, correct], flag_status)

        with gr.Tab("Batch"):
            files = gr.File(file_count="multiple", file_types=["image"], label="Leaf photos")
            batch_button = gr.Button("Classify all", variant="primary")
            batch_table = gr.Dataframe(label="Top 3 predictions per image")
            batch_button.click(classify_batch, [files, model], batch_table)

        with gr.Tab("About"):
            gr.Markdown(
                "**How it works.** A convolutional neural network trained from scratch in PyTorch on "
                "the PlantVillage dataset (55,447 images, 39 classes, stratified 70/15/15 split). "
                "Images are resized and normalised exactly as in training.\n\n"
                "**Limitations.** Training photos show single leaves on plain backgrounds. Photos taken "
                "outdoors, with several leaves or other plants, will be less reliable, and plants or "
                "diseases outside the list are always mapped to one of the known classes. Use the "
                "low-confidence warning and the example images to judge a prediction, and ask an expert "
                "before acting on it.\n\n"
                "**Privacy.** Uploaded images are not stored unless you flag a prediction.\n\n"
                "**Classes:** " + ", ".join(pretty_name(c) for c in class_names)
            )
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", nargs="*", default=None)
    args, _ = p.parse_known_args()
    build_app(find_checkpoints(args.checkpoint)).launch()


if __name__ == "__main__":
    main()
