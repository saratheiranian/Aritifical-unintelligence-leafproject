"""Grad-CAM: which parts of a leaf photo the model used for its prediction.

Grad-CAM (Selvaraju et al., 2017) takes the feature maps of the last convolutional stage, weights
each map by how much it pushed the score of the explained class up (the average gradient), and adds
them up. Red areas in the heatmap mattered most for the prediction, blue areas hardly at all. It
shows whether the model looks at the disease itself or at shortcuts such as the background or the
lighting.

    python -m src.gradcam --checkpoint outputs/final_model.pt --no-wandb
    python -m src.gradcam --checkpoint outputs/final_model.pt outputs/cnn_augment-none.pt \
        outputs/cnn_augment-strong.pt

The first model is scanned over the test images to find correct predictions (one per class, a
fixed random choice) and its most confident mistakes. Outputs go to outputs/gradcam/:
gradcam_examples.png (photo and heatmap for each example), gradcam_mistakes.png (each confident
mistake explained for the predicted and for the true class), gradcam_models.png (random photos
chosen independently of any model, explained by every model, when there are two or more),
gradcam_distortions.png (a diseased and a healthy leaf under the blur, noise, dark and zoom levels
of the robustness test), gradcam_examples.csv, gradcam_mistakes.csv (every mistake with its
confidence) and gradcam_confused_pairs.csv (how often each true class was read as each wrong one).
Without --no-wandb the figures, tables and mistake statistics are also logged to Weights & Biases.

Heatmaps use the colour-blind-friendly "inferno" colour map: bright yellow = most important,
dark = hardly at all. Titles mark correct predictions with a tick and wrong ones with a cross, so
the meaning does not depend on colour alone.

The Grad-CAM functions only need torch, numpy, Pillow and matplotlib (no dataset library), so the
app can use them too.
"""

import argparse
import os
import textwrap
from itertools import zip_longest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.config import WANDB_ENTITY, WANDB_PROJECT
from src.distortions import apply_distortion
from src.inference import pretty_name
from src.models import get_device

OUTPUT_DIR = os.path.join("outputs", "gradcam")
SEED = 42  # same as src.data.SEED (not imported, so this module works without the dataset library)
DISPLAY_SIZE = 256  # PlantVillage photos are 256x256; models see them resized to img_size
COLORMAP = "inferno"  # perceptually uniform and readable with colour blindness (unlike "jet")
CONFIDENT = 0.9  # a mistake at or above this confidence counts as "confidently wrong"

# Robustness levels where the final model loses the most accuracy (see docs/results.md)
DISTORTION_LEVELS = [("clean", None), ("blur", 2), ("noise", 25), ("dark", 0.5), ("zoom", 0.6)]


# ---------------------------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------------------------


def target_layer(model):
    """The last convolutional stage, whose feature maps feed the classifier.

    PlantCNN and ResidualCNN: model.features. ResNet18: model.layer4. Anything else: its last
    Conv2d layer."""
    for name in ("features", "layer4"):
        if hasattr(model, name):
            return getattr(model, name)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("Grad-CAM needs a model with at least one convolutional layer")
    return convs[-1]


def grad_cam(model, inputs, class_idx=None, layer=None):
    """Grad-CAM heatmaps for a batch of preprocessed images.

    model: in eval mode (as load_checkpoint returns it), so dropout does not add noise.
    inputs: tensor (n, 3, H, W), already resized and normalised like the test images.
    class_idx: the class to explain, one for all images or one per image. Default: the predicted
        class of each image.
    Returns (cams, probs): cams is a numpy array (n, H, W) scaled to 0-1 per image (all zeros if
    nothing in the image speaks for the class); probs is (n, num_classes), the softmax output.

    The model is left untouched: the hook is removed afterwards and no parameter gradients are
    stored, so the function is safe to call on a model that is also used for predictions.
    """
    layer = layer or target_layer(model)
    stored = {}
    handle = layer.register_forward_hook(lambda module, args, output: stored.update(acts=output))
    try:
        with torch.enable_grad():
            # requires_grad on the input makes the gradient flow even through frozen layers
            inputs = inputs.detach().requires_grad_(True)
            logits = model(inputs)
            if class_idx is None:
                targets = logits.argmax(dim=1)
            else:
                targets = torch.as_tensor(class_idx, device=logits.device).reshape(-1)
                targets = targets.expand(len(inputs))
            score = logits.gather(1, targets.view(-1, 1)).sum()
            acts = stored["acts"]
            (grads,) = torch.autograd.grad(score, acts)
    finally:
        handle.remove()

    weights = grads.mean(dim=(2, 3), keepdim=True)  # importance of each feature map
    cams = F.relu((weights * acts).sum(dim=1, keepdim=True))  # only evidence *for* the class
    cams = F.interpolate(cams, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    peak = cams.flatten(1).max(dim=1).values.view(-1, 1, 1)
    cams = torch.where(peak > 0, cams / peak.clamp_min(1e-12), torch.zeros_like(cams))
    probs = torch.softmax(logits, dim=1)
    return cams.detach().cpu().numpy(), probs.detach().cpu().numpy()


def overlay(image, cam, alpha=0.5, size=(DISPLAY_SIZE, DISPLAY_SIZE)):
    """The photo with the heatmap on top: bright yellow = most important for the prediction,
    dark = least.

    The photo is resized to a square, as the model saw it."""
    photo = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR),
                       dtype=np.float32) / 255
    heat = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(size, Image.Resampling.BILINEAR)
    colours = matplotlib.colormaps[COLORMAP](np.asarray(heat) / 255)[..., :3]
    blended = (1 - alpha) * photo + alpha * colours
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))


def explain(model, transform, images, device="cpu", class_idx=None):
    """Grad-CAM for a list of PIL images. Returns (cams, predicted classes, their probabilities)."""
    inputs = torch.stack([transform(image.convert("RGB")) for image in images]).to(device)
    cams, probs = grad_cam(model, inputs, class_idx=class_idx)
    preds = probs.argmax(axis=1)
    return cams, preds, probs[np.arange(len(preds)), preds]


def explain_class(model, transform, images, classes, device="cpu"):
    """Grad-CAM for a chosen class per image (for example the true class of a mistake).

    Returns (cams, probability the model gave to that class)."""
    inputs = torch.stack([transform(image.convert("RGB")) for image in images]).to(device)
    cams, probs = grad_cam(model, inputs, class_idx=list(classes))
    return cams, probs[np.arange(len(classes)), list(classes)]


# ---------------------------------------------------------------------------------------------
# Choosing the test images to explain
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def predict(model, transform, test, image_col, indices, device="cpu", batch_size=64):
    """Softmax probabilities of one model for the given test positions, shape (n, num_classes)."""
    probs = []
    for start in range(0, len(indices), batch_size):
        batch = test[indices[start:start + batch_size]][image_col]
        inputs = torch.stack([transform(image.convert("RGB")) for image in batch]).to(device)
        probs.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
    return np.concatenate(probs)


def choose_examples(labels, preds, confidence, n_correct=8, n_wrong=4, seed=SEED):
    """Positions of the images to explain.

    correct: correct predictions from n_correct different classes (a fixed random choice).
    wrong: the n_wrong mistakes the model was most sure about, the most informative ones to see."""
    labels, preds, confidence = np.asarray(labels), np.asarray(preds), np.asarray(confidence)
    rng = np.random.default_rng(seed)
    right = np.flatnonzero(labels == preds)
    classes = rng.permutation(np.unique(labels[right]))[:n_correct]
    correct = [int(rng.choice(right[labels[right] == c])) for c in classes]
    mistakes = np.flatnonzero(labels != preds)
    mistakes = mistakes[np.argsort(-confidence[mistakes], kind="stable")][:n_wrong]
    return correct, [int(i) for i in mistakes]


def choose_random(labels, n=4, seed=SEED + 1):
    """Positions of n photos from n different classes, chosen without looking at any prediction.

    Used to compare models fairly: photos picked because one model got them right would favour
    that model."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels))[:n]
    return [int(rng.choice(np.flatnonzero(labels == c))) for c in classes]


def pick_for_distortions(names, n=2):
    """Order example positions so that diseased and healthy leaves alternate, diseased first.

    names: class name of each example. "Background_without_leaves" counts as neither."""
    healthy = [i for i, name in enumerate(names) if "healthy" in name.lower()]
    diseased = [i for i, name in enumerate(names)
                if "healthy" not in name.lower() and "background" not in name.lower()]
    order = [i for pair in zip_longest(diseased, healthy) for i in pair if i is not None]
    order += [i for i in range(len(names)) if i not in order]
    return order[:n]


def summarise_mistakes(labels, preds, confidence, class_names, indices):
    """Every mistake, how often each pair of classes is confused, and how sure the model was.

    Returns (mistakes, pairs, stats): mistakes has one row per wrong prediction (most confident
    first), pairs counts each (true, predicted) combination, and stats holds the number of mistakes,
    how many were at least CONFIDENT sure, and the median confidence of mistakes and of correct
    predictions."""
    labels, preds, confidence = np.asarray(labels), np.asarray(preds), np.asarray(confidence)
    wrong = np.flatnonzero(labels != preds)
    mistakes = pd.DataFrame({
        "test_index": [int(indices[i]) for i in wrong],
        "true": [class_names[int(labels[i])] for i in wrong],
        "predicted": [class_names[int(preds[i])] for i in wrong],
        "confidence": np.round(confidence[wrong], 4),
    }).sort_values("confidence", ascending=False, kind="stable").reset_index(drop=True)
    pairs = (mistakes.groupby(["true", "predicted"]).size().rename("count").reset_index()
             .sort_values(["count", "true", "predicted"], ascending=[False, True, True])
             .reset_index(drop=True))
    confident = int(np.sum(confidence[wrong] >= CONFIDENT))
    right = confidence[labels == preds]
    stats = {
        "mistakes": len(wrong),
        "confident_mistakes": confident,
        "confident_share": confident / len(wrong) if len(wrong) else 0.0,
        "median_confidence_mistakes": float(np.median(confidence[wrong])) if len(wrong) else None,
        "median_confidence_correct": float(np.median(right)) if len(right) else None,
    }
    return mistakes, pairs, stats


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------


def label(name, width=24):
    """Class name as short wrapped text for figure titles."""
    return "\n".join(textwrap.wrap(pretty_name(name), width))


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def mark(correct):
    """Tick or cross, so a title does not rely on colour alone."""
    return "\u2713" if correct else "\u2717"


def _hide_ticks(axes):
    for ax in np.ravel(axes):
        ax.set_xticks([])
        ax.set_yticks([])


def plot_examples(examples, class_names, path, model_name="", per_row=4):
    """Photo and heatmap side by side for each example; the prediction is green if correct.

    examples: dicts with image, cam, label, pred and confidence."""
    rows = int(np.ceil(len(examples) / per_row))
    fig, axes = plt.subplots(rows, 2 * per_row, figsize=(4 * per_row, 2.6 * rows), squeeze=False,
                             layout="constrained")
    _hide_ticks(axes)
    for ax in axes.flat[2 * len(examples):]:
        ax.axis("off")
    for i, ex in enumerate(examples):
        photo_ax, cam_ax = axes.flat[2 * i], axes.flat[2 * i + 1]
        photo_ax.imshow(ex["image"].resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        photo_ax.set_title(f"true: {label(class_names[ex['label']])}", fontsize=7)
        cam_ax.imshow(overlay(ex["image"], ex["cam"]))
        correct = ex["pred"] == ex["label"]
        cam_ax.set_title(f"{mark(correct)} pred: {label(class_names[ex['pred']])} "
                         f"({ex['confidence']:.0%})", fontsize=7,
                         color="green" if correct else "red")
    fig.suptitle(f"Grad-CAM of {model_name}: bright = most important for the prediction "
                 f"({mark(True)} correct, {mark(False)} wrong)")
    return _save(fig, path)


def plot_models(images, labels, per_model, class_names, path):
    """The same photos explained by every model: one row per photo, one column per model.

    per_model: {model name: (cams, preds, confidences)}."""
    names = list(per_model)
    fig, axes = plt.subplots(len(images), 1 + len(names),
                             figsize=(2.6 * (1 + len(names)), 2.7 * len(images)), squeeze=False,
                             layout="constrained")
    _hide_ticks(axes)
    for row, (image, true) in enumerate(zip(images, labels)):
        axes[row, 0].imshow(image.resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        axes[row, 0].set_title(f"true: {label(class_names[true])}", fontsize=7)
        for col, name in enumerate(names, start=1):
            cams, preds, confidence = per_model[name]
            axes[row, col].imshow(overlay(image, cams[row]))
            right = preds[row] == true
            axes[row, col].set_title(
                f"{name}\n{mark(right)} {label(class_names[preds[row]])} ({confidence[row]:.0%})",
                fontsize=7, color="green" if right else "red")
    fig.suptitle("Random photos (chosen without any model) explained by each model "
                 f"({mark(True)} correct, {mark(False)} wrong)")
    return _save(fig, path)


def plot_distortions(rows, levels, class_names, path, model_name=""):
    """How the heatmap and the prediction change on distorted versions of the same photo.

    rows: one list per photo with (distorted image, cam, pred, confidence, true) per level."""
    fig, axes = plt.subplots(len(rows), len(levels),
                             figsize=(2.6 * len(levels), 2.8 * len(rows)), squeeze=False,
                             layout="constrained")
    _hide_ticks(axes)
    for r, row in enumerate(rows):
        for c, ((name, strength), (image, cam, pred, confidence, true)) in enumerate(zip(levels, row)):
            level = name if strength is None else f"{name} {strength:g}"
            axes[r, c].imshow(overlay(image, cam))
            axes[r, c].set_title(
                f"{level}\n{mark(pred == true)} {label(class_names[pred])} ({confidence:.0%})",
                fontsize=7, color="green" if pred == true else "red")
        axes[r, 0].set_ylabel(f"true:\n{label(class_names[row[0][4]], 18)}", fontsize=7)
    fig.suptitle(f"Grad-CAM of {model_name} on distorted photos (robustness test levels)")
    return _save(fig, path)


def plot_mistakes(rows, class_names, path, model_name=""):
    """Each mistake explained twice: for the wrong class the model chose and for the true class.

    rows: dicts with image, label, pred, cam_pred, conf_pred, cam_true and prob_true."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(8, 2.8 * len(rows)), squeeze=False,
                             layout="constrained")
    _hide_ticks(axes)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image"].resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        axes[r, 0].set_title(f"true: {label(class_names[row['label']])}", fontsize=7)
        axes[r, 1].imshow(overlay(row["image"], row["cam_pred"]))
        axes[r, 1].set_title(f"{mark(False)} why {label(class_names[row['pred']])}\n"
                             f"(chosen, {row['conf_pred']:.0%})", fontsize=7, color="red")
        axes[r, 2].imshow(overlay(row["image"], row["cam_true"]))
        axes[r, 2].set_title(f"why {label(class_names[row['label']])}\n"
                             f"(true class, {row['prob_true']:.0%})", fontsize=7)
    fig.suptitle(f"Most confident mistakes of {model_name}: evidence for the chosen and the true class")
    return _save(fig, path)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", default=["outputs/final_model.pt"],
                   help="one or more model files; the first one picks the examples")
    p.add_argument("--sample", type=int, default=None,
                   help="scan a fixed random sample of this many test images (default: all 8,318)")
    p.add_argument("--n-correct", type=int, default=8, help="correct examples, one per class")
    p.add_argument("--n-wrong", type=int, default=4, help="most confident mistakes to show")
    p.add_argument("--n-distorted", type=int, default=2, help="photos in the distortion figure")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-name", default="gradcam", help="name of the WandB run")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--no-wandb", action="store_true", help="do not log to Weights & Biases")
    args = p.parse_args(argv)

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=args.run_name,
                         job_type="eval", tags=["eval", "gradcam"], config=vars(args))
        print(f"WandB run ID: {run.id}")

    from src import robustness  # needs the dataset library, so imported only here

    test, image_col, label_col, class_names = robustness.load_test_split()
    device = get_device()
    models = robustness.load_models(args.checkpoint, device)
    first = next(iter(models))
    model, transform = models[first]

    # 1. Scan the test images with the first model: every mistake, and the examples to show
    indices = robustness.sample_indices(len(test), args.sample)
    probs = predict(model, transform, test, image_col, indices, device, args.batch_size)
    labels = np.asarray(test[label_col])[indices]
    preds, confidence = probs.argmax(axis=1), probs.max(axis=1)
    mistakes, pairs, stats = summarise_mistakes(labels, preds, confidence, class_names, indices)
    os.makedirs(args.output_dir, exist_ok=True)
    mistakes.to_csv(os.path.join(args.output_dir, "gradcam_mistakes.csv"), index=False)
    pairs.to_csv(os.path.join(args.output_dir, "gradcam_confused_pairs.csv"), index=False)
    print(f"{first}: {np.mean(preds == labels):.1%} correct on {len(indices)} test images, "
          f"{stats['mistakes']} mistakes, {stats['confident_mistakes']} of them at least "
          f"{CONFIDENT:.0%} confident")
    medians = [(stats[key], what) for key, what in [("median_confidence_mistakes", "mistakes"),
                                                     ("median_confidence_correct", "correct")]]
    print("Median confidence: " + ", ".join(f"{value:.1%} on {what}" for value, what in medians
                                            if value is not None))
    if stats["mistakes"]:
        print("Most confused pairs:\n" + pairs.head(5).to_string(index=False))
    correct, wrong = choose_examples(labels, preds, confidence, args.n_correct, args.n_wrong)
    chosen = [indices[i] for i in correct + wrong]
    images = [test[int(i)][image_col].convert("RGB") for i in chosen]
    true = [int(labels[i]) for i in correct + wrong]

    # 2. Photo and heatmap for every example
    cams, ex_preds, ex_conf = explain(model, transform, images, device)
    examples = [{"image": im, "cam": cam, "label": t, "pred": int(pr), "confidence": float(cf)}
                for im, cam, t, pr, cf in zip(images, cams, true, ex_preds, ex_conf)]
    table = pd.DataFrame({
        "test_index": [int(i) for i in chosen],
        "kind": ["correct"] * len(correct) + ["mistake"] * len(wrong),
        "true": [class_names[t] for t in true],
        "predicted": [class_names[int(p_)] for p_ in ex_preds],
        "confidence": np.round(ex_conf, 4),
    })
    table.to_csv(os.path.join(args.output_dir, "gradcam_examples.csv"), index=False)
    figures = {"examples": plot_examples(
        examples, class_names, os.path.join(args.output_dir, "gradcam_examples.png"), first)}

    # 3. Each confident mistake explained for the chosen (wrong) class and for the true class
    if wrong:
        w = list(range(len(correct), len(chosen)))
        true_cams, true_probs = explain_class(model, transform, [images[i] for i in w],
                                              [true[i] for i in w], device)
        rows = [{"image": images[i], "label": true[i], "pred": int(ex_preds[i]),
                 "cam_pred": cams[i], "conf_pred": float(ex_conf[i]),
                 "cam_true": tc, "prob_true": float(tp)}
                for i, tc, tp in zip(w, true_cams, true_probs)]
        figures["mistakes"] = plot_mistakes(
            rows, class_names, os.path.join(args.output_dir, "gradcam_mistakes.png"), first)

    # 4. Random photos, chosen without looking at any model, explained by every model
    if len(models) > 1:
        positions = choose_random(labels, n=4)
        photos = [test[int(indices[k])][image_col].convert("RGB") for k in positions]
        per_model = {name: explain(m, t, photos, device) for name, (m, t) in models.items()}
        figures["models"] = plot_models(photos, [int(labels[k]) for k in positions], per_model,
                                        class_names,
                                        os.path.join(args.output_dir, "gradcam_models.png"))

    # 5. A diseased and a healthy leaf under the robustness distortions (same seeded noise)
    rows = []
    focus = list(range(len(correct))) or list(range(len(chosen)))
    for k in pick_for_distortions([class_names[true[i]] for i in focus], args.n_distorted):
        i = focus[k]
        distorted = [apply_distortion(images[i], name, strength, seed=SEED + int(chosen[i]))
                     for name, strength in DISTORTION_LEVELS]
        d_cams, d_preds, d_conf = explain(model, transform, distorted, device)
        rows.append(list(zip(distorted, d_cams, d_preds, d_conf, [true[i]] * len(distorted))))
    figures["distortions"] = plot_distortions(
        rows, DISTORTION_LEVELS, class_names,
        os.path.join(args.output_dir, "gradcam_distortions.png"), first)

    print(table.to_string(index=False))
    print(f"Saved {len(figures)} figure(s) and 3 tables to {args.output_dir}")

    if run:
        run.log({f"gradcam/{name}": wandb.Image(path) for name, path in figures.items()}
                | {"gradcam/examples_table": wandb.Table(dataframe=table),
                   "gradcam/mistakes_table": wandb.Table(dataframe=mistakes),
                   "gradcam/confused_pairs": wandb.Table(dataframe=pairs)})
        run.summary["gradcam/scanned_images"] = len(indices)
        for key, value in stats.items():
            if value is not None:
                run.summary[f"gradcam/{key}"] = value
        print(f"Logged to WandB run {run.id}: {run.url}")
        run.finish()
    return table


if __name__ == "__main__":
    main()
