r"""Grad-CAM: which parts of a leaf photo the model used for its prediction.

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
Two measurements turn the examples into numbers (--attention-sample photos, default 300):
- Heat on the leaf: for random test photos, the share of each model's heatmap that falls on the
  leaf rather than the background. The leaf is found by colour (plant hues, not the grey or bluish
  background and shadows); photos whose mask looks implausible (under 15% or over 95% leaf) are
  skipped and counted. All models are measured on the same photos with the same masks, so the
  comparison between them stays fair even where a mask is imperfect. gradcam_leaf_masks.png shows
  the masks for a visual check.
- Sanity check (Adebayo et al., 2018): the first model's heatmaps are compared with those of the
  same model with random weights, for several random initialisations (--random-seeds, default 5)
  because one draw can be unusual. If Grad-CAM explains what the model learned, the two should
  differ (low rank correlation); gradcam_sanity.png shows examples for the first seed.
Results: gradcam_attention.csv (every photo and model), gradcam_attention_summary.csv and
gradcam_sanity_summary.csv (one row per random initialisation).

Without --no-wandb the figures, tables and statistics are also logged to Weights & Biases.

Heatmaps use the colour-blind-friendly "inferno" colour map: bright yellow = most important,
dark = hardly at all. Titles mark correct predictions with a tick and wrong ones with a cross, so
the meaning does not depend on colour alone.

The Grad-CAM functions only need torch, numpy, Pillow and matplotlib (no dataset library), so the
app can use them too.
"""

import argparse
import copy
import os
import textwrap
import warnings
from itertools import zip_longest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.stats import spearmanr
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
NO_LEAF_CLASS = "Background_without_leaves"  # photos without a leaf: no leaf mask possible
LEAF_AREA_RANGE = (0.15, 0.95)  # masks outside this share of the photo are treated as failures

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
# Measurements: heat on the leaf, and the sanity check
# ---------------------------------------------------------------------------------------------


def leaf_mask(image, size=(128, 128)):
    """True where the photo shows the leaf, False for the background and shadows.

    PlantVillage backgrounds and shadows are grey, bluish or pinkish, while leaves (healthy or
    diseased) have green to yellow hues or strong colours. A pixel counts as leaf if its hue is
    between 40 and 200 degrees with some saturation, or if it is strongly coloured (brown or
    yellow lesions), unless it has the background colour read from the photo's border or is a
    darker shade of it (a shadow). Small gaps are closed, holes inside the leaf filled, and specks
    removed. Known limits: parts of very dark or glossy leaves can be missed, and brightly lit
    background or a shadow with a yellowish tint can count as leaf. This errs towards keeping
    pale lesions next to the background, which a stricter colour match removes.
    size is (width, height), the size of the heatmap the mask is compared with."""
    small = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    rgb = np.asarray(small.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32) / 255
    hsv = rgb_to_hsv(rgb)
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    plant_hue = (hue >= 40) & (hue <= 200) & (sat > 0.10)
    mask = (plant_hue | (sat > 0.30)) & (val > 0.10)
    # Coloured backgrounds (for example pink-brown paper) can pass the colour test. When most of
    # the photo's border is not plant-coloured, the border shows the background: leave out pixels
    # of that colour. Close-ups that fill the frame (corn) have a plant-coloured border and skip it.
    edge = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3),
                           rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
    edge_plant = np.concatenate([plant_hue[:4].ravel(), plant_hue[-4:].ravel(),
                                 plant_hue[:, :4].ravel(), plant_hue[:, -4:].ravel()])
    if (~edge_plant).mean() >= 0.5:
        background = np.median(edge[~edge_plant], axis=0)
        if background.max() - background.min() > 0.2 * background.max():
            # Only for a coloured background: on grey ones this would also cut pale leaf parts.
            mask &= np.linalg.norm(rgb - background, axis=-1) > 0.12
        # A shadow is a darker copy of the background: same colour balance, lower brightness.
        tint = rgb / np.maximum(rgb.sum(axis=-1, keepdims=True), 1e-6)
        bg_tint = background / max(background.sum(), 1e-6)
        shadow = (np.linalg.norm(tint - bg_tint, axis=-1) < 0.03) & (val < 0.85 * background.max())
        mask &= ~shadow
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=2)
    parts, n = ndimage.label(mask)
    if n > 1:  # keep the leaf (and large pieces of it), drop small specks
        sizes = ndimage.sum(mask, parts, range(1, n + 1))
        mask = np.isin(parts, 1 + np.flatnonzero(sizes >= 0.2 * sizes.max()))
    return mask


def heat_on_leaf(cam, mask):
    """Share of the heatmap that falls on the leaf (NaN if the heatmap is empty).

    If the heat were spread evenly, this would equal the leaf's share of the photo."""
    total = float(cam.sum())
    return float(cam[mask].sum()) / total if total > 0 else np.nan


def randomised_copy(model, seed=SEED):
    """The same architecture with fresh random weights, for the sanity check. The original model
    is not changed. The weights are drawn on the CPU and then moved to the model's device, so a
    seed gives the same weights on a Mac GPU, an NVIDIA GPU or a CPU."""
    device = next(model.parameters()).device
    random_model = copy.deepcopy(model).cpu()
    torch.manual_seed(seed)
    for module in random_model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        if hasattr(module, "reset_running_stats"):
            module.reset_running_stats()
    return random_model.to(device).eval()


def bootstrap_ci(values, n_boot=2000, seed=SEED):
    """Mean and 95% bootstrap confidence interval (NaN values are left out)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_correlation(a, b, step=4):
    """Spearman rank correlation between two heatmaps (every step-th pixel, for speed); NaN if
    either is constant."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a[::step, ::step].ravel(), b[::step, ::step].ravel())[0])


def choose_attention_photos(test, image_col, labels, positions, n, size, no_leaf_class=None,
                            seed=SEED + 2):
    """Random photos (positions into the scanned images) with a plausible leaf mask.

    Returns (chosen positions, their images, their masks, number skipped because the mask looked
    wrong). Photos of the no-leaf class are never chosen."""
    rng = np.random.default_rng(seed)
    chosen, images, masks, skipped = [], [], [], 0
    for k in rng.permutation(len(positions)):
        if len(chosen) == n:
            break
        if no_leaf_class is not None and labels[k] == no_leaf_class:
            continue
        image = test[int(positions[k])][image_col].convert("RGB")
        mask = leaf_mask(image, size)
        if not LEAF_AREA_RANGE[0] <= mask.mean() <= LEAF_AREA_RANGE[1]:
            skipped += 1
            continue
        chosen.append(int(k))
        images.append(image)
        masks.append(mask)
    return chosen, images, masks, skipped


def random_name(first, seed):
    """Name of the first model with the random weights of one seed."""
    return f"{first} (random weights, seed {seed})"


def measure_attention(models, images, masks, device="cpu", batch_size=64, seeds=(SEED,)):
    """Heat on the leaf for every photo and model, plus the sanity check for the first model.

    Returns (per_photo, example_cams): per_photo has one row per photo and model with the leaf
    area and the heat on the leaf. The first model also appears once per seed with random
    weights; those rows hold the rank correlation between its trained and random heatmaps.
    One random draw can be unusual, so several seeds show how much the check varies.
    example_cams holds the first model's trained heatmaps and those of the first seed for the
    first 4 photos, for the figure."""
    first = next(iter(models))
    to_measure = dict(models)
    for seed in seeds:
        to_measure[random_name(first, seed)] = (randomised_copy(models[first][0], seed),
                                                models[first][1])
    cams = {name: [] for name in to_measure}
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        for name, (model, transform) in to_measure.items():
            cams[name].extend(explain(model, transform, batch, device)[0])
    rows = []
    for name in to_measure:
        for i, (cam, mask) in enumerate(zip(cams[name], masks)):
            if cam.shape != mask.shape:  # a model with another image size
                mask = np.asarray(Image.fromarray(mask).resize(cam.shape[::-1], Image.Resampling.NEAREST))
            rows.append({"photo": i, "model": name, "leaf_area": float(mask.mean()),
                         "heat_on_leaf": heat_on_leaf(cam, mask)})
    per_photo = pd.DataFrame(rows)
    per_photo["rank_corr_with_trained"] = np.nan
    for seed in seeds:
        name = random_name(first, seed)
        per_photo.loc[per_photo["model"] == name, "rank_corr_with_trained"] = [
            rank_correlation(a, b) for a, b in zip(cams[first], cams[name])]
    examples = {"trained": cams[first][:4], "random": cams[random_name(first, seeds[0])][:4]}
    return per_photo, examples


def summarise_sanity(per_photo):
    """One row per random seed: the mean rank correlation between the trained and the random
    heatmaps with its 95% interval, and the heat on the leaf with random weights."""
    rows = []
    for name, part in per_photo[per_photo["rank_corr_with_trained"].notna()].groupby(
            "model", sort=False):
        mean, low, high = bootstrap_ci(part["rank_corr_with_trained"])
        rows.append({"model": name, "rank_corr": mean, "ci_low": low, "ci_high": high,
                     "heat_on_leaf": float(part["heat_on_leaf"].mean())})
    return pd.DataFrame(rows)


def summarise_attention(per_photo, first):
    """One row per model: mean leaf area, mean heat on the leaf with its 95% interval, and the
    paired difference from the first model (same photos) with its 95% interval."""
    base = per_photo[per_photo["model"] == first].set_index("photo")["heat_on_leaf"]
    rows = []
    for name, part in per_photo.groupby("model", sort=False):
        mean, low, high = bootstrap_ci(part["heat_on_leaf"])
        row = {"model": name, "photos": int(part["heat_on_leaf"].notna().sum()),
               "leaf_area": float(part["leaf_area"].mean()),
               "heat_on_leaf": mean, "ci_low": low, "ci_high": high}
        if name != first:
            diff = part.set_index("photo")["heat_on_leaf"] - base
            row["diff_vs_first"], row["diff_ci_low"], row["diff_ci_high"] = bootstrap_ci(diff)
        rows.append(row)
    return pd.DataFrame(rows)


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


def plot_leaf_masks(images, masks, path, n=12):
    """The leaf masks of the first measured photos, for a visual check: the parts counted as
    background are tinted purple."""
    n = min(n, len(images))
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows), squeeze=False,
                             layout="constrained")
    _hide_ticks(axes)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, image, mask in zip(axes.flat, images, masks):
        photo = np.asarray(image.resize(mask.shape[::-1]), dtype=np.float32) / 255
        photo[~mask] = photo[~mask] * 0.3 + np.array([0.6, 0.1, 0.6]) * 0.7
        ax.imshow(photo)
        ax.set_title(f"leaf {mask.mean():.0%}", fontsize=8)
    fig.suptitle("Leaf masks used to measure heat on the leaf (purple = counted as background)")
    return _save(fig, path)


def plot_sanity(images, trained, random, path, model_name=""):
    """Heatmaps of the trained model next to the same model with random weights."""
    fig, axes = plt.subplots(len(images), 3, figsize=(7.5, 2.6 * len(images)), squeeze=False,
                             layout="constrained")
    _hide_ticks(axes)
    for r, (image, cam_t, cam_r) in enumerate(zip(images, trained, random)):
        axes[r, 0].imshow(image.resize((DISPLAY_SIZE, DISPLAY_SIZE)))
        axes[r, 1].imshow(overlay(image, cam_t))
        axes[r, 2].imshow(overlay(image, cam_r))
        if r == 0:
            axes[r, 0].set_title("photo", fontsize=8)
            axes[r, 1].set_title(f"{model_name}\n(trained)", fontsize=8)
            axes[r, 2].set_title(f"{model_name}\n(random weights)", fontsize=8)
    fig.suptitle("Sanity check: heatmaps should change when the learned weights are removed")
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


def run_gradcam(argv, started):
    """Grad-CAM from the command line. The W&B run, once started, is added to `started`, so that
    main() can close it."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", default=["outputs/final_model.pt"],
                   help="one or more model files; the first one picks the examples")
    p.add_argument("--sample", type=int, default=None,
                   help="scan a fixed random sample of this many test images (default: all 8,318)")
    p.add_argument("--n-correct", type=int, default=8, help="correct examples, one per class")
    p.add_argument("--n-wrong", type=int, default=4, help="most confident mistakes to show")
    p.add_argument("--n-distorted", type=int, default=2,
                   help="photos in the distortion figure (0 = no distortion figure)")
    p.add_argument("--attention-sample", type=int, default=300,
                   help="random photos for the heat-on-leaf measure and the sanity check "
                        "(0 = skip both)")
    p.add_argument("--random-seeds", type=int, default=5,
                   help="random initialisations for the sanity check (default 5)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--run-name", default="gradcam", help="name of the WandB run")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--no-wandb", action="store_true", help="do not log to Weights & Biases")
    args = p.parse_args(argv)
    if args.sample is not None and args.sample < 1:
        p.error("--sample must be at least 1")
    if args.batch_size < 1:
        p.error("--batch-size must be at least 1")
    if args.random_seeds < 1:
        p.error("--random-seeds must be at least 1")
    for option in ("n_correct", "n_wrong", "n_distorted", "attention_sample"):
        if getattr(args, option) < 0:
            p.error(f"--{option.replace('_', '-')} cannot be negative")
    if args.n_correct == 0 and args.n_wrong == 0:
        p.error("--n-correct and --n-wrong cannot both be 0: there would be nothing to explain")
    missing = [path for path in args.checkpoint if not os.path.exists(path)]
    if missing:
        p.error(f"checkpoint not found: {', '.join(missing)}")

    from src import robustness  # needs the dataset library, so imported only here

    error = robustness.duplicate_names_error(args.checkpoint)
    if error:
        p.error(error)

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
    if not correct and not wrong:
        p.error(f"no examples to explain: {first} has no correct predictions or no mistakes of "
                "the kind requested on these images (try other --n-correct / --n-wrong values)")
    chosen = [indices[i] for i in correct + wrong]

    # Started only now that the inputs are checked and the data, models and examples are ready,
    # so a mistake in the command does not leave a crashed or empty run in the team's project
    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=args.run_name,
                         job_type="eval", tags=["eval", "gradcam"], config=vars(args))
        print(f"WandB run ID: {run.id}")
        started.append(run)
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
    if rows:
        figures["distortions"] = plot_distortions(
            rows, DISTORTION_LEVELS, class_names,
            os.path.join(args.output_dir, "gradcam_distortions.png"), first)

    # 6. Heat on the leaf for every model, and the sanity check, on random photos
    attention = None
    if args.attention_sample:
        no_leaf = class_names.index(NO_LEAF_CLASS) if NO_LEAF_CLASS in class_names else None
        height, width = transform(images[0]).shape[-2:]
        picked, photos, masks, skipped = choose_attention_photos(
            test, image_col, labels, indices, args.attention_sample, (width, height), no_leaf)
        if photos:
            seeds = tuple(SEED + k for k in range(args.random_seeds))
            per_photo, examples = measure_attention(models, photos, masks, device,
                                                    args.batch_size, seeds)
            per_photo.insert(0, "test_index", [int(indices[picked[i]]) for i in per_photo["photo"]])
            per_photo.to_csv(os.path.join(args.output_dir, "gradcam_attention.csv"), index=False)
            attention = summarise_attention(per_photo, first)
            attention.to_csv(os.path.join(args.output_dir, "gradcam_attention_summary.csv"),
                             index=False)
            sanity = summarise_sanity(per_photo)
            sanity.to_csv(os.path.join(args.output_dir, "gradcam_sanity_summary.csv"), index=False)
            print(f"Heat on the leaf ({len(photos)} random photos, {skipped} skipped because the "
                  f"leaf mask looked wrong):\n" + attention.round(3).to_string(index=False))
            corr = sanity["rank_corr"]
            print(f"Sanity check: rank correlation between {first} and the same model with random "
                  f"weights, over {len(sanity)} random initialisation(s): mean {corr.mean():.2f}, "
                  f"from {corr.min():.2f} to {corr.max():.2f}\n"
                  + sanity.round(3).to_string(index=False))
            figures["leaf_masks"] = plot_leaf_masks(
                photos, masks, os.path.join(args.output_dir, "gradcam_leaf_masks.png"))
            figures["sanity"] = plot_sanity(
                photos[:4], examples["trained"], examples["random"],
                os.path.join(args.output_dir, "gradcam_sanity.png"), first)
        else:
            print("Heat on the leaf: no photos with a plausible leaf mask, so nothing measured")

    print(table.to_string(index=False))
    print(f"Saved {len(figures)} figure(s) and the tables to {args.output_dir}")

    if run:
        run.log({f"gradcam/{name}": wandb.Image(path) for name, path in figures.items()}
                | {"gradcam/examples_table": wandb.Table(dataframe=table),
                   "gradcam/mistakes_table": wandb.Table(dataframe=mistakes),
                   "gradcam/confused_pairs": wandb.Table(dataframe=pairs)})
        run.summary["gradcam/scanned_images"] = len(indices)
        if attention is not None:
            run.log({"gradcam/attention_summary": wandb.Table(dataframe=attention)})
            for _, row in attention.iterrows():
                run.summary[f"gradcam/heat_on_leaf/{row['model']}"] = row["heat_on_leaf"]
            run.log({"gradcam/sanity_summary": wandb.Table(dataframe=sanity)})
            run.summary["gradcam/sanity_rank_corr"] = corr.mean()
            run.summary["gradcam/sanity_rank_corr_min"] = corr.min()
            run.summary["gradcam/sanity_rank_corr_max"] = corr.max()
            run.summary["gradcam/attention_photos"] = len(photos)
            run.summary["gradcam/attention_skipped"] = skipped
        for key, value in stats.items():
            if value is not None:
                run.summary[f"gradcam/{key}"] = value
        print(f"Logged to WandB run {run.id}: {run.url}")
    return table


def main(argv=None):
    """Closes the W&B run even if something fails halfway, so no run is left marked as running
    in the team's project. A run that crashed is marked as failed, so it is not mistaken for a
    finished result."""
    started = []
    try:
        table = run_gradcam(argv, started)
    except BaseException:  # also Ctrl+C: the run did not finish either
        for run in started:
            run.finish(exit_code=1)
        raise
    for run in started:
        run.finish()
    return table


if __name__ == "__main__":
    main()
