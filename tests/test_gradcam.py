"""Tests for src/gradcam.py. The Grad-CAM tests need no dataset library; only the test of main()
does, so only that one is skipped where it is missing."""

import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from src import gradcam
from src.models import PlantCNN, ResidualCNN, resnet18_finetune
from tests.eval_fakes import fake_test_split, fake_wandb
from tests.test_distortions import leaf


class BrightnessModel(nn.Module):
    """A model whose answer we know: class 0 = "bright", class 1 = "dark".

    Its only feature map is the brightness of each pixel, so Grad-CAM for class 0 must light up
    exactly where the image is bright, and class 1 has no positive evidence anywhere."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 2, bias=False))
        with torch.no_grad():
            self.features.weight.fill_(1.0)
            self.head[2].weight.copy_(torch.tensor([[1.0], [-1.0]]))

    def forward(self, x):
        return self.head(torch.relu(self.features(x)))


def bright_patch():
    """A dark 1x3x16x16 image with a bright 4x4 patch in the top-left corner."""
    x = torch.zeros(1, 3, 16, 16)
    x[:, :, 2:6, 2:6] = 1.0
    return x


def test_target_layer_is_the_last_convolutional_stage():
    cnn = PlantCNN(depth=3)
    assert gradcam.target_layer(cnn) is cnn.features
    res = ResidualCNN(depth=2)
    assert gradcam.target_layer(res) is res.features
    resnet = resnet18_finetune(pretrained=False)
    assert gradcam.target_layer(resnet) is resnet.layer4
    plain = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Conv2d(4, 8, 3), nn.Flatten())
    assert gradcam.target_layer(plain) is plain[2]
    with pytest.raises(ValueError):
        gradcam.target_layer(nn.Sequential(nn.Flatten(), nn.Linear(3, 2)))


def test_heatmap_points_at_the_evidence():
    cams, probs = gradcam.grad_cam(BrightnessModel(), bright_patch())
    assert probs.argmax() == 0  # the image counts as "bright"
    cam = cams[0]
    assert cam.shape == (16, 16)
    assert cam.max() == pytest.approx(1.0)
    peak = np.unravel_index(cam.argmax(), cam.shape)
    assert 2 <= peak[0] < 6 and 2 <= peak[1] < 6  # the hottest pixel is inside the patch
    assert cam[10:, 10:].max() < 0.05  # the dark corner far away is cold


def test_a_class_without_evidence_gives_an_empty_map():
    cams, _ = gradcam.grad_cam(BrightnessModel(), bright_patch(), class_idx=1)
    assert np.all(cams == 0)
    assert not np.isnan(cams).any()


def test_grad_cam_leaves_the_model_untouched():
    model = PlantCNN(base_channels=8, depth=3, dropout=0.0).eval()
    inputs = torch.randn(3, 3, 64, 64)
    cams, probs = gradcam.grad_cam(model, inputs)
    assert cams.shape == (3, 64, 64)
    assert cams.min() >= 0 and cams.max() <= 1
    assert probs.shape == (3, 39)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert not model.training
    assert all(not m._forward_hooks for m in model.modules())  # the hook was removed
    assert all(p.grad is None for p in model.parameters())  # no gradients left behind


def test_grad_cam_works_through_a_frozen_backbone():
    model = resnet18_finetune(pretrained=False, freeze_backbone=True).eval()
    cams, _ = gradcam.grad_cam(model, torch.randn(1, 3, 64, 64))
    assert cams.shape == (1, 64, 64)


def test_overlay_is_bright_where_the_heatmap_is_hot():
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    cam = np.zeros((16, 16))
    cam[:8, :8] = 1.0
    out = np.asarray(gradcam.overlay(image, cam, size=(64, 64))).astype(int)
    assert out.shape == (64, 64, 3)
    hot, cold = out[8, 8], out[56, 56]
    assert hot.sum() > cold.sum() + 150  # hot regions are much brighter (yellow vs dark)
    assert hot[0] > hot[2]  # and yellow rather than blue


def test_choose_examples():
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    preds = np.array([0, 0, 1, 2, 2, 0, 3, 3])
    confidence = np.array([0.9, 0.8, 0.9, 0.7, 0.9, 0.95, 0.6, 0.9])
    correct, wrong = gradcam.choose_examples(labels, preds, confidence, n_correct=3, n_wrong=5)
    assert len(correct) == 3
    assert all(labels[i] == preds[i] for i in correct)
    assert len({labels[i] for i in correct}) == 3  # all from different classes
    assert wrong == [5, 3]  # every mistake, the most confident first
    again = gradcam.choose_examples(labels, preds, confidence, n_correct=3, n_wrong=5)
    assert again == (correct, wrong)  # repeatable


def test_help_keeps_the_example_command_on_two_lines():
    assert "outputs/cnn_augment-none.pt \\\n" in gradcam.__doc__


def test_label_is_readable():
    assert gradcam.label("Tomato___Early_blight") == "Tomato: Early blight"
    assert gradcam.mark(True) == "\u2713" and gradcam.mark(False) == "\u2717"


def test_choose_random_ignores_predictions():
    labels = np.repeat(np.arange(5), 4)
    picked = gradcam.choose_random(labels, n=4)
    assert len(picked) == 4
    assert len({labels[i] for i in picked}) == 4  # all from different classes
    assert picked == gradcam.choose_random(labels, n=4)  # repeatable
    # no predictions are passed in, so the choice cannot favour any model


def test_pick_for_distortions_puts_a_diseased_leaf_first():
    names = ["Strawberry___healthy", "Background_without_leaves", "Tomato___Early_blight",
             "Potato___healthy", "Corn___Common_rust"]
    assert gradcam.pick_for_distortions(names, n=2) == [2, 0]  # diseased, then healthy
    assert gradcam.pick_for_distortions(names, n=5) == [2, 0, 4, 3, 1]  # background last
    assert gradcam.pick_for_distortions(["class_0", "class_1"], n=2) == [0, 1]


def test_summarise_mistakes():
    labels = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array([0, 1, 1, 0, 1, 2])
    confidence = np.array([0.99, 0.95, 0.8, 0.6, 0.97, 0.9])
    names = ["a", "b", "c"]
    mistakes, pairs, stats = gradcam.summarise_mistakes(labels, preds, confidence, names,
                                                        indices=[10, 11, 12, 13, 14, 15])
    assert list(mistakes["test_index"]) == [14, 11, 13]  # most confident first
    assert list(mistakes.columns) == ["test_index", "true", "predicted", "confidence"]
    assert pairs.iloc[0].to_dict() == {"true": "a", "predicted": "b", "count": 1}
    assert pairs["count"].sum() == 3
    assert stats["mistakes"] == 3
    assert stats["confident_mistakes"] == 2  # 0.97 and 0.95 are at least 90%
    assert stats["confident_share"] == pytest.approx(2 / 3)
    assert stats["median_confidence_mistakes"] == pytest.approx(0.95)
    assert stats["median_confidence_correct"] == pytest.approx(0.9)

    _, no_pairs, perfect = gradcam.summarise_mistakes(labels, labels, confidence, names, range(6))
    assert perfect["mistakes"] == 0 and perfect["median_confidence_mistakes"] is None
    assert no_pairs.empty


def test_explain_class_explains_the_requested_class():
    model = PlantCNN(base_channels=8, depth=3, dropout=0.0).eval()
    from src.transforms import get_transforms

    images = [leaf(), leaf()]
    cams, probs = gradcam.explain_class(model, get_transforms(64), images, [3, 7])
    assert cams.shape == (2, 64, 64)
    _, all_probs = gradcam.grad_cam(model, torch.stack([get_transforms(64)(i) for i in images]))
    assert probs == pytest.approx(all_probs[[0, 1], [3, 7]], abs=1e-6)


def test_main_saves_figures_and_logs_to_wandb(checkpoint, tmp_path, monkeypatch):
    """main() with a fake dataset, two models and a fake wandb: no download or account needed."""
    pytest.importorskip("datasets")
    from src import data

    assert gradcam.SEED == data.SEED

    fake_test_split(monkeypatch, n_images=6)
    monkeypatch.setattr(gradcam, "get_device", lambda: "cpu")
    second = tmp_path / "other.pt"
    shutil.copy(checkpoint, second)
    started, logged, run = fake_wandb(monkeypatch)

    out = tmp_path / "gradcam"
    table = gradcam.main(["--checkpoint", str(checkpoint), str(second), "--output-dir", str(out),
                          "--n-correct", "2", "--n-wrong", "2", "--batch-size", "4",
                          "--random-seeds", "2"])
    for name in ["gradcam_examples.png", "gradcam_mistakes.png", "gradcam_models.png",
                 "gradcam_distortions.png", "gradcam_examples.csv", "gradcam_mistakes.csv",
                 "gradcam_confused_pairs.csv", "gradcam_attention.csv",
                 "gradcam_attention_summary.csv", "gradcam_leaf_masks.png", "gradcam_sanity.png",
                 "gradcam_sanity_summary.csv"]:
        assert (out / name).exists(), name
    assert set(table.columns) == {"test_index", "kind", "true", "predicted", "confidence"}
    assert started[0]["tags"] == ["eval", "gradcam"]
    assert {"gradcam/examples", "gradcam/mistakes", "gradcam/models", "gradcam/distortions",
            "gradcam/mistakes_table", "gradcam/confused_pairs"} <= set(logged)
    assert run.summary["gradcam/scanned_images"] == 6
    assert run.summary["gradcam/mistakes"] == len(pd.read_csv(out / "gradcam_mistakes.csv"))
    assert "gradcam/attention_summary" in logged
    assert "gradcam/sanity_summary" in logged
    sanity = pd.read_csv(out / "gradcam_sanity_summary.csv")
    assert len(sanity) == 2  # one row per random seed
    assert run.summary["gradcam/sanity_rank_corr"] == pytest.approx(sanity["rank_corr"].mean())
    assert run.summary["gradcam/sanity_rank_corr_min"] == pytest.approx(sanity["rank_corr"].min())
    assert run.summary["gradcam/sanity_rank_corr_max"] == pytest.approx(sanity["rank_corr"].max())
    summary = pd.read_csv(out / "gradcam_attention_summary.csv")
    assert len(summary) == 4  # two models and the first one with two sets of random weights
    assert summary["model"].iloc[-1].endswith("(random weights, seed 43)")
    assert logged["exit_codes"] == [None]  # closed once, as a finished run


def fake_run_setup(monkeypatch, n_images=6):
    """Fake test split on the CPU and a fake wandb that records what is started and logged."""
    fake_test_split(monkeypatch, n_images)
    monkeypatch.setattr(gradcam, "get_device", lambda: "cpu")
    started, logged, _ = fake_wandb(monkeypatch)
    return started, logged


@pytest.mark.parametrize("bad, message", [
    (["--sample", "0"], "--sample must be at least 1"),
    (["--batch-size", "0"], "--batch-size must be at least 1"),
    (["--n-correct", "-1"], "--n-correct cannot be negative"),
    (["--n-wrong", "-1"], "--n-wrong cannot be negative"),
    (["--n-distorted", "-1"], "--n-distorted cannot be negative"),
    (["--attention-sample", "-1"], "--attention-sample cannot be negative"),
    (["--random-seeds", "0"], "--random-seeds must be at least 1"),
    (["--n-correct", "0", "--n-wrong", "0"], "cannot both be 0"),
    (["--checkpoint", "missing.pt"], "checkpoint not found: missing.pt"),
])
def test_bad_input_stops_with_a_clear_message_before_any_wandb_run(
        bad, message, checkpoint, tmp_path, monkeypatch, capsys):
    started, _ = fake_run_setup(monkeypatch)
    with pytest.raises(SystemExit):
        gradcam.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path)] + bad)
    assert message in capsys.readouterr().err
    assert started == []  # nothing is left behind in WandB


def test_nothing_to_explain_stops_before_any_wandb_run(checkpoint, tmp_path, monkeypatch, capsys):
    """A model that is wrong on every image, asked for correct examples only."""
    from src import robustness
    from src.transforms import get_transforms

    class AlwaysWrong(nn.Module):
        def forward(self, x):
            logits = torch.zeros(len(x), 39)
            logits[:, 38] = 10.0  # the fake images are all labelled 0, 1 or 2
            return logits

    started, _ = fake_run_setup(monkeypatch)
    monkeypatch.setattr(robustness, "load_models",
                        lambda paths, device: {"wrong": (AlwaysWrong(), get_transforms(64))})
    with pytest.raises(SystemExit):
        gradcam.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path),
                      "--n-wrong", "0"])
    assert "no examples to explain" in capsys.readouterr().err
    assert started == []


def test_n_distorted_0_skips_the_distortion_figure(checkpoint, tmp_path, monkeypatch):
    started, logged = fake_run_setup(monkeypatch)
    gradcam.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path),
                  "--n-correct", "2", "--n-wrong", "2", "--n-distorted", "0"])
    assert not (tmp_path / "gradcam_distortions.png").exists()
    assert (tmp_path / "gradcam_examples.png").exists()
    assert "gradcam/distortions" not in logged
    assert len(started) == 1 and logged["exit_codes"] == [None]


def leaf_with_shadow():
    """A 128x128 photo: grey paper, a dark grey shadow, and a green leaf with brown spots."""
    image = Image.new("RGB", (128, 128), (200, 195, 190))
    pixels = np.asarray(image).copy()
    yy, xx = np.mgrid[:128, :128]
    shadow = ((xx - 74) / 40) ** 2 + ((yy - 74) / 30) ** 2 < 1
    leaf_area = ((xx - 64) / 40) ** 2 + ((yy - 64) / 30) ** 2 < 1
    pixels[shadow] = (95, 95, 100)
    pixels[leaf_area] = (60, 130, 50)
    for cx, cy in [(55, 60), (70, 70), (64, 50)]:
        pixels[(xx - cx) ** 2 + (yy - cy) ** 2 < 9] = (120, 80, 40)
    return Image.fromarray(pixels), leaf_area


def test_leaf_mask_finds_the_leaf_but_not_the_shadow_or_background():
    image, truth = leaf_with_shadow()
    mask = gradcam.leaf_mask(image, (128, 128))
    overlap = (mask & truth).sum() / (mask | truth).sum()
    assert overlap > 0.9  # the leaf, including its brown spots
    from scipy import ndimage

    # the shadow is not leaf (apart from the soft 1-2 pixel edge next to the leaf)
    shadow_only = ~ndimage.binary_dilation(truth, iterations=3) & (np.asarray(image)[..., 0] < 100)
    assert shadow_only.sum() > 500
    assert not mask[shadow_only].any()


def test_heat_on_leaf():
    mask = np.zeros((4, 4), bool)
    mask[:2] = True
    assert gradcam.heat_on_leaf(mask.astype(float), mask) == 1.0
    assert gradcam.heat_on_leaf(np.ones((4, 4)), mask) == 0.5  # even heat = the leaf's area
    assert np.isnan(gradcam.heat_on_leaf(np.zeros((4, 4)), mask))


def test_randomised_copy_leaves_the_original_untouched():
    model = PlantCNN(base_channels=8, depth=3).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    random_model = gradcam.randomised_copy(model)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert not torch.equal(random_model.features[0][0].weight, model.features[0][0].weight)
    assert not random_model.training
    again = gradcam.randomised_copy(model)
    assert torch.equal(again.features[0][0].weight, random_model.features[0][0].weight)
    other = gradcam.randomised_copy(model, seed=7)
    assert not torch.equal(other.features[0][0].weight, random_model.features[0][0].weight)


def test_randomised_copy_draws_the_weights_on_the_cpu(monkeypatch):
    """The same seed must give the same weights on any device: a Mac GPU draws other numbers
    than a CPU, so the copy is moved to the CPU, the weights are drawn, and only then is it moved
    back to the model's device."""
    model = PlantCNN(base_channels=8, depth=3).eval()
    events = []
    real_cpu, real_to, real_reset = nn.Module.cpu, nn.Module.to, nn.Conv2d.reset_parameters
    monkeypatch.setattr(nn.Module, "cpu", lambda self: events.append("cpu") or real_cpu(self))
    monkeypatch.setattr(nn.Module, "to",
                        lambda self, *a, **k: events.append(("to", *a)) or real_to(self, *a, **k))
    monkeypatch.setattr(nn.Conv2d, "reset_parameters",
                        lambda self: events.append("reset") or real_reset(self))
    gradcam.randomised_copy(model)
    assert events[0] == "cpu" and "reset" in events
    assert events[-1] == ("to", torch.device("cpu"))  # back to the model's own device
    assert events.index(events[-1]) > max(i for i, e in enumerate(events) if e == "reset")


def test_bootstrap_ci_and_rank_correlation():
    assert gradcam.bootstrap_ci([0.5, 0.5, np.nan]) == (0.5, 0.5, 0.5)
    mean, low, high = gradcam.bootstrap_ci(np.arange(100) / 100)
    assert low < mean < high and mean == pytest.approx(0.495)
    assert gradcam.bootstrap_ci(np.arange(100) / 100) == (mean, low, high)  # repeatable
    assert all(np.isnan(gradcam.bootstrap_ci([np.nan])))
    cam = np.random.default_rng(0).random((16, 16))
    assert gradcam.rank_correlation(cam, cam) == pytest.approx(1.0)
    assert np.isnan(gradcam.rank_correlation(cam, np.zeros((16, 16))))


class GreenModel(nn.Module):
    """Answers class 0 with evidence only where the photo is green, so its heat is on the leaf."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 2, bias=False))
        with torch.no_grad():
            self.features.weight.copy_(torch.tensor([-1.0, 1.0, -1.0]).view(1, 3, 1, 1))
            self.head[2].weight.copy_(torch.tensor([[1.0], [-1.0]]))

    def forward(self, x):
        return self.head(torch.relu(self.features(x)))


def test_measure_attention_sees_heat_on_the_leaf():
    from src.transforms import get_transforms

    image, _ = leaf_with_shadow()
    transform = get_transforms(64)
    mask = gradcam.leaf_mask(image, (64, 64))
    models = {"green": (GreenModel().eval(), transform), "same": (GreenModel().eval(), transform)}
    per_photo, examples = gradcam.measure_attention(models, [image, image], [mask, mask],
                                                    seeds=(1, 2))
    green = per_photo[per_photo["model"] == "green"]
    assert (green["heat_on_leaf"] > 0.9).all()  # almost all heat is on the leaf
    assert green["leaf_area"].iloc[0] == pytest.approx(mask.mean())
    assert per_photo["rank_corr_with_trained"].notna().sum() == 4  # 2 photos x 2 random seeds
    assert per_photo.loc[per_photo["rank_corr_with_trained"].notna(), "model"].str.contains(
        "random weights").all()
    assert len(examples["trained"]) == 2 and len(examples["random"]) == 2
    seed_1, seed_2 = (per_photo[per_photo["model"] == f"green (random weights, seed {k})"]
                      for k in (1, 2))
    assert not np.allclose(seed_1["heat_on_leaf"], seed_2["heat_on_leaf"])  # different weights

    summary = gradcam.summarise_attention(per_photo, "green")
    assert list(summary["model"]) == ["green", "same", "green (random weights, seed 1)",
                                      "green (random weights, seed 2)"]
    assert summary.loc[1, "diff_vs_first"] == pytest.approx(0.0)  # same model, same photos
    assert np.isnan(summary.loc[0, "diff_vs_first"])


def test_choose_attention_photos_skips_bad_masks_and_no_leaf_photos():
    datasets = pytest.importorskip("datasets")
    image, _ = leaf_with_shadow()
    blank = Image.new("RGB", (128, 128), (200, 195, 190))  # no leaf: the mask is empty
    features = datasets.Features({"image": datasets.Image(),
                                  "label": datasets.ClassLabel(num_classes=39)})
    test = datasets.Dataset.from_dict({"image": [image, blank, image],
                                       "label": [0, 1, 38]}, features=features)
    chosen, photos, masks, skipped = gradcam.choose_attention_photos(
        test, "image", np.array([0, 1, 38]), [0, 1, 2], n=5, size=(64, 64), no_leaf_class=38)
    assert chosen == [0] and len(photos) == 1 and len(masks) == 1
    assert skipped == 1  # the blank photo; the no-leaf photo is not even tried


def test_attention_sample_0_skips_the_measurements(checkpoint, tmp_path, monkeypatch):
    started, logged = fake_run_setup(monkeypatch)
    gradcam.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path),
                  "--n-correct", "2", "--n-wrong", "2", "--attention-sample", "0"])
    assert not (tmp_path / "gradcam_attention.csv").exists()
    assert "gradcam/attention_summary" not in logged
    assert len(started) == 1 and logged["exit_codes"] == [None]


def test_leaf_mask_drops_small_specks():
    image, truth = leaf_with_shadow()
    pixels = np.asarray(image).copy()
    pixels[2:6, 120:124] = (60, 130, 50)  # a tiny green speck in the corner
    mask = gradcam.leaf_mask(Image.fromarray(pixels), (128, 128))
    assert not mask[2:6, 120:124].any()
    assert (mask & truth).sum() / truth.sum() > 0.9


def test_measure_attention_with_models_of_different_image_sizes():
    """A second model with smaller inputs (like ResNet with 224 against 128) gets a resized mask."""
    from src.transforms import get_transforms

    image, _ = leaf_with_shadow()
    mask = gradcam.leaf_mask(image, (64, 64))
    models = {"big": (GreenModel().eval(), get_transforms(64)),
              "small": (GreenModel().eval(), get_transforms(32))}
    per_photo, _ = gradcam.measure_attention(models, [image], [mask])
    small = per_photo[per_photo["model"] == "small"].iloc[0]
    assert small["leaf_area"] == pytest.approx(mask.mean(), abs=0.03)
    assert small["heat_on_leaf"] > 0.9


def test_choose_attention_photos_stops_at_n():
    datasets = pytest.importorskip("datasets")
    image, _ = leaf_with_shadow()
    features = datasets.Features({"image": datasets.Image(),
                                  "label": datasets.ClassLabel(num_classes=39)})
    test = datasets.Dataset.from_dict({"image": [image] * 3, "label": [0, 1, 2]},
                                      features=features)
    chosen, photos, _, skipped = gradcam.choose_attention_photos(
        test, "image", np.array([0, 1, 2]), [0, 1, 2], n=2, size=(64, 64))
    assert len(chosen) == 2 and len(photos) == 2 and skipped == 0


def test_leaf_mask_figure_with_an_uneven_number_of_photos(tmp_path):
    image, _ = leaf_with_shadow()
    mask = gradcam.leaf_mask(image, (64, 64))
    path = gradcam.plot_leaf_masks([image] * 7, [mask] * 7, str(tmp_path / "masks.png"))
    assert (tmp_path / "masks.png").exists() and path.endswith("masks.png")


def test_no_plausible_leaf_masks_measures_nothing(checkpoint, tmp_path, monkeypatch, capsys):
    started, logged = fake_run_setup(monkeypatch)
    monkeypatch.setattr(gradcam, "LEAF_AREA_RANGE", (0.99, 1.0))  # every mask counts as wrong
    gradcam.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path),
                  "--n-correct", "2", "--n-wrong", "2"])
    assert "nothing measured" in capsys.readouterr().out
    assert not (tmp_path / "gradcam_attention.csv").exists()
    assert len(started) == 1 and logged["exit_codes"] == [None]


def test_leaf_mask_leaves_out_a_coloured_background():
    """Pink-brown paper is strongly coloured, like a lesion; the border shows it is background."""
    from PIL import ImageDraw

    image = Image.new("RGB", (128, 128), (130, 93, 91))
    ImageDraw.Draw(image).ellipse((34, 14, 94, 114), fill=(77, 78, 31))
    noise = np.random.default_rng(0).normal(0, 6, (128, 128, 3))
    image = Image.fromarray(np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255)
                            .astype(np.uint8))
    yy, xx = np.mgrid[:128, :128]
    truth = ((xx - 64) / 30) ** 2 + ((yy - 64) / 50) ** 2 < 1
    mask = gradcam.leaf_mask(image, (128, 128))
    assert (mask & truth).sum() / (mask | truth).sum() > 0.9


def test_leaf_mask_keeps_a_leaf_that_fills_the_frame():
    """Corn close-ups have no background: the border is leaf, so nothing is removed."""
    green = np.zeros((128, 128, 3), np.uint8)
    green[...] = (90, 140, 60)
    green[:, 60:68] = (200, 200, 170)  # a pale midrib
    mask = gradcam.leaf_mask(Image.fromarray(green), (128, 128))
    assert mask.mean() > 0.9


def test_leaf_mask_leaves_out_a_shadow_on_grey_background():
    """A dark shadow with a yellowish tint passes the hue test; it has the background's colour balance."""
    from PIL import ImageDraw

    image = Image.new("RGB", (128, 128), (150, 150, 142))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 26, 110, 116), fill=(88, 90, 80))  # shadow, offset down and right
    draw.ellipse((30, 14, 98, 104), fill=(122, 154, 58))  # yellow-green leaf
    yy, xx = np.mgrid[:128, :128]
    leaf = ((xx - 64) / 34) ** 2 + ((yy - 59) / 45) ** 2 < 1
    mask = gradcam.leaf_mask(image, (128, 128))
    assert (mask & leaf).sum() / (mask | leaf).sum() > 0.9


def test_leaf_mask_keeps_pale_leaf_parts_on_grey_background():
    """Powdery mildew is pale and nearly grey; brighter than the background, so not a shadow."""
    from PIL import ImageDraw

    image = Image.new("RGB", (128, 128), (130, 130, 130))
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 14, 104, 114), fill=(90, 140, 60))
    draw.chord((24, 14, 104, 114), 180, 360, fill=(185, 198, 175))  # pale top half of the leaf
    yy, xx = np.mgrid[:128, :128]
    leaf = ((xx - 64) / 40) ** 2 + ((yy - 64) / 50) ** 2 < 1
    mask = gradcam.leaf_mask(image, (128, 128))
    assert (mask & leaf).sum() / (mask | leaf).sum() > 0.9


def test_leaf_mask_keeps_dry_lesions_next_to_a_beige_background():
    """Dry tan tissue has nearly the colour balance of beige paper, but it is part of the leaf."""
    from PIL import ImageDraw

    image = Image.new("RGB", (128, 128), (200, 180, 150))
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 14, 104, 114), fill=(90, 140, 60))
    draw.chord((24, 14, 104, 114), 90, 270, fill=(178, 152, 110))  # dry left half of the leaf
    yy, xx = np.mgrid[:128, :128]
    leaf = ((xx - 64) / 40) ** 2 + ((yy - 64) / 50) ** 2 < 1
    mask = gradcam.leaf_mask(image, (128, 128))
    assert (mask & leaf).sum() / (mask | leaf).sum() > 0.9


def test_summarise_sanity_gives_one_row_per_seed():
    per_photo = pd.DataFrame({
        "photo": [0, 1, 0, 1, 0, 1],
        "model": ["m"] * 2 + ["m (random weights, seed 1)"] * 2 + ["m (random weights, seed 2)"] * 2,
        "heat_on_leaf": [0.9, 0.8, 0.5, 0.4, 0.6, 0.6],
        "rank_corr_with_trained": [np.nan, np.nan, 0.1, 0.3, 0.5, 0.5],
    })
    sanity = gradcam.summarise_sanity(per_photo)
    assert list(sanity["model"]) == ["m (random weights, seed 1)", "m (random weights, seed 2)"]
    assert list(sanity["rank_corr"]) == pytest.approx([0.2, 0.5])
    assert list(sanity["heat_on_leaf"]) == pytest.approx([0.45, 0.6])
    assert (sanity["ci_low"] <= sanity["rank_corr"]).all()


def test_two_model_files_with_the_same_name_are_refused(checkpoint, tmp_path, monkeypatch, capsys):
    other = tmp_path / "new"
    other.mkdir()
    copy = other / checkpoint.name
    copy.write_bytes(checkpoint.read_bytes())
    started, _ = fake_run_setup(monkeypatch)
    with pytest.raises(SystemExit):
        gradcam.main(["--checkpoint", str(checkpoint), str(copy), "--output-dir", str(tmp_path)])
    assert "two model files have the same name" in capsys.readouterr().err
    assert started == []


@pytest.mark.parametrize("error", [RuntimeError("out of memory"), KeyboardInterrupt()])
def test_wandb_run_is_marked_failed_when_grad_cam_stops_halfway(
        error, checkpoint, tmp_path, monkeypatch):
    """A crash or Ctrl+C closes the run as failed (exit code 1), not as a finished result."""
    started, logged = fake_run_setup(monkeypatch)

    def crash(*args, **kwargs):
        raise error

    monkeypatch.setattr(gradcam, "plot_examples", crash)
    with pytest.raises(type(error)):
        gradcam.main(["--checkpoint", str(checkpoint), "--output-dir", str(tmp_path),
                      "--n-correct", "2", "--n-wrong", "2"])
    assert len(started) == 1
    assert logged["exit_codes"] == [1]  # closed once, as failed, not left running

