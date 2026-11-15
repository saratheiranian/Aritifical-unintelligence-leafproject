"""Tests for src/gradcam.py. The Grad-CAM tests need no dataset library; only the test of main()
does, so only that one is skipped where it is missing."""

import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from src import gradcam
from src.models import PlantCNN, ResidualCNN, resnet18_finetune
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
    datasets = pytest.importorskip("datasets")
    from src import data, robustness

    assert gradcam.SEED == data.SEED

    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(num_classes=39),
    })
    test = datasets.Dataset.from_dict(
        {"image": [leaf() for _ in range(6)], "label": [0, 1, 2, 0, 1, 2]}, features=features
    )
    monkeypatch.setattr(robustness, "load_test_split",
                        lambda: (test, "image", "label", [f"class_{i}" for i in range(39)]))
    monkeypatch.setattr(gradcam, "get_device", lambda: "cpu")
    second = tmp_path / "other.pt"
    shutil.copy(checkpoint, second)

    logged = {}
    run = types.SimpleNamespace(id="abc123", url="https://wandb.ai/run", summary={},
                                log=logged.update, finish=lambda: logged.update(finished=True))
    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: logged.update(init=kwargs) or run,
        Table=lambda dataframe: dataframe,
        Image=lambda path: path,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    out = tmp_path / "gradcam"
    table = gradcam.main(["--checkpoint", str(checkpoint), str(second), "--output-dir", str(out),
                          "--n-correct", "2", "--n-wrong", "2", "--batch-size", "4"])
    for name in ["gradcam_examples.png", "gradcam_mistakes.png", "gradcam_models.png",
                 "gradcam_distortions.png", "gradcam_examples.csv", "gradcam_mistakes.csv",
                 "gradcam_confused_pairs.csv"]:
        assert (out / name).exists(), name
    assert set(table.columns) == {"test_index", "kind", "true", "predicted", "confidence"}
    assert logged["init"]["tags"] == ["eval", "gradcam"]
    assert {"gradcam/examples", "gradcam/mistakes", "gradcam/models", "gradcam/distortions",
            "gradcam/mistakes_table", "gradcam/confused_pairs"} <= set(logged)
    assert run.summary["gradcam/scanned_images"] == 6
    assert run.summary["gradcam/mistakes"] == len(pd.read_csv(out / "gradcam_mistakes.csv"))
    assert logged["finished"]
