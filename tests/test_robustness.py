import numpy as np
import pytest
from PIL import Image

from src import robustness

datasets = pytest.importorskip("datasets")


def leaf(width=96, height=72):
    """A small fake leaf photo: a green ellipse with a brown spot on a beige background."""
    image = Image.new("RGB", (width, height), (200, 190, 170))
    pixels = np.asarray(image).copy()
    yy, xx = np.mgrid[:height, :width]
    inside = ((xx - width / 2) / (width / 3)) ** 2 + ((yy - height / 2) / (height / 3)) ** 2 < 1
    pixels[inside] = (60, 140, 50)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("name", list(robustness.DISTORTIONS))
def test_every_distortion_keeps_size_and_colour_mode(name):
    image = leaf()
    for strength in robustness.DISTORTIONS[name][1]:
        out = robustness.apply_distortion(image, name, strength)
        assert out.size == image.size
        assert out.mode == "RGB"


def test_clean_returns_the_image_unchanged():
    image = leaf()
    assert robustness.apply_distortion(image, "clean", None) is image


def test_noise_is_repeatable_and_depends_on_the_seed():
    image = leaf()
    first = np.asarray(robustness.apply_distortion(image, "noise", 25, seed=1))
    again = np.asarray(robustness.apply_distortion(image, "noise", 25, seed=1))
    other = np.asarray(robustness.apply_distortion(image, "noise", 25, seed=2))
    assert (first == again).all()
    assert (first != other).any()


def test_dark_and_bright_change_brightness_in_the_right_direction():
    image = leaf()
    mean = np.asarray(image).mean()
    assert np.asarray(robustness.apply_distortion(image, "dark", 0.5)).mean() < mean
    assert np.asarray(robustness.apply_distortion(image, "bright", 1.5)).mean() > mean


def test_rotation_fills_corners_with_the_background_colour():
    rotated = np.asarray(robustness.apply_distortion(leaf(), "rotation", 30))
    assert tuple(rotated[0, 0]) != (0, 0, 0)


def test_levels_start_with_clean_and_cover_every_strength():
    levels = robustness.all_levels()
    assert levels[0] == ("clean", None)
    assert len(levels) == 1 + sum(len(s) for _, s in robustness.DISTORTIONS.values())


def test_sample_indices_is_fixed_and_sorted():
    assert robustness.sample_indices(10) == list(range(10))
    sample = robustness.sample_indices(100, sample=20)
    assert len(sample) == 20 and sample == sorted(sample)
    assert sample == robustness.sample_indices(100, sample=20)


def test_evaluate_summarise_and_plot(checkpoint, tmp_path):
    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(num_classes=39),
    })
    test = datasets.Dataset.from_dict(
        {"image": [leaf() for _ in range(6)], "label": [0, 1, 2, 0, 1, 2]}, features=features
    )
    models = robustness.load_models([str(checkpoint)], "cpu")
    levels = [("clean", None), ("blur", 2), ("noise", 25)]
    results = robustness.evaluate_distortions(
        models, test, "image", "label", 39, list(range(6)), batch_size=4, levels=levels
    )
    assert len(results) == len(levels)
    assert results["accuracy"].between(0, 1).all()

    summary = robustness.summarise(results)
    assert list(summary.index) == ["cnn"]
    assert summary.loc["cnn", "worst_case"] in {"blur 2", "noise 25"}

    full = robustness.evaluate_distortions(
        models, test, "image", "label", 39, [0, 1], batch_size=2
    )
    path = tmp_path / "chart.png"
    robustness.plot_results(full, str(path))
    assert path.exists()
