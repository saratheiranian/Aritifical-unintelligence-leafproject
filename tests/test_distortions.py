"""Tests for src/distortions.py. These need only numpy and Pillow, so they also run where the
dataset library is not installed."""

import numpy as np
import pytest
from PIL import Image

from src import distortions


def leaf(width=96, height=72):
    """A small fake leaf photo: a green ellipse on a beige background."""
    image = Image.new("RGB", (width, height), (200, 190, 170))
    pixels = np.asarray(image).copy()
    yy, xx = np.mgrid[:height, :width]
    inside = ((xx - width / 2) / (width / 3)) ** 2 + ((yy - height / 2) / (height / 3)) ** 2 < 1
    pixels[inside] = (60, 140, 50)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("name", list(distortions.DISTORTIONS))
def test_every_distortion_keeps_size_and_colour_mode(name):
    image = leaf()
    for strength in distortions.DISTORTIONS[name][1]:
        out = distortions.apply_distortion(image, name, strength)
        assert out.size == image.size
        assert out.mode == "RGB"


def test_clean_returns_the_image_unchanged():
    image = leaf()
    assert distortions.apply_distortion(image, "clean", None) is image


def test_noise_is_repeatable_and_depends_on_the_seed():
    image = leaf()
    first = np.asarray(distortions.apply_distortion(image, "noise", 25, seed=1))
    again = np.asarray(distortions.apply_distortion(image, "noise", 25, seed=1))
    other = np.asarray(distortions.apply_distortion(image, "noise", 25, seed=2))
    assert (first == again).all()
    assert (first != other).any()


def test_dark_and_bright_change_brightness_in_the_right_direction():
    image = leaf()
    mean = np.asarray(image).mean()
    assert np.asarray(distortions.apply_distortion(image, "dark", 0.5)).mean() < mean
    assert np.asarray(distortions.apply_distortion(image, "bright", 1.5)).mean() > mean


def test_rotation_fills_corners_with_the_background_colour():
    rotated = np.asarray(distortions.apply_distortion(leaf(), "rotation", 30))
    assert tuple(rotated[0, 0]) != (0, 0, 0)