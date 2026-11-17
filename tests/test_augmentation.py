import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.augment_preview import augmented_copies, preview_figure
from src.sweep import EXPERIMENTS, to_cli
from src.train import parse_args
from src.transforms import (
    AUGMENT_LEVELS,
    RandomGaussianNoise,
    augmentation_steps,
    get_transforms,
)


def leaf(size=(256, 192)):
    image = Image.new("RGB", size, (40, 120, 40))
    image.paste(Image.new("RGB", (60, 60), (200, 180, 60)), (20, 20))
    return image


@pytest.mark.parametrize("level", AUGMENT_LEVELS + (True, False))
def test_every_level_gives_a_normalised_tensor_of_the_right_size(level):
    x = get_transforms(96, augment=level)(leaf())
    assert x.shape == (3, 96, 96) and x.dtype == torch.float32


def test_levels_add_more_random_steps():
    counts = [len(augmentation_steps(level)) for level in AUGMENT_LEVELS]
    assert counts[0] == 0 and counts[0] < counts[1] < counts[2]
    assert len(augmentation_steps(True)) == counts[1]
    assert augmentation_steps(False) == [] and augmentation_steps(None) == []


def test_strong_level_crops_to_the_image_size():
    crops = [s for s in augmentation_steps("strong", 112) if isinstance(s, transforms.RandomResizedCrop)]
    assert crops and crops[0].size == (112, 112)


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        get_transforms(64, augment="extreme")


def test_strong_augmentation_changes_the_image_but_none_does_not():
    torch.manual_seed(0)
    base = get_transforms(64, augment="none")(leaf())
    assert torch.equal(base, get_transforms(64, augment="none")(leaf()))
    assert not torch.equal(base, get_transforms(64, augment="strong")(leaf()))


def test_augmented_copies_are_reproducible():
    a = augmented_copies(leaf(), "strong", 2, img_size=64, seed=3)
    b = augmented_copies(leaf(), "strong", 2, img_size=64, seed=3)
    assert len(a) == 2 and all(im.size == (64, 64) for im in a)
    assert [list(im.getdata()) for im in a] == [list(im.getdata()) for im in b]


def test_preview_figure_has_one_column_per_level_and_copy():
    fig = preview_figure([leaf(), leaf()], titles=["a", "b"], copies=2, img_size=48)
    assert len(fig.axes) == 2 * len(AUGMENT_LEVELS) * 2
    assert [ax.get_title() for ax in fig.axes[:8:2]] == list(AUGMENT_LEVELS)


def test_train_cli_and_sweep_accept_augment_levels():
    assert parse_args([]).augment == "basic"
    assert parse_args(["--augment", "strong"]).augment == "strong"
    assert parse_args(["--no-augment"]).no_augment
    assert to_cli(EXPERIMENTS["augment-strong"]) == ["--augment", "strong"]


def test_noise_level_is_strong_plus_gaussian_noise():
    strong, noise = augmentation_steps("strong"), augmentation_steps("noise")
    assert len(noise) == len(strong) + 1
    assert isinstance(noise[-1], RandomGaussianNoise)


def test_gaussian_noise_changes_pixels_but_keeps_size_and_mode():
    image = leaf((64, 48))
    torch.manual_seed(0)
    out = RandomGaussianNoise(max_std=25, p=1.0)(image)
    assert out.size == image.size and out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(image))
    assert np.asarray(out).min() >= 0 and np.asarray(out).max() <= 255


def test_gaussian_noise_with_p_zero_returns_the_image_unchanged():
    image = leaf((32, 32))
    assert RandomGaussianNoise(p=0.0)(image) is image


def test_gaussian_noise_is_repeatable_with_a_seed():
    image = leaf((32, 32))
    torch.manual_seed(5)
    a = np.asarray(RandomGaussianNoise(p=1.0)(image))
    torch.manual_seed(5)
    b = np.asarray(RandomGaussianNoise(p=1.0)(image))
    assert np.array_equal(a, b)


def test_gaussian_noise_repr_shows_its_settings():
    assert repr(RandomGaussianNoise()) == "RandomGaussianNoise(max_std=25.0, p=0.5)"


def test_train_accepts_the_noise_level():
    assert parse_args(["--augment", "noise"]).augment == "noise"
