"""Image distortions used by the robustness test (src/robustness.py).

Each distortion takes a PIL image and a strength, and returns a new PIL image of the same size.
This module only needs numpy and Pillow, so its tests run even without the dataset library.
"""

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def gaussian_blur(image, sigma):
    """Out-of-focus or shaky photo. sigma = blur radius in pixels."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_noise(image, std, seed=0):
    """Grainy photo taken in low light. std = noise strength on the 0-255 pixel scale."""
    rng = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.float32)
    noisy = pixels + rng.normal(0.0, std, size=pixels.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def brightness(image, factor):
    """Shade (factor below 1) or strong sunlight (factor above 1). 1 = unchanged."""
    return ImageEnhance.Brightness(image).enhance(factor)


def border_colour(image):
    """Average colour of the image's outer edge (its background)."""
    pixels = np.asarray(image)
    edges = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    return tuple(int(v) for v in edges.mean(axis=0))


def rotation(image, degrees):
    """Photo taken at an angle. The empty corners are filled with the background colour rather
    than black, so the model cannot use black corners as a clue."""
    return image.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=border_colour(image))


def jpeg_compression(image, quality):
    """Photo sent through a messaging app. quality 100 = best, 1 = worst."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def zoom(image, keep):
    """Close-up showing only the middle of the leaf. keep = share of the width and height kept."""
    width, height = image.size
    w, h = int(width * keep), int(height * keep)
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h)).resize(
        (width, height), Image.Resampling.BILINEAR
    )


# name: (function, three strengths from mild to severe)
DISTORTIONS = {
    "blur": (gaussian_blur, [1, 2, 4]),
    "noise": (gaussian_noise, [10, 25, 50]),
    "dark": (brightness, [0.7, 0.5, 0.3]),
    "bright": (brightness, [1.3, 1.6, 2.0]),
    "rotation": (rotation, [30, 60, 90]),
    "jpeg": (jpeg_compression, [30, 15, 5]),
    "zoom": (zoom, [0.8, 0.6, 0.4]),
}


def apply_distortion(image, name, strength, seed=0):
    """Apply one distortion by name. The name "clean" returns the image unchanged.

    seed is only used by the noise distortion."""
    if name == "clean":
        return image
    function, _ = DISTORTIONS[name]
    if function is gaussian_noise:
        return function(image, strength, seed=seed)
    return function(image, strength)