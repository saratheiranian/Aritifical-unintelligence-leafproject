import importlib.util
from pathlib import Path

import pytest
from PIL import Image


def load_app():
    pytest.importorskip("gradio")
    spec = importlib.util.spec_from_file_location("app", Path(__file__).parent.parent / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def make_photo(path, orientation=None):
    """A 40x20 photo with a red block in the top-left corner, saved the way a phone does."""
    photo = Image.new("RGB", (40, 20))
    photo.paste((255, 0, 0), (0, 0, 8, 8))
    if orientation is None:
        photo.save(path)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation  # EXIF tag that tells viewers how to turn the photo
        photo.save(path, exif=exif)


@pytest.mark.parametrize("orientation, size, red_at", [
    (3, (40, 20), (37, 17)),  # upside down: the red block ends up bottom-right
    (6, (20, 40), (17, 2)),   # turned 90 degrees: the red block ends up top-right
    (8, (20, 40), (2, 37)),   # turned 270 degrees: the red block ends up bottom-left
])
def test_phone_photo_is_turned_the_right_way_up(tmp_path, orientation, size, red_at):
    app = load_app()
    path = tmp_path / "phone.jpg"
    make_photo(path, orientation)
    photo = app.open_photo(path)
    assert photo.size == size
    assert photo.getpixel(red_at)[0] > 150


def test_normal_photo_is_not_changed(tmp_path):
    app = load_app()
    path = tmp_path / "normal.jpg"
    make_photo(path)
    photo = app.open_photo(path)
    assert photo.size == (40, 20)
    assert photo.getpixel((2, 2))[0] > 150
