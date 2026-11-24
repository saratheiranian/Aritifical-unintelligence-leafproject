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


def test_phone_photo_is_turned_the_right_way_up(tmp_path):
    app = load_app()
    exif = Image.Exif()
    exif[0x0112] = 6  # the phone says "rotate 90 degrees"
    path = tmp_path / "phone.jpg"
    Image.new("RGB", (40, 20)).save(path, exif=exif)
    assert app.open_photo(path).size == (20, 40)


def test_normal_photo_is_not_changed(tmp_path):
    app = load_app()
    path = tmp_path / "normal.jpg"
    Image.new("RGB", (40, 20)).save(path)
    assert app.open_photo(path).size == (40, 20)