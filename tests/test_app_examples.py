import pytest
from PIL import Image

from tests.test_app_photos import load_app

EXAMPLE_LABEL = "Or pick an example (caption = true class)"


def build_with_examples(tmp_path, monkeypatch, checkpoint):
    """Build the app with two example files: a green apple leaf and a red tomato leaf."""
    gr = pytest.importorskip("gradio")
    app = load_app()
    Image.new("RGB", (8, 8), "red").save(tmp_path / "Tomato___Late_blight.jpg")
    Image.new("RGB", (8, 8), "green").save(tmp_path / "Apple___Apple_scab.jpg")
    monkeypatch.setattr(app, "EXAMPLES_DIR", tmp_path)
    demo = app.build_app({"cnn": str(checkpoint)})
    galleries = [b for b in demo.blocks.values() if isinstance(b, gr.Gallery) and b.label == EXAMPLE_LABEL]
    assert len(galleries) == 1
    return gr, demo, galleries[0]


def test_example_gallery_shows_each_picture_with_its_true_class(tmp_path, monkeypatch, checkpoint):
    """Issue #57: the captions under the example pictures were missing with Gradio 6."""
    _, _, gallery = build_with_examples(tmp_path, monkeypatch, checkpoint)
    # files are sorted by name, so Apple comes first
    assert [item["image"]["orig_name"] for item in gallery.value] == [
        "Apple___Apple_scab.jpg", "Tomato___Late_blight.jpg"]
    assert [item["caption"] for item in gallery.value] == ["Apple: Apple scab", "Tomato: Late blight"]


def test_clicking_an_example_loads_that_picture(tmp_path, monkeypatch, checkpoint):
    gr, demo, _ = build_with_examples(tmp_path, monkeypatch, checkpoint)
    fns = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
    pick_example = next(f.fn for f in fns if getattr(f.fn, "__name__", "") == "pick_example")

    photo = pick_example(gr.SelectData(None, {"index": 1, "value": None}))  # the second picture

    assert photo.getpixel((4, 4))[0] > 150  # red, so the tomato leaf
