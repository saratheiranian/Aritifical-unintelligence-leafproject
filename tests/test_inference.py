import pytest
from PIL import Image

from src.inference import Predictor, pretty_name


def test_pretty_name():
    assert pretty_name("Tomato___Late_blight") == "Tomato: Late blight"
    assert pretty_name("Background_without_leaves") == "Background without leaves"


def test_top_k_is_sorted_probabilities(checkpoint):
    predictor = Predictor(checkpoint)
    top = predictor.top_k(Image.new("RGB", (256, 256), (90, 160, 60)), k=5)
    assert len(top) == 5
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs) and sum(probs) <= 1.0 + 1e-6
    assert all(name in predictor.class_names for name, _ in top)


def test_predict_proba_handles_any_size_and_mode(checkpoint):
    predictor = Predictor(checkpoint)
    images = [Image.new("RGB", (300, 200)), Image.new("L", (64, 64)), Image.new("RGBA", (128, 128))]
    probs = predictor.predict_proba(images)
    assert probs.shape == (3, 39)
    assert probs.sum(dim=1).sub(1).abs().max() < 1e-5


def test_app_builds(checkpoint):
    pytest.importorskip("gradio")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("app", Path(__file__).parent.parent / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    assert app.build_app({"cnn": str(checkpoint)}) is not None
