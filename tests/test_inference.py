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


def test_app_builds_with_captioned_examples(checkpoint, tmp_path):
    pytest.importorskip("gradio")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("app", Path(__file__).parent.parent / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    examples = tmp_path / "examples"
    examples.mkdir()
    Image.new("RGB", (64, 64), (40, 120, 40)).save(examples / "Cherry_(including_sour)___healthy.jpg")
    app.EXAMPLES_DIR = examples
    assert app.build_app({"cnn": str(checkpoint)}) is not None


def load_app_with_fake_hub(monkeypatch, tmp_path, list_repo_files, repo="user/plant-model", snapshot_download=None):
    """Load app.py with a pretend huggingface_hub, so the tests need no internet."""
    pytest.importorskip("gradio")
    import importlib.util
    import sys
    import types
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("app", Path(__file__).parent.parent / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    downloads = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.list_repo_files = list_repo_files
    fake_hub.hf_hub_download = lambda repo, filename: f"/cache/{filename}"
    fake_hub.snapshot_download = snapshot_download or (lambda repo, **kwargs: downloads.append((repo, kwargs)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HF_MODEL_REPO", repo)
    monkeypatch.setattr(app, "HERE", tmp_path)  # no local models/ folder
    return app, downloads


def test_hf_repo_also_downloads_the_example_images(monkeypatch, tmp_path):
    files = ["README.md", "resnet18.pt", "examples/Apple___healthy.jpg"]
    app, downloads = load_app_with_fake_hub(monkeypatch, tmp_path, lambda repo: files)

    assert app.find_checkpoints(None) == {"resnet18": "/cache/resnet18.pt"}
    assert downloads == [
        ("user/plant-model", {"allow_patterns": ["examples/*", "class_examples/*"], "local_dir": tmp_path}),
    ]


def test_app_still_starts_if_the_example_download_fails(monkeypatch, tmp_path, capsys):
    def failing_download(repo, **kwargs):
        raise OSError("rate limit")

    app, _ = load_app_with_fake_hub(monkeypatch, tmp_path, lambda repo: ["resnet18.pt"],
                                    snapshot_download=failing_download)

    assert app.find_checkpoints(None) == {"resnet18": "/cache/resnet18.pt"}
    assert "could not download the example images" in capsys.readouterr().out


def test_wrong_repo_name_gives_one_clear_message(monkeypatch, tmp_path, capsys):
    def repo_not_found(repo):
        raise OSError("404 Repository Not Found")

    app, downloads = load_app_with_fake_hub(monkeypatch, tmp_path, repo_not_found, repo="user/plant-model-typo")
    with pytest.raises(SystemExit, match="Could not read the Hugging Face repo 'user/plant-model-typo'") as stop:
        app.find_checkpoints(None)
    assert "404 Repository Not Found" in str(stop.value)  # the original error is shown too
    assert "HF_TOKEN" in str(stop.value)  # private repos work with a token
    assert downloads == []  # the pictures are not tried, so there is no misleading gallery warning
    assert "galleries" not in capsys.readouterr().out


def test_repo_without_models_gives_a_clear_message(monkeypatch, tmp_path):
    app, downloads = load_app_with_fake_hub(monkeypatch, tmp_path, lambda repo: ["README.md", "examples/a.jpg"],
                                            repo="user/repo-without-models")
    with pytest.raises(SystemExit, match="'user/repo-without-models' has no .pt model files"):
        app.find_checkpoints(None)
    assert downloads == []
