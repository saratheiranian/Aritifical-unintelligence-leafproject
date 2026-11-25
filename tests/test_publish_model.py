import sys
import types

import pytest

from src import publish_model
from src.publish_model import example_folders, upload_examples


class FakeApi:
    def __init__(self):
        self.uploads = []

    def upload_folder(self, folder_path, path_in_repo, repo_id, **kwargs):
        self.uploads.append((path_in_repo, repo_id, kwargs))


def make_images(app_dir, *folders):
    for name in folders:
        (app_dir / name / "Apple___healthy").mkdir(parents=True)
        (app_dir / name / "Apple___healthy" / "0.jpg").write_bytes(b"jpg")


def test_example_folders_finds_both_folders(tmp_path):
    make_images(tmp_path, "examples", "class_examples")
    assert example_folders(tmp_path) == [tmp_path / "examples", tmp_path / "class_examples"]


def test_example_folders_stops_if_a_folder_is_missing(tmp_path):
    make_images(tmp_path, "examples")
    with pytest.raises(SystemExit, match="make_app_assets"):
        example_folders(tmp_path)


def test_example_folders_stops_if_a_folder_is_empty(tmp_path):
    make_images(tmp_path, "examples")
    (tmp_path / "class_examples").mkdir()
    with pytest.raises(SystemExit, match="class_examples"):
        example_folders(tmp_path)


def test_upload_examples_sends_only_jpgs_and_replaces_old_images(tmp_path):
    make_images(tmp_path, "examples", "class_examples")
    api = FakeApi()
    upload_examples(api, "user/plant-model", example_folders(tmp_path))
    assert [(name, repo) for name, repo, _ in api.uploads] == [
        ("examples", "user/plant-model"),
        ("class_examples", "user/plant-model"),
    ]
    for _, _, kwargs in api.uploads:
        assert kwargs["allow_patterns"] == "*.jpg"
        assert kwargs["delete_patterns"] == "*"


def test_main_checks_the_images_before_uploading_anything(monkeypatch, tmp_path):
    created = []
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = lambda: created.append("api")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(publish_model, "APP_DIR", tmp_path)  # no images here
    with pytest.raises(SystemExit, match="make_app_assets"):
        publish_model.main(["--repo-id", "user/plant-model", "--checkpoint", "missing.pt", "--with-examples"])
    assert created == []  # stopped before the model or README were uploaded
