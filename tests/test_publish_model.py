import pytest

from src.publish_model import upload_examples


class FakeApi:
    def __init__(self):
        self.uploads = []

    def upload_folder(self, folder_path, path_in_repo, repo_id):
        self.uploads.append((path_in_repo, repo_id))


def test_upload_examples_sends_both_folders(tmp_path):
    (tmp_path / "examples").mkdir()
    (tmp_path / "class_examples").mkdir()
    api = FakeApi()
    upload_examples(api, "user/plant-model", app_dir=tmp_path)
    assert api.uploads == [("examples", "user/plant-model"), ("class_examples", "user/plant-model")]


def test_upload_examples_stops_if_the_images_are_missing(tmp_path):
    (tmp_path / "examples").mkdir()
    api = FakeApi()
    with pytest.raises(SystemExit, match="make_app_assets"):
        upload_examples(api, "user/plant-model", app_dir=tmp_path)
    assert api.uploads == []