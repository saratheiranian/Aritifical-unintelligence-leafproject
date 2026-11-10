import numpy as np
import pytest
from PIL import Image

from src import field_test
from src.field_test import (
    list_images,
    normalize,
    resolve_mapping,
    restricted_predictions,
)

PV_NAMES = ["Apple___Apple_scab", "Corn_(maize)___Common_rust_", "Tomato___healthy", "Squash___Powdery_mildew"]


def test_normalize_ignores_case_and_punctuation():
    assert normalize("Corn_(maize)___Common_rust_") == "cornmaizecommonrust"
    assert normalize("Apple Scab Leaf") == normalize("apple_scab_leaf")


def test_resolve_mapping_matches_folders_and_reports_unmatched():
    folders = ["Apple Scab Leaf", "corn rust leaf", "Tomato leaf", "Mystery leaf", "Peach leaf"]
    mapping, unmatched = resolve_mapping(folders, PV_NAMES)
    assert mapping == {"Apple Scab Leaf": 0, "corn rust leaf": 1, "Tomato leaf": 2}
    # Mystery leaf is not in the table; Peach leaf is, but the model has no Peach class
    assert unmatched == ["Mystery leaf", "Peach leaf"]


def test_every_table_target_is_a_plantvillage_style_name():
    for target in field_test.PLANTDOC_TO_PLANTVILLAGE.values():
        assert "___" in target


def test_list_images_skips_non_images(tmp_path):
    (tmp_path / "Tomato leaf").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "Tomato leaf" / "a.jpg")
    (tmp_path / "Tomato leaf" / "notes.txt").write_text("x")
    items = list_images(tmp_path, {"Tomato leaf": 2})
    assert [(p.name, i) for p, i in items] == [("a.jpg", 2)]


def test_restricted_predictions_only_picks_allowed_classes():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_array_equal(restricted_predictions(probs, {1, 2}), [1, 2])


@pytest.fixture
def plantdoc_dir(tmp_path):
    """Fake PlantDoc test folder whose classes map to the conftest model classes 0 and 1."""
    root = tmp_path / "plantdoc"
    rng = np.random.default_rng(0)
    for folder in ["Leaf A", "Leaf B", "Unknown leaf"]:
        (root / folder).mkdir(parents=True)
        for i in range(3):
            pixels = rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / folder / f"{i}.png")
    return root


def test_main_reports_accuracies_and_writes_csv(checkpoint, plantdoc_dir, tmp_path, monkeypatch, capsys):
    table = {"Leaf A": "Plant0___Disease_0", "Leaf B": "Plant1___Disease_1"}
    monkeypatch.setattr(field_test, "PLANTDOC_TO_PLANTVILLAGE", table)
    out = tmp_path / "out"
    summary = field_test.main(
        ["--checkpoint", str(checkpoint), "--plantdoc-dir", str(plantdoc_dir), "--output-dir", str(out)]
    )
    assert summary["images"] == 6 and summary["classes"] == 2
    assert 0.0 <= summary["accuracy_all_classes"] <= summary["accuracy_shared_classes"] <= 1.0
    assert (out / "field_report.csv").exists() and (out / "field_summary.csv").exists()
    assert "Not used (no matching class): Unknown leaf" in capsys.readouterr().out


def test_main_stops_when_nothing_matches(checkpoint, plantdoc_dir, tmp_path):
    with pytest.raises(SystemExit):
        field_test.main(["--checkpoint", str(checkpoint), "--plantdoc-dir", str(plantdoc_dir),
                         "--output-dir", str(tmp_path / "out")])


def test_find_class_falls_back_to_ignoring_brackets():
    names = ["Corn___Common_rust", "Cherry___healthy", "Corn___healthy"]
    assert field_test.find_class("Corn_(maize)___Common_rust_", names) == 0
    assert field_test.find_class("Cherry_(including_sour)___healthy", names) == 1
    assert field_test.find_class("Corn_(maize)___Northern_Leaf_Blight", names) is None
