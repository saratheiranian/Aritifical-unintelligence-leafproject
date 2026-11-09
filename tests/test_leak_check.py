import math

import pandas as pd
import pytest

from src.leak_check import leak_report, leaked_test_positions


def duplicates_table():
    # group 0: train + test copy (leak); group 1: two test copies (no leak); group 2: val + test (leak)
    return pd.DataFrame(
        {
            "group": [0, 0, 1, 1, 2, 2],
            "row": [10, 20, 30, 31, 40, 50],
            "split": ["train", "test", "test", "test", "val", "test"],
        }
    )


def test_only_test_images_with_a_copy_in_another_split_are_leaked():
    test_rows = [99, 20, 30, 31, 50]  # dataset rows of the test images, in test order
    assert leaked_test_positions(duplicates_table(), test_rows) == [1, 4]


def test_no_duplicates_means_no_leaked_images():
    empty = pd.DataFrame(columns=["group", "row", "split"])
    assert leaked_test_positions(empty, [1, 2, 3]) == []


def test_leak_report_splits_accuracy_into_leaked_and_clean():
    preds = [0, 1, 1, 0]
    labels = [0, 1, 0, 0]  # third prediction is wrong
    report = leak_report(preds, labels, leaked_positions=[0, 1])
    assert report["test_images"] == 4 and report["leaked_images"] == 2
    assert report["accuracy_all"] == pytest.approx(0.75)
    assert report["accuracy_leaked"] == pytest.approx(1.0)
    assert report["accuracy_clean"] == pytest.approx(0.5)


def test_leak_report_without_leaked_images_gives_nan_for_that_group():
    report = leak_report([1, 1], [1, 0], leaked_positions=[])
    assert math.isnan(report["accuracy_leaked"])
    assert report["accuracy_clean"] == report["accuracy_all"] == pytest.approx(0.5)
