"""Tests for ``data list`` and ``datasets.stats.get_dataset_overview``."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.data import data
from cvbench.cli.generate import generate
from cvbench.datasets.shapes import CLASSES
from cvbench.datasets.stats import get_dataset_overview


@pytest.fixture
def yolo_root(tmp_path) -> Path:
    out = tmp_path / "data" / "synth_yolo"
    result = CliRunner().invoke(
        generate,
        [str(out), "--format", "yolo", "--train", "5", "--val", "2", "--test", "0",
         "--image-size", "64", "--max-objects", "3"],
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


@pytest.fixture
def cls_root(tmp_path) -> Path:
    out = tmp_path / "data" / "synth_cls"
    result = CliRunner().invoke(
        generate,
        [str(out), "--train", "4", "--val", "2", "--test", "0", "--image-size", "32"],
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


def test_get_dataset_overview_classification(cls_root):
    overview = get_dataset_overview(cls_root)
    assert set(overview) == {"train", "val"}
    assert overview["train"]["classes"] == len(CLASSES)
    assert overview["train"]["images"] == 4 * len(CLASSES)
    assert overview["val"]["images"] == 2 * len(CLASSES)


def test_get_dataset_overview_yolo(yolo_root):
    overview = get_dataset_overview(yolo_root)
    assert set(overview) == {"train", "val"}
    assert overview["train"]["classes"] == len(CLASSES)
    assert overview["train"]["images"] == 5
    assert overview["val"]["images"] == 2


def test_get_dataset_overview_missing_splits(tmp_path):
    (tmp_path / "empty").mkdir()
    assert get_dataset_overview(tmp_path / "empty") == {}


def test_data_list_empty(tmp_path):
    result = CliRunner().invoke(data, ["list", str(tmp_path)])
    assert result.exit_code == 0
    assert "No datasets found" in result.output


def test_data_list_shows_classification_dataset(cls_root):
    result = CliRunner().invoke(data, ["list", str(cls_root.parent)])
    assert result.exit_code == 0, result.output
    assert "synth_cls" in result.output
    assert "cls" in result.output


def test_data_list_shows_yolo_dataset(yolo_root):
    result = CliRunner().invoke(data, ["list", str(yolo_root.parent)])
    assert result.exit_code == 0, result.output
    assert "synth_yolo" in result.output
    assert "det" in result.output


def test_data_list_default_data_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(data, ["list"])
    assert result.exit_code == 0
    assert "No datasets found in 'data'" in result.output
