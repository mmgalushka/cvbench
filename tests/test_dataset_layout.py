"""Tests for cvbench.datasets.layout — YOLO dataset sniffing and parsing."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.generate import generate
from cvbench.datasets import layout
from cvbench.datasets.shapes import CLASSES


@pytest.fixture
def yolo_root(tmp_path) -> Path:
    out = tmp_path / "yolo"
    result = CliRunner().invoke(
        generate,
        [str(out), "--format", "yolo", "--train", "5", "--val", "2", "--test", "0",
         "--image-size", "64", "--max-objects", "3"],
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


@pytest.fixture
def cls_root(tmp_path) -> Path:
    out = tmp_path / "cls"
    result = CliRunner().invoke(
        generate,
        [str(out), "--train", "2", "--val", "1", "--test", "0", "--image-size", "32"],
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


def test_is_yolo_dataset(yolo_root, cls_root):
    assert layout.is_yolo_dataset(yolo_root)
    assert not layout.is_yolo_dataset(cls_root)


def test_detect_task_name(yolo_root, cls_root):
    assert layout.detect_task_name(yolo_root) == "detection"
    assert layout.detect_task_name(cls_root) == "classification"


def test_yolo_root_walks_up_from_split_dir(yolo_root):
    split_dir = yolo_root / "images" / "train"
    assert layout.yolo_root(split_dir) == yolo_root
    assert layout.yolo_root(yolo_root) is None  # not itself a split dir


def test_yolo_class_names_from_data_yaml(yolo_root):
    assert layout.yolo_class_names(yolo_root) == CLASSES


def test_class_names_fall_back_to_label_ids(yolo_root):
    (yolo_root / "data.yaml").unlink()
    names = layout.yolo_class_names(yolo_root)
    assert names and all(n.isdigit() for n in names)


def test_read_yolo_boxes_normalizes_and_clamps(tmp_path):
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.6 0.6\n1 0.05 0.05 0.2 0.2\n")
    boxes = layout.read_yolo_boxes(label)
    assert boxes[0] == (0, (0.2, 0.2, 0.6, 0.6))
    cls_id, (x, y, w, h) = boxes[1]
    assert cls_id == 1
    assert x >= 0.0 and y >= 0.0
    assert x + w <= 1.0001 and y + h <= 1.0001


def test_read_yolo_boxes_missing_file(tmp_path):
    assert layout.read_yolo_boxes(tmp_path / "missing.txt") == []


def test_list_images_sorted_and_filtered(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"")
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    names = [p.name for p in layout.list_images(tmp_path)]
    assert names == ["a.png", "b.jpg"]
