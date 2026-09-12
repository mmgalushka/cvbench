"""Synthetic dataset generation tests — classification and YOLO layouts."""
import random
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.generate import generate
from cvbench.datasets.shapes import CLASSES, random_shape, shape_bbox
from cvbench.datasets.synth import generate_detection_image, to_yolo_line

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", CLASSES)
def test_shape_bbox_within_image(cls):
    rng = random.Random(0)
    for _ in range(50):
        bbox = shape_bbox(random_shape(cls, 64, rng), 64)
        x1, y1, x2, y2 = bbox
        assert 0 <= x1 < x2 <= 64
        assert 0 <= y1 < y2 <= 64


def test_detection_image_has_boxes():
    rng = random.Random(1)
    img, boxes = generate_detection_image(96, rng, max_objects=4)
    assert img.size == (96, 96)
    assert 1 <= len(boxes) <= 4
    assert all(cls in CLASSES for cls, _ in boxes)


def test_to_yolo_line_normalizes_to_centre_format():
    line = to_yolo_line("square", (20.0, 40.0, 60.0, 80.0), 100)
    cls_id, xc, yc, w, h = line.split()
    assert int(cls_id) == CLASSES.index("square")
    assert (float(xc), float(yc)) == (0.4, 0.6)
    assert (float(w), float(h)) == (0.4, 0.4)


# ---------------------------------------------------------------------------
# CLI — classification format
# ---------------------------------------------------------------------------

def test_generate_classification_layout(tmp_path):
    out = tmp_path / "cls"
    result = CliRunner().invoke(
        generate,
        [str(out), "--train", "2", "--val", "1", "--test", "1", "--image-size", "32"],
    )
    assert result.exit_code == 0, result.output

    for split, n in [("train", 2), ("val", 1), ("test", 1)]:
        for cls in CLASSES:
            images = sorted((out / split / cls).glob("*.jpg"))
            assert len(images) == n


def test_generate_refuses_existing_dir_without_overwrite(tmp_path):
    out = tmp_path / "cls"
    out.mkdir()
    result = CliRunner().invoke(generate, [str(out), "--train", "1", "--val", "1", "--test", "1"])
    assert result.exit_code != 0
    assert "--overwrite" in result.output


# ---------------------------------------------------------------------------
# CLI — YOLO format
# ---------------------------------------------------------------------------

@pytest.fixture
def yolo_dataset(tmp_path) -> Path:
    out = tmp_path / "yolo"
    result = CliRunner().invoke(
        generate,
        [str(out), "--format", "yolo", "--train", "4", "--val", "2", "--test", "0",
         "--image-size", "64", "--max-objects", "3"],
    )
    assert result.exit_code == 0, result.output
    return out


def test_yolo_layout(yolo_dataset):
    assert len(list((yolo_dataset / "images" / "train").glob("*.jpg"))) == 4
    assert len(list((yolo_dataset / "labels" / "train").glob("*.txt"))) == 4
    assert len(list((yolo_dataset / "images" / "val").glob("*.jpg"))) == 2
    # test split was requested with 0 images
    assert not (yolo_dataset / "images" / "test").exists()


def test_yolo_labels_are_valid(yolo_dataset):
    for label_path in (yolo_dataset / "labels").rglob("*.txt"):
        lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
        assert 1 <= len(lines) <= 3
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            cls_id = int(parts[0])
            assert 0 <= cls_id < len(CLASSES)
            xc, yc, w, h = (float(p) for p in parts[1:])
            assert 0 < w <= 1 and 0 < h <= 1
            assert xc - w / 2 >= 0 and xc + w / 2 <= 1.0001
            assert yc - h / 2 >= 0 and yc + h / 2 <= 1.0001


def test_yolo_data_yaml(yolo_dataset):
    import yaml

    cfg = yaml.safe_load((yolo_dataset / "data.yaml").read_text())
    assert cfg["names"] == dict(enumerate(CLASSES))
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert "test" not in cfg


def test_generate_rejects_zero_max_objects(tmp_path):
    result = CliRunner().invoke(
        generate, [str(tmp_path / "yolo"), "--format", "yolo", "--max-objects", "0"]
    )
    assert result.exit_code != 0
    assert "--max-objects" in result.output


def test_generate_is_reproducible(tmp_path):
    runs = []
    for name in ("a", "b"):
        out = tmp_path / name
        CliRunner().invoke(
            generate,
            [str(out), "--format", "yolo", "--train", "3", "--val", "0", "--test", "0",
             "--image-size", "48", "--seed", "7"],
        )
        runs.append(sorted(p.read_text() for p in (out / "labels" / "train").glob("*.txt")))
    assert runs[0] == runs[1]
