"""Tests for ``data explore`` — unreadable images and cross-split leaks."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.data import explore
from cvbench.cli.generate import generate


def _save(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


@pytest.fixture
def clean_root(tmp_path) -> Path:
    root = tmp_path / "ds"
    seed = 0
    for split in ("train", "val"):
        for cls in ("cat", "dog"):
            seed += 1
            _save(root / split / cls / f"{seed}.png", seed)
    return root


def test_clean_dataset_exits_zero(clean_root):
    result = CliRunner().invoke(explore, [str(clean_root)])
    assert result.exit_code == 0, result.output
    assert "No unreadable images found" in result.output
    assert "No cross-split duplicates found" in result.output
    assert "No conflicting labels found" in result.output


def test_corrupt_images_listed_not_crash(clean_root):
    (clean_root / "train" / "cat" / "empty.jpg").write_bytes(b"")
    (clean_root / "train" / "dog" / "bad.jpg").write_bytes(b"not an image")
    result = CliRunner().invoke(explore, [str(clean_root)])
    assert result.exit_code == 1
    assert "2 unreadable image(s)" in result.output
    assert "empty file" in result.output
    assert "cannot decode" in result.output
    assert "data prep" in result.output
    assert "Brightness distribution" in result.output  # rest of the report still runs


def test_cross_split_leak_listed(clean_root):
    shutil.copyfile(clean_root / "train" / "cat" / "1.png", clean_root / "val" / "cat" / "copy.png")
    result = CliRunner().invoke(explore, [str(clean_root)])
    assert result.exit_code == 1
    assert "more than one split" in result.output
    assert "copy.png" in result.output
    assert "data prep" in result.output


def test_label_conflict_listed_including_within_split(clean_root):
    shutil.copyfile(clean_root / "train" / "cat" / "1.png", clean_root / "train" / "dog" / "copy.png")
    result = CliRunner().invoke(explore, [str(clean_root)])
    assert result.exit_code == 1
    assert "conflicting labels" in result.output
    assert "train/dog/copy.png" in result.output
    assert "more than one split" not in result.output  # a conflict is not double-listed as a leak


def test_yolo_label_conflict_listed(tmp_path):
    root = tmp_path / "yolo"
    arr = np.random.default_rng(2).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    for split, label in (("train", "0 0.5 0.5 0.2 0.2\n"), ("val", "0 0.1 0.1 0.2 0.2\n")):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.fromarray(arr).save(root / "images" / split / "a.png")
        (root / "labels" / split / "a.txt").write_text(label)
    (root / "data.yaml").write_text("names: [x]\n")
    result = CliRunner().invoke(explore, [str(root)])
    assert result.exit_code == 1
    assert "conflicting labels" in result.output


def test_yolo_dataset_reports_leak_and_corrupt(tmp_path):
    root = tmp_path / "yolo"
    result = CliRunner().invoke(
        generate,
        [str(root), "--format", "yolo", "--train", "2", "--val", "1", "--test", "0",
         "--image-size", "32", "--max-objects", "2"],
    )
    assert result.exit_code == 0, result.output
    clean = CliRunner().invoke(explore, [str(root)])
    assert clean.exit_code == 0, clean.output

    train_img = next((root / "images" / "train").iterdir())
    shutil.copyfile(train_img, root / "images" / "val" / "leak.png")
    (root / "images" / "train" / "empty.jpg").write_bytes(b"")
    bad = CliRunner().invoke(explore, [str(root)])
    assert bad.exit_code == 1
    assert "leak.png" in bad.output
    assert "empty file" in bad.output


def _write_truncated_bmp(path: Path) -> None:
    """A BMP PIL reads but TensorFlow's decoder rejects (size differs from its header)."""
    import io
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(99).integers(0, 255, (15, 15, 3), dtype=np.uint8)).save(buf, format="BMP")
    path.write_bytes(buf.getvalue()[:-2])


def test_tf_incompatible_image_flagged(clean_root):
    _write_truncated_bmp(clean_root / "train" / "cat" / "bmp_as.jpg")
    result = CliRunner().invoke(explore, [str(clean_root)])
    assert result.exit_code == 1
    assert "1 image(s) PIL reads but TensorFlow rejects" in result.output
    assert "bmp_as.jpg" in result.output
    assert "Input size should match" in result.output
    assert "No unreadable images found" in result.output  # it is not corrupt, just incompatible
