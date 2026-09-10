"""Tests for ``data dedup`` — exact-duplicate detection and removal."""
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.data import dedup


def _all_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def cls_root_with_dupes(tmp_path) -> Path:
    """cat/dog classification dataset with a same-content duplicate in 'cat'."""
    root = tmp_path / "cls"
    rng = np.random.default_rng(0)
    for i, (split, cls) in enumerate([("train", "cat"), ("train", "dog"), ("val", "cat"), ("val", "dog")]):
        d = root / split / cls
        d.mkdir(parents=True)
        arr = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) + i  # distinct content per split/class
        Image.fromarray(arr).save(d / "a.jpg", quality=100)
    # duplicate of train/cat/a.jpg, different name, same split (byte-identical file
    # copy, not a JPEG re-encode, since re-compressing isn't guaranteed lossless)
    import shutil
    shutil.copyfile(root / "train" / "cat" / "a.jpg", root / "train" / "cat" / "b.jpg")
    return root


@pytest.fixture
def cls_root_with_leak(tmp_path) -> Path:
    """Same image present in both train and val -> cross-split leak."""
    root = tmp_path / "cls"
    arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    (root / "train" / "cat").mkdir(parents=True)
    (root / "val" / "cat").mkdir(parents=True)
    Image.fromarray(arr).save(root / "train" / "cat" / "a.jpg", quality=100)
    Image.fromarray(arr).save(root / "val" / "cat" / "a.jpg", quality=100)
    return root


def test_dedup_drops_duplicate_keeps_unique(cls_root_with_dupes, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(dedup, [str(cls_root_with_dupes), str(dst)])
    assert result.exit_code == 0, result.output

    remaining = _all_files(dst)
    assert "train/cat/a.jpg" in remaining
    assert "train/cat/b.jpg" not in remaining  # duplicate dropped
    assert "train/dog/a.jpg" in remaining
    assert "val/cat/a.jpg" in remaining
    assert "val/dog/a.jpg" in remaining
    assert "1 duplicate group" in result.output


def test_dedup_no_across_splits_by_default(cls_root_with_leak, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(dedup, [str(cls_root_with_leak), str(dst)])
    assert result.exit_code == 0, result.output
    assert "leak" not in result.output.lower()


def test_dedup_across_splits_flags_leak(cls_root_with_leak, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(dedup, [str(cls_root_with_leak), str(dst), "--across-splits"])
    assert result.exit_code == 0, result.output
    assert "leak" in result.output.lower()


def test_dedup_dry_run_writes_nothing(cls_root_with_dupes, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(dedup, [str(cls_root_with_dupes), str(dst), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not dst.exists()


def test_dedup_src_untouched(cls_root_with_dupes, tmp_path):
    before = _all_files(cls_root_with_dupes)
    dst = tmp_path / "dst"
    CliRunner().invoke(dedup, [str(cls_root_with_dupes), str(dst)])
    assert _all_files(cls_root_with_dupes) == before


def test_dedup_yolo_keeps_label_pairs(tmp_path):
    from cvbench.cli.generate import generate

    out = tmp_path / "yolo"
    result = CliRunner().invoke(
        generate,
        [str(out), "--format", "yolo", "--train", "3", "--val", "0", "--test", "0",
         "--image-size", "32", "--max-objects", "2"],
    )
    assert result.exit_code == 0, result.output

    # duplicate one train image under a new name (with its own label copy)
    img = next((out / "images" / "train").glob("*.jpg"))
    label = (out / "labels" / "train" / img.stem).with_suffix(".txt")
    dup_img = img.with_name("dup_" + img.name)
    dup_img.write_bytes(img.read_bytes())
    if label.is_file():
        (out / "labels" / "train" / dup_img.stem).with_suffix(".txt").write_text(label.read_text())

    dst = tmp_path / "yolo_dst"
    result = CliRunner().invoke(dedup, [str(out), str(dst)])
    assert result.exit_code == 0, result.output

    dst_images = list((dst / "images" / "train").glob("*.jpg"))
    for dst_img in dst_images:
        dst_label = (dst / "labels" / "train" / dst_img.stem).with_suffix(".txt")
        assert dst_label.is_file() == label.is_file()
    assert not any(p.name == dup_img.name for p in dst_images)
    assert (dst / "data.yaml").is_file()
