"""Tests for ``data prep`` — content-hash renaming and dedup in one pass."""
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.data import prep
from cvbench.cli.generate import generate
from cvbench.datasets import hashify as hashify_mod


def _all_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def yolo_root(tmp_path) -> Path:
    out = tmp_path / "yolo"
    result = CliRunner().invoke(
        generate,
        [str(out), "--format", "yolo", "--train", "3", "--val", "0", "--test", "0",
         "--image-size", "32", "--max-objects", "2"],
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


@pytest.fixture
def cls_root(tmp_path) -> Path:
    out = tmp_path / "cls"
    result = CliRunner().invoke(
        generate,
        [str(out), "--train", "2", "--val", "0", "--test", "0", "--image-size", "32"],
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


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


def test_prep_is_deterministic_and_idempotent(cls_root, tmp_path):
    dst1 = tmp_path / "dst1"
    dst2 = tmp_path / "dst2"
    r1 = CliRunner().invoke(prep, [str(cls_root), str(dst1)])
    r2 = CliRunner().invoke(prep, [str(cls_root), str(dst2)])
    assert r1.exit_code == 0 and r2.exit_code == 0, r1.output + r2.output

    names1 = {p.relative_to(dst1) for p in dst1.rglob("*") if p.is_file()}
    names2 = {p.relative_to(dst2) for p in dst2.rglob("*") if p.is_file()}
    assert names1 == names2


def test_prep_names_are_32_hex_chars(cls_root, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(prep, [str(cls_root), str(dst)])
    assert result.exit_code == 0, result.output
    for f in dst.rglob("*.jpg"):
        assert len(f.stem) == 32
        int(f.stem, 16)  # valid hex


def test_prep_drops_duplicate_no_suffix(cls_root_with_dupes, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(prep, [str(cls_root_with_dupes), str(dst)])
    assert result.exit_code == 0, result.output

    remaining = _all_files(dst)
    assert len(remaining) == 4  # train/cat (deduped to 1), train/dog, val/cat, val/dog
    for f in dst.rglob("*.jpg"):
        assert "-" not in f.stem
    assert "1 duplicate group" in result.output


def test_prep_yolo_renames_labels_in_lockstep(yolo_root, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(prep, [str(yolo_root), str(dst)])
    assert result.exit_code == 0, result.output

    assert (dst / "data.yaml").is_file()
    images = list((dst / "images").rglob("*.jpg"))
    assert images
    for img in images:
        rel = img.relative_to(dst / "images")
        label = dst / "labels" / rel.with_suffix(".txt")
        assert label.is_file(), f"missing label for {img}"
        assert label.stem == img.stem


def test_prep_yolo_dedup_drops_duplicate_label_pair(tmp_path):
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
    result = CliRunner().invoke(prep, [str(out), str(dst)])
    assert result.exit_code == 0, result.output

    dst_images = list((dst / "images" / "train").glob("*.jpg"))
    for dst_img in dst_images:
        dst_label = (dst / "labels" / "train" / dst_img.stem).with_suffix(".txt")
        assert dst_label.is_file() == label.is_file()
    assert not any(p.name == dup_img.name for p in dst_images)
    assert (dst / "data.yaml").is_file()


def test_prep_flat_dataset_duplicate_is_not_a_leak(tmp_path):
    """A duplicate group inside an unsplit flat pool has no split to leak across."""
    root = tmp_path / "pool"
    arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    (root / "cat").mkdir(parents=True)
    Image.fromarray(arr).save(root / "cat" / "a.jpg", quality=100)
    Image.fromarray(arr).save(root / "cat" / "b.jpg", quality=100)

    dst = tmp_path / "dst"
    result = CliRunner().invoke(prep, [str(root), str(dst), "--across-splits"])
    assert result.exit_code == 0, result.output
    assert "1 duplicate group" in result.output
    assert "No cross-split leakage detected" in result.output


def test_prep_no_across_splits_by_default(cls_root_with_leak, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(prep, [str(cls_root_with_leak), str(dst)])
    assert result.exit_code == 0, result.output
    assert "leak" not in result.output.lower()


def test_prep_across_splits_flags_leak(cls_root_with_leak, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(prep, [str(cls_root_with_leak), str(dst), "--across-splits"])
    assert result.exit_code == 0, result.output
    assert "leak" in result.output.lower()


def test_prep_dry_run_writes_nothing(cls_root_with_dupes, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(prep, [str(cls_root_with_dupes), str(dst), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not dst.exists()


def test_prep_src_untouched(cls_root_with_dupes, tmp_path):
    before = _all_files(cls_root_with_dupes)
    dst = tmp_path / "dst"
    CliRunner().invoke(prep, [str(cls_root_with_dupes), str(dst)])
    assert _all_files(cls_root_with_dupes) == before


def test_hash_image_file_is_full_md5(cls_root):
    img = next(cls_root.rglob("*.jpg"))
    full = hashify_mod.hash_image_file(img)
    assert len(full) == 32
    int(full, 16)  # valid hex
