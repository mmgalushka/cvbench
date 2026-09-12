"""Tests for ``data hashify`` — content-hash renaming."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.data import hashify
from cvbench.cli.generate import generate
from cvbench.datasets import hashify as hashify_mod


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


def test_hashify_is_deterministic_and_idempotent(cls_root, tmp_path):
    dst1 = tmp_path / "dst1"
    dst2 = tmp_path / "dst2"
    r1 = CliRunner().invoke(hashify, [str(cls_root), str(dst1)])
    r2 = CliRunner().invoke(hashify, [str(cls_root), str(dst2)])
    assert r1.exit_code == 0 and r2.exit_code == 0, r1.output + r2.output

    names1 = {p.relative_to(dst1) for p in dst1.rglob("*") if p.is_file()}
    names2 = {p.relative_to(dst2) for p in dst2.rglob("*") if p.is_file()}
    assert names1 == names2


def test_hashify_names_are_16_hex_chars(cls_root, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(hashify, [str(cls_root), str(dst)])
    assert result.exit_code == 0, result.output
    for f in dst.rglob("*.jpg"):
        assert len(f.stem) == 16
        int(f.stem, 16)  # valid hex


def test_hashify_collision_gets_numeric_suffix(tmp_path):
    import numpy as np
    from PIL import Image

    src = tmp_path / "src" / "dog"
    src.mkdir(parents=True)
    arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(src / "a.jpg", quality=100)
    Image.fromarray(arr).save(src / "b.jpg", quality=100)  # byte-identical after decode

    dst = tmp_path / "dst"
    result = CliRunner().invoke(hashify, [str(tmp_path / "src"), str(dst)])
    assert result.exit_code == 0, result.output

    files = sorted((dst / "dog").iterdir())
    assert len(files) == 2
    assert "collision" in result.output.lower()


def test_hashify_yolo_renames_labels_in_lockstep(yolo_root, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(hashify, [str(yolo_root), str(dst)])
    assert result.exit_code == 0, result.output

    assert (dst / "data.yaml").is_file()
    images = list((dst / "images").rglob("*.jpg"))
    assert images
    for img in images:
        rel = img.relative_to(dst / "images")
        label = dst / "labels" / rel.with_suffix(".txt")
        assert label.is_file(), f"missing label for {img}"
        assert label.stem == img.stem


def test_hashify_dry_run_writes_nothing(cls_root, tmp_path):
    dst = tmp_path / "dst"
    result = CliRunner().invoke(hashify, [str(cls_root), str(dst), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not dst.exists()


def test_hashify_src_untouched(cls_root, tmp_path):
    before = {p.relative_to(cls_root) for p in cls_root.rglob("*") if p.is_file()}
    dst = tmp_path / "dst"
    CliRunner().invoke(hashify, [str(cls_root), str(dst)])
    after = {p.relative_to(cls_root) for p in cls_root.rglob("*") if p.is_file()}
    assert before == after


def test_hash_array_matches_content_hash_prefix(cls_root):
    img = next(cls_root.rglob("*.jpg"))
    import numpy as np
    from PIL import Image
    full = hashify_mod.hash_array(np.array(Image.open(img).convert("RGB")))
    assert hashify_mod.content_hash(img) == full[:16]
