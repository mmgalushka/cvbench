"""Tests for ``data clean`` — copy a dataset, dropping OS/editor junk."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.data import clean
from cvbench.cli.generate import generate


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


def _seed_junk(root: Path) -> None:
    """Drop a handful of junk files/dirs under ROOT."""
    (root / ".DS_Store").write_bytes(b"")
    (root / "Thumbs.db").write_bytes(b"")
    macosx = root / "__MACOSX"
    macosx.mkdir()
    (macosx / "._foo.jpg").write_bytes(b"")
    (root / "._bar.jpg").write_bytes(b"")
    (root / "notes.txt~").write_bytes(b"")


def _all_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_clean_drops_junk_keeps_real_files(cls_root, tmp_path):
    real_files_before = _all_files(cls_root)
    _seed_junk(cls_root)

    dst = tmp_path / "cls_clean"
    result = CliRunner().invoke(clean, [str(cls_root), str(dst)])
    assert result.exit_code == 0, result.output

    assert _all_files(dst) == real_files_before
    assert not (dst / "__MACOSX").exists()


def test_clean_yolo_layout_preserves_labels(yolo_root, tmp_path):
    _seed_junk(yolo_root)
    dst = tmp_path / "yolo_clean"
    result = CliRunner().invoke(clean, [str(yolo_root), str(dst)])
    assert result.exit_code == 0, result.output

    assert (dst / "data.yaml").is_file()
    src_images = list((yolo_root / "images").rglob("*.jpg"))
    dst_images = list((dst / "images").rglob("*.jpg"))
    assert len(dst_images) == len(src_images)
    for img in dst_images:
        rel = img.relative_to(dst / "images")
        assert (dst / "labels" / rel.with_suffix(".txt")).is_file()


def test_clean_dry_run_writes_nothing(cls_root, tmp_path):
    _seed_junk(cls_root)
    dst = tmp_path / "cls_clean"
    result = CliRunner().invoke(clean, [str(cls_root), str(dst), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not dst.exists()


def test_clean_src_untouched(cls_root, tmp_path):
    _seed_junk(cls_root)
    before = _all_files(cls_root)
    dst = tmp_path / "cls_clean"
    CliRunner().invoke(clean, [str(cls_root), str(dst)])
    assert _all_files(cls_root) == before


def test_clean_rejects_nonempty_dst(cls_root, tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("x")
    result = CliRunner().invoke(clean, [str(cls_root), str(dst)])
    assert result.exit_code != 0
    assert "already contains" in result.output
