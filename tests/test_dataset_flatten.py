"""Tests for ``data flatten`` — pool an already-split dataset back to flat."""
from pathlib import Path

import numpy as np
import yaml
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.data import flatten
from cvbench.cli.generate import generate


def _make_split_classification(root: Path, classes: list[str], per_class: int = 5) -> Path:
    for split in ["train", "val"]:
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(per_class):
                arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i:03d}.jpg", quality=90)
    return root


def _make_flat_classification(root: Path, per_class: int = 5) -> Path:
    for cls in ["cat", "dog"]:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(per_class):
            arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i:03d}.jpg", quality=90)
    return root


def test_flatten_pools_classification_splits(tmp_path):
    src = _make_split_classification(tmp_path / "src", ["cat", "dog"], per_class=7)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(flatten, [str(src), str(dst)])
    assert result.exit_code == 0, result.output

    for cls in ["cat", "dog"]:
        assert len(list((dst / cls).glob("*.jpg"))) == 14  # train + val, 7 each
    assert not (dst / "train").exists()
    assert not (dst / "val").exists()


def test_flatten_dry_run_writes_nothing(tmp_path):
    src = _make_split_classification(tmp_path / "src", ["cat"], per_class=3)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(flatten, [str(src), str(dst), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not dst.exists()


def test_flatten_src_untouched(tmp_path):
    src = _make_split_classification(tmp_path / "src", ["cat"], per_class=3)
    before = {p.relative_to(src) for p in src.rglob("*.jpg")}
    dst = tmp_path / "dst"
    CliRunner().invoke(flatten, [str(src), str(dst)])
    assert {p.relative_to(src) for p in src.rglob("*.jpg")} == before


def test_flatten_rejects_already_flat_dataset(tmp_path):
    src = _make_flat_classification(tmp_path / "src", per_class=3)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(flatten, [str(src), str(dst)])
    assert result.exit_code != 0
    assert "already flat" in result.output
    assert not dst.exists()


def test_flatten_rejects_dst_with_existing_files(tmp_path):
    src = _make_split_classification(tmp_path / "src", ["cat"], per_class=2)
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "existing.txt").write_text("x")
    result = CliRunner().invoke(flatten, [str(src), str(dst)])
    assert result.exit_code != 0
    assert "already contains files" in result.output


def test_flatten_yolo_pools_images_and_labels(tmp_path):
    src = tmp_path / "yolo_src"
    result = CliRunner().invoke(
        generate,
        [str(src), "--format", "yolo", "--train", "6", "--val", "4", "--test", "0",
         "--image-size", "48", "--max-objects", "2"],
    )
    assert result.exit_code == 0, result.output

    dst = tmp_path / "yolo_dst"
    result = CliRunner().invoke(flatten, [str(src), str(dst)])
    assert result.exit_code == 0, result.output

    assert (dst / "data.yaml").is_file()
    data_yaml = yaml.safe_load((dst / "data.yaml").read_text())
    assert "images" in data_yaml
    assert not any(k in data_yaml for k in ("train", "val", "test"))

    images = list((dst / "images").glob("*.jpg"))
    assert len(images) == 10  # 6 + 4, no split subdirs
    for img in images:
        label = dst / "labels" / f"{img.stem}.txt"
        assert label.is_file()
