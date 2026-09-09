"""Tests for ``data split`` — stratified train/val/test partitioning."""
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.data import split
from cvbench.cli.generate import generate


def _make_flat_classification(root: Path, per_class: int = 20) -> Path:
    for cls in ["cat", "dog"]:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(per_class):
            arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i:03d}.jpg", quality=90)
    return root


def test_split_stratifies_classification_pool(tmp_path):
    src = _make_flat_classification(tmp_path / "src", per_class=20)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(split, [str(src), str(dst), "--train", "0.8", "--val", "0.1", "--test", "0.1"])
    assert result.exit_code == 0, result.output

    for cls in ["cat", "dog"]:
        n_train = len(list((dst / "train" / cls).glob("*.jpg")))
        n_val = len(list((dst / "val" / cls).glob("*.jpg")))
        n_test = len(list((dst / "test" / cls).glob("*.jpg")))
        assert n_train + n_val + n_test == 20
        assert n_train == pytest.approx(16, abs=1)
        assert n_val == pytest.approx(2, abs=1)
        assert n_test == pytest.approx(2, abs=1)


def test_split_deterministic_given_seed(tmp_path):
    src = _make_flat_classification(tmp_path / "src", per_class=10)

    def run(dst_name):
        dst = tmp_path / dst_name
        r = CliRunner().invoke(split, [str(src), str(dst), "--seed", "7"])
        assert r.exit_code == 0, r.output
        return {p.relative_to(dst) for p in dst.rglob("*.jpg")}

    assert run("dst1") == run("dst2")


def test_split_resplits_already_split_dataset(tmp_path):
    src = tmp_path / "src"
    for split_name in ["train", "val"]:
        _make_flat_classification(src / split_name, per_class=10)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(split, [str(src), str(dst), "--train", "0.5", "--val", "0.25", "--test", "0.25"])
    assert result.exit_code == 0, result.output

    total_per_class = {
        cls: sum(len(list((dst / s / cls).glob("*.jpg"))) for s in ("train", "val", "test"))
        for cls in ["cat", "dog"]
    }
    assert total_per_class == {"cat": 20, "dog": 20}


def test_split_rejects_bad_ratios(tmp_path):
    src = _make_flat_classification(tmp_path / "src", per_class=5)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(split, [str(src), str(dst), "--train", "0.5", "--val", "0.5", "--test", "0.5"])
    assert result.exit_code != 0
    assert "sum to 1.0" in result.output


def test_split_dry_run_writes_nothing(tmp_path):
    src = _make_flat_classification(tmp_path / "src", per_class=5)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(split, [str(src), str(dst), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not dst.exists()


def test_split_src_untouched(tmp_path):
    src = _make_flat_classification(tmp_path / "src", per_class=5)
    before = {p.relative_to(src) for p in src.rglob("*.jpg")}
    dst = tmp_path / "dst"
    CliRunner().invoke(split, [str(src), str(dst)])
    assert {p.relative_to(src) for p in src.rglob("*.jpg")} == before


def test_split_yolo_flat_pool_keeps_pairs_and_rewrites_data_yaml(tmp_path):
    src = tmp_path / "yolo_src"
    result = CliRunner().invoke(
        generate,
        [str(src), "--format", "yolo", "--train", "20", "--val", "0", "--test", "0",
         "--image-size", "48", "--max-objects", "3"],
    )
    assert result.exit_code == 0, result.output

    # generate always writes a split subdir; re-flatten to images/*+labels/* directly
    import shutil
    flat_src = tmp_path / "yolo_flat"
    (flat_src / "images").mkdir(parents=True)
    (flat_src / "labels").mkdir(parents=True)
    for img in (src / "images" / "train").glob("*.jpg"):
        shutil.copy2(img, flat_src / "images" / img.name)
    for lbl in (src / "labels" / "train").glob("*.txt"):
        shutil.copy2(lbl, flat_src / "labels" / lbl.name)

    dst = tmp_path / "yolo_dst"
    result = CliRunner().invoke(split, [str(flat_src), str(dst), "--train", "0.6", "--val", "0.2", "--test", "0.2"])
    assert result.exit_code == 0, result.output

    assert (dst / "data.yaml").is_file()
    total = 0
    for split_name in ("train", "val", "test"):
        img_dir = dst / "images" / split_name
        if not img_dir.is_dir():
            continue
        for img in img_dir.glob("*.jpg"):
            label = dst / "labels" / split_name / f"{img.stem}.txt"
            assert label.is_file()
            total += 1
    assert total == 20
