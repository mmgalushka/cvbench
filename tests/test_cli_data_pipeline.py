"""CLI-level tests for data upsample/split/flatten/prep — wiring, errors, dry-run."""
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.data import data, explore


def _save(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


def _pool(root: Path, per_class: int = 10) -> Path:
    seed = 0
    for cls in ("cat", "dog"):
        for i in range(per_class):
            seed += 1
            _save(root / cls / f"{i}.png", seed)
    return root


def _count(root: Path) -> int:
    return sum(1 for p in root.rglob("*.png"))


def _run(*args):
    return CliRunner().invoke(data, [str(a) for a in args])


# --- upsample -------------------------------------------------------------

def test_upsample_copies_originals_and_generates_to_target(tmp_path):
    src = tmp_path / "src"
    for i in range(3):
        _save(src / f"{i}.png", i)
    aug = tmp_path / "aug.yaml"
    assert _run("aug", "--preset", "light", "-o", aug).exit_code == 0
    dst = tmp_path / "dst"
    result = _run("upsample", src, dst, "--augmentation", aug, "--target", 6)
    assert result.exit_code == 0, result.output
    assert "Copied 3 original(s)" in result.output
    assert "Generated 3 augmented image(s)" in result.output
    assert len([p for p in dst.iterdir() if p.is_file()]) == 6


def test_upsample_rejects_target_not_above_source(tmp_path):
    src = tmp_path / "src"
    for i in range(3):
        _save(src / f"{i}.png", i)
    aug = tmp_path / "aug.yaml"
    aug.write_text("transforms: []\n")
    result = _run("upsample", src, tmp_path / "dst", "--augmentation", aug, "--target", 3)
    assert result.exit_code != 0
    assert "already has 3 images" in result.output


def test_upsample_rejects_missing_or_empty_source_and_nonempty_dst(tmp_path):
    aug = tmp_path / "aug.yaml"
    aug.write_text("transforms: []\n")
    args = ("--augmentation", aug, "--target", 5)

    assert "not found" in _run("upsample", tmp_path / "nope", tmp_path / "d", *args).output

    empty = tmp_path / "empty"
    empty.mkdir()
    assert "No images found" in _run("upsample", empty, tmp_path / "d", *args).output

    src = tmp_path / "src"
    _save(src / "a.png", 1)
    dst = tmp_path / "dst"
    _save(dst / "x.png", 2)
    assert "already contains" in _run("upsample", src, dst, *args).output


# --- explore --------------------------------------------------------------

def test_explore_rejects_missing_directory(tmp_path):
    result = CliRunner().invoke(explore, [str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_explore_rejects_dataset_without_class_dirs(tmp_path):
    (tmp_path / "ds" / "train").mkdir(parents=True)
    result = CliRunner().invoke(explore, [str(tmp_path / "ds")])
    assert result.exit_code != 0
    assert "No class subdirectories" in result.output


# --- split / flatten / prep ----------------------------------------------

def test_split_then_flatten_roundtrip(tmp_path):
    src = _pool(tmp_path / "src")
    split_dir = tmp_path / "split"
    result = _run("split", src, split_dir, "--seed", 1)
    assert result.exit_code == 0, result.output
    assert _count(split_dir) == 20

    flat = tmp_path / "flat"
    result = _run("flatten", split_dir, flat)
    assert result.exit_code == 0, result.output
    assert _count(flat) == 20


def test_split_and_flatten_dry_run_write_nothing(tmp_path):
    src = _pool(tmp_path / "src")
    assert _run("split", src, tmp_path / "s", "--dry-run").exit_code == 0
    assert not (tmp_path / "s").exists()

    split_dir = tmp_path / "split"
    _run("split", src, split_dir)
    assert _run("flatten", split_dir, tmp_path / "f", "--dry-run").exit_code == 0
    assert not (tmp_path / "f").exists()


def test_split_rejects_already_split_and_flatten_rejects_already_flat(tmp_path):
    src = _pool(tmp_path / "src")
    split_dir = tmp_path / "split"
    _run("split", src, split_dir)

    result = _run("split", split_dir, tmp_path / "again")
    assert result.exit_code != 0
    assert "already split" in result.output

    result = _run("flatten", src, tmp_path / "f")
    assert result.exit_code != 0
    assert "already flat" in result.output


@pytest.mark.parametrize("cmd", ["split", "flatten"])
def test_split_flatten_reject_missing_src_and_nonempty_dst(cmd, tmp_path):
    assert "not found" in _run(cmd, tmp_path / "nope", tmp_path / "d").output
    src = _pool(tmp_path / "src")
    dst = tmp_path / "dst"
    _save(dst / "x.png", 1)
    assert "already contains" in _run(cmd, src, dst).output


def test_split_rejects_bad_ratios(tmp_path):
    src = _pool(tmp_path / "src")
    result = _run("split", src, tmp_path / "d", "--train", 0.9, "--val", 0.9)
    assert result.exit_code != 0
    assert "sum to 1.0" in result.output


def test_prep_writes_output_and_dry_run_does_not(tmp_path):
    src = _pool(tmp_path / "src", per_class=3)
    assert _run("prep", src, tmp_path / "dry", "--dry-run").exit_code == 0
    assert not (tmp_path / "dry").exists()

    out = tmp_path / "out"
    result = _run("prep", src, out)
    assert result.exit_code == 0, result.output
    assert _count(out) == 6
