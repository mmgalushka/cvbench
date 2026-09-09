"""Tests for ``data merge`` — combine matching splits across N datasets."""
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.data import merge
from cvbench.cli.generate import generate


def _make_split_classification(root: Path, classes: list[str], per_class: int = 3) -> Path:
    for split in ["train", "val"]:
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(per_class):
                arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i:03d}.jpg", quality=90)
    return root


def test_merge_classification_unions_classes_and_splits(tmp_path):
    src = tmp_path / "src"
    _make_split_classification(src / "a", ["cat", "dog"], per_class=3)
    _make_split_classification(src / "b", ["dog", "bird"], per_class=2)

    dst = tmp_path / "dst"
    result = CliRunner().invoke(merge, [str(src), str(dst)])
    assert result.exit_code == 0, result.output

    for split in ["train", "val"]:
        assert len(list((dst / split / "cat").glob("*.jpg"))) == 3
        assert len(list((dst / split / "dog").glob("*.jpg"))) == 5  # 3 + 2
        assert len(list((dst / split / "bird").glob("*.jpg"))) == 2


def test_merge_classification_no_filename_collisions(tmp_path):
    src = tmp_path / "src"
    _make_split_classification(src / "a", ["cat"], per_class=3)
    _make_split_classification(src / "b", ["cat"], per_class=3)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(merge, [str(src), str(dst)])
    assert result.exit_code == 0, result.output
    assert len(list((dst / "train" / "cat").glob("*.jpg"))) == 6


def test_merge_requires_two_sources(tmp_path):
    src = tmp_path / "src"
    _make_split_classification(src / "only_one", ["cat"], per_class=2)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(merge, [str(src), str(dst)])
    assert result.exit_code != 0
    assert "at least 2" in result.output


def test_merge_rejects_mixed_layouts(tmp_path):
    src = tmp_path / "src"
    _make_split_classification(src / "cls_ds", ["cat"], per_class=2)
    yolo_out = src / "yolo_ds"
    r = CliRunner().invoke(
        generate,
        [str(yolo_out), "--format", "yolo", "--train", "2", "--val", "0", "--test", "0",
         "--image-size", "32", "--max-objects", "1"],
    )
    assert r.exit_code == 0, r.output

    dst = tmp_path / "dst"
    result = CliRunner().invoke(merge, [str(src), str(dst)])
    assert result.exit_code != 0
    assert "mixes" in result.output.lower()


def test_merge_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "src"
    _make_split_classification(src / "a", ["cat"], per_class=2)
    _make_split_classification(src / "b", ["cat"], per_class=2)
    dst = tmp_path / "dst"
    result = CliRunner().invoke(merge, [str(src), str(dst), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not dst.exists()


def test_merge_yolo_remaps_class_indices_and_writes_data_yaml(tmp_path):
    src = tmp_path / "src"
    a = src / "a"
    b = src / "b"
    ra = CliRunner().invoke(
        generate, [str(a), "--format", "yolo", "--train", "4", "--val", "0", "--test", "0",
                   "--image-size", "48", "--max-objects", "2"],
    )
    rb = CliRunner().invoke(
        generate, [str(b), "--format", "yolo", "--train", "4", "--val", "0", "--test", "0",
                   "--image-size", "48", "--max-objects", "2"],
    )
    assert ra.exit_code == 0 and rb.exit_code == 0

    # give 'b' a shuffled class order relative to 'a', so remap is exercised
    import yaml
    b_yaml_path = b / "data.yaml"
    b_yaml = yaml.safe_load(b_yaml_path.read_text())
    names = b_yaml["names"]
    shuffled = {0: names[3], 1: names[2], 2: names[1], 3: names[0]}
    b_yaml["names"] = shuffled
    b_yaml_path.write_text(yaml.safe_dump(b_yaml))
    # remap b's label files to match the shuffled names (swap 0<->3, 1<->2)
    swap = {0: 3, 1: 2, 2: 1, 3: 0}
    for label in (b / "labels" / "train").glob("*.txt"):
        lines = label.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            old_id = int(float(parts[0]))
            new_lines.append(" ".join([str(swap[old_id]), *parts[1:]]))
        label.write_text("\n".join(new_lines) + "\n")

    dst = tmp_path / "dst"
    result = CliRunner().invoke(merge, [str(src), str(dst)])
    assert result.exit_code == 0, result.output

    assert (dst / "data.yaml").is_file()
    merged_yaml = yaml.safe_load((dst / "data.yaml").read_text())
    merged_names = [merged_yaml["names"][i] for i in sorted(merged_yaml["names"])]
    assert set(merged_names) == set(names.values())

    images = list((dst / "images" / "train").glob("*.jpg"))
    assert len(images) == 8
    for img in images:
        label = dst / "labels" / "train" / f"{img.stem}.txt"
        assert label.is_file()
        for line in label.read_text().splitlines():
            cls_id = int(line.split()[0])
            assert 0 <= cls_id < len(merged_names)
