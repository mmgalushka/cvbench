"""Tests for ``data convert`` — Pascal VOC and COCO into the YOLO layout."""
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.data import convert
from cvbench.core.layout import get_layout
from cvbench.datasets import (
    adapters,  # noqa: F401  (registers the layouts)
    layout,
)
from cvbench.datasets import convert as convert_mod

W, H = 100, 50


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (W, H)).save(path)


def _voc_xml(name: str, objects: list[tuple[str, int, int]]) -> str:
    """Objects are ``(class, difficult, truncated)``; every box is x 10..30, y 5..25."""
    objs = "".join(
        f"<object><name>{c}</name><truncated>{t}</truncated><difficult>{d}</difficult>"
        "<bndbox><xmin>10</xmin><ymin>5</ymin><xmax>30</xmax><ymax>25</ymax></bndbox></object>"
        for c, d, t in objects
    )
    return (f"<annotation><filename>{name}.jpg</filename>"
            f"<size><width>{W}</width><height>{H}</height></size>{objs}</annotation>")


def _make_voc(root: Path, n: int = 20, lists: bool = False) -> Path:
    stems = [f"{i:03d}" for i in range(n)]
    for i, stem in enumerate(stems):
        _image(root / "JPEGImages" / f"{stem}.jpg")
        objects = [("cat" if i % 2 else "dog", 0, 0)]
        if i == 0:
            objects += [("dog", 1, 0), ("bird", 0, 1)]
        (root / "Annotations").mkdir(exist_ok=True)
        (root / "Annotations" / f"{stem}.xml").write_text(_voc_xml(stem, objects))
    if lists:
        sets = root / "ImageSets" / "Main"
        sets.mkdir(parents=True)
        (sets / "train.txt").write_text("\n".join(stems[:14]))
        (sets / "val.txt").write_text("\n".join(stems[14:17]))
        (sets / "test.txt").write_text("\n".join(stems[17:19]))  # stems[19] is in no list
    return root


def _coco_json(images: list[str], categories: dict[int, str], crowd: bool = False) -> dict:
    used = [k for k, v in categories.items() if not v.startswith("super")]
    anns = []
    for i, _ in enumerate(images):
        anns.append({"id": i, "image_id": i, "category_id": used[i % len(used)], "bbox": [10, 5, 20, 20],
                     "iscrowd": 0})
    if crowd:
        anns.append({"id": 99, "image_id": 0, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 1})
    return {
        "images": [{"id": i, "file_name": n, "width": W, "height": H} for i, n in enumerate(images)],
        "annotations": anns,
        "categories": [{"id": k, "name": v} for k, v in categories.items()],
    }


def _make_coco_roboflow(root: Path) -> Path:
    for folder, n in (("train", 8), ("valid", 3), ("test", 2)):
        names = [f"{folder}{i}.jpg" for i in range(n)]
        for name in names:
            _image(root / folder / name)
        (root / folder / "_annotations.coco.json").write_text(
            json.dumps(_coco_json(names, {0: "supercat", 1: "dog", 2: "cat"}, crowd=folder == "train"))
        )
    return root


def _make_coco_single(root: Path, n: int = 20) -> Path:
    names = [f"{i:03d}.jpg" for i in range(n)]
    for name in names:
        _image(root / "images" / name)
    (root / "annotations.json").write_text(json.dumps(_coco_json(names, {1: "dog", 2: "cat"})))
    return root


def _yolo_counts(root: Path) -> dict[str, int]:
    return {s: len(list((root / "images" / s).glob("*.jpg"))) for s in layout.SPLIT_NAMES
            if (root / "images" / s).is_dir()}


def test_detect_format(tmp_path):
    assert convert_mod.detect_format(_make_voc(tmp_path / "voc")) == "voc"
    assert convert_mod.detect_format(_make_coco_roboflow(tmp_path / "rf")) == "coco"
    assert convert_mod.detect_format(_make_coco_single(tmp_path / "single")) == "coco"
    (tmp_path / "plain").mkdir()
    assert convert_mod.detect_format(tmp_path / "plain") is None
    (tmp_path / "other.json").write_text("{}")
    assert convert_mod.detect_format(tmp_path) is None


def test_foreign_adapters_are_recognised_but_not_trainable(tmp_path):
    voc = _make_voc(tmp_path / "voc")
    adapter = get_layout("voc")
    assert adapter.sniff(voc) and not get_layout("coco").sniff(voc)
    [msg] = adapter.errors(voc)
    assert "Pascal VOC" in msg and "data convert" in msg
    assert get_layout("coco").sniff(_make_coco_single(tmp_path / "c"))


def test_voc_uses_source_split_lists_and_reports_drops(tmp_path):
    src = _make_voc(tmp_path / "voc", lists=True)
    dst = tmp_path / "out"
    result = CliRunner().invoke(convert, [str(src), str(dst)])
    assert result.exit_code == 0, result.output

    assert _yolo_counts(dst) == {"train": 14, "val": 3, "test": 2}
    names = yaml.safe_load((dst / "data.yaml").read_text())["names"]
    assert names == {0: "bird", 1: "cat", 2: "dog"}  # sorted, stable
    first = (dst / "labels" / "train" / "000.txt").read_text().splitlines()
    # dog (difficult, dropped) is gone; the truncated bird is kept
    assert [ln.split()[0] for ln in first] == ["2", "0"]
    assert first[0] == "2 0.200000 0.300000 0.200000 0.400000"
    assert "difficult" in result.output and "truncated" in result.output
    assert "none of the source's split lists" in result.output


def test_voc_classes_set_ids_and_drop_other_classes(tmp_path):
    src = _make_voc(tmp_path / "voc")
    dst = tmp_path / "out"
    result = CliRunner().invoke(convert, [str(src), str(dst), "--classes", "dog,cat"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load((dst / "data.yaml").read_text())["names"] == {0: "dog", 1: "cat"}
    assert "not selected" in result.output
    all_ids = {ln.split()[0] for f in (dst / "labels").rglob("*.txt") for ln in f.read_text().splitlines()}
    assert all_ids == {"0", "1"}


def test_unknown_class_lists_the_available_ones(tmp_path):
    src = _make_voc(tmp_path / "voc")
    result = CliRunner().invoke(convert, [str(src), str(tmp_path / "out"), "--classes", "zebra"])
    assert result.exit_code != 0
    assert "zebra" in result.output and "cat" in result.output


def test_voc_without_lists_is_split_by_seed(tmp_path):
    src = _make_voc(tmp_path / "voc", n=20)
    a, b = tmp_path / "a", tmp_path / "b"
    for dst in (a, b):
        r = CliRunner().invoke(convert, [str(src), str(dst), "--seed", "3"])
        assert r.exit_code == 0, r.output
    counts = _yolo_counts(a)
    assert sum(counts.values()) == 20 and counts["val"] >= 1
    names = lambda root: sorted(p.name for p in (root / "images" / "val").glob("*.jpg"))  # noqa: E731
    assert names(a) == names(b)


def test_resplit_ignores_source_lists(tmp_path):
    src = _make_voc(tmp_path / "voc", lists=True)
    dst = tmp_path / "out"
    args = ["--resplit", "--train", "0.5", "--val", "0.25", "--test", "0.25"]
    r = CliRunner().invoke(convert, [str(src), str(dst), *args])
    assert r.exit_code == 0, r.output
    counts = _yolo_counts(dst)
    assert sum(counts.values()) == 20
    assert counts["train"] != 14


def test_limit_samples_images(tmp_path):
    src = _make_voc(tmp_path / "voc", n=20)
    dst = tmp_path / "out"
    r = CliRunner().invoke(convert, [str(src), str(dst), "--limit", "8"])
    assert r.exit_code == 0, r.output
    assert sum(_yolo_counts(dst).values()) == 8
    assert "--limit" in r.output


def test_coco_roboflow_folders_map_valid_to_val(tmp_path):
    src = _make_coco_roboflow(tmp_path / "rf")
    dst = tmp_path / "out"
    r = CliRunner().invoke(convert, [str(src), str(dst)])
    assert r.exit_code == 0, r.output
    assert _yolo_counts(dst) == {"train": 8, "val": 3, "test": 2}
    # "supercat" has no boxes, so it is not a class; ids are sorted names
    assert yaml.safe_load((dst / "data.yaml").read_text())["names"] == {0: "cat", 1: "dog"}
    assert "crowd" in r.output


def test_coco_single_file_is_split_and_boxes_are_normalised(tmp_path):
    src = _make_coco_single(tmp_path / "c")
    dst = tmp_path / "out"
    r = CliRunner().invoke(convert, [str(src), str(dst)])
    assert r.exit_code == 0, r.output
    assert sum(_yolo_counts(dst).values()) == 20
    label = next((dst / "labels").rglob("*.txt")).read_text().split()
    assert [float(v) for v in label[1:]] == pytest.approx([0.2, 0.3, 0.2, 0.4])


def test_coco_standard_layout_with_year_folders(tmp_path):
    root = tmp_path / "coco"
    for split, n in (("train", 6), ("val", 2)):
        names = [f"{split}{i}.jpg" for i in range(n)]
        for name in names:
            _image(root / f"{split}2017" / name)
        (root / "annotations").mkdir(exist_ok=True)
        (root / "annotations" / f"instances_{split}2017.json").write_text(
            json.dumps(_coco_json(names, {1: "dog", 2: "cat"}))
        )
    dst = tmp_path / "out"
    r = CliRunner().invoke(convert, [str(root), str(dst)])
    assert r.exit_code == 0, r.output
    assert _yolo_counts(dst) == {"train": 6, "val": 2}
    assert "no val" not in r.output.lower()


def test_dry_run_writes_nothing(tmp_path):
    src = _make_voc(tmp_path / "voc")
    dst = tmp_path / "out"
    r = CliRunner().invoke(convert, [str(src), str(dst), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert not dst.exists()
    assert "Would write" in r.output


def test_rejects_non_empty_dst_bad_ratios_and_unknown_source(tmp_path):
    src = _make_voc(tmp_path / "voc")
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x")
    assert CliRunner().invoke(convert, [str(src), str(full)]).exit_code != 0
    assert CliRunner().invoke(convert, [str(src), str(tmp_path / "o"), "--train", "0.9"]).exit_code != 0
    plain = tmp_path / "plain"
    plain.mkdir()
    r = CliRunner().invoke(convert, [str(plain), str(tmp_path / "o2")])
    assert r.exit_code != 0 and "not a Pascal VOC or COCO" in r.output


def test_converted_dataset_is_a_trainable_yolo_dataset(tmp_path):
    src = _make_voc(tmp_path / "voc", lists=True)
    dst = tmp_path / "out"
    assert CliRunner().invoke(convert, [str(src), str(dst)]).exit_code == 0
    assert layout.is_yolo_dataset(dst)
    assert layout.detection_layout_errors(dst) == []
    assert convert_mod.detect_format(dst) is None
