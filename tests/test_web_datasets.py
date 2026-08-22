"""Datasets API tests — YOLO detection support alongside classification folders."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.generate import CLASSES, generate

pytest.importorskip("fastapi", reason="requires the 'web' extra")

from cvbench.web.api import datasets as api  # noqa: E402


@pytest.fixture
def yolo_root(tmp_path) -> Path:
    out = tmp_path / "yolo"
    result = CliRunner().invoke(
        generate,
        [str(out), "--format", "yolo", "--train", "5", "--val", "2", "--test", "0",
         "--image-size", "64", "--max-objects", "3"],
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


@pytest.fixture
def cls_root(tmp_path) -> Path:
    out = tmp_path / "cls"
    result = CliRunner().invoke(
        generate,
        [str(out), "--train", "2", "--val", "1", "--test", "0", "--image-size", "32"],
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


# ---------------------------------------------------------------------------
# Dataset entries
# ---------------------------------------------------------------------------

def test_yolo_dataset_entry(yolo_root):
    entry = api._dataset_entry(yolo_root)
    assert entry["format"] == "yolo"
    assert entry["classes"] == CLASSES
    assert sorted(entry["splits"]) == ["train", "val"]


def test_classification_dataset_entry(cls_root):
    entry = api._dataset_entry(cls_root)
    assert entry["format"] == "classification"
    assert sorted(entry["splits"]) == ["train", "val"]


def test_class_names_fall_back_to_label_ids(yolo_root):
    (yolo_root / "data.yaml").unlink()
    names = api._yolo_class_names(yolo_root)
    assert names and all(n.isdigit() for n in names)


# ---------------------------------------------------------------------------
# Image listing
# ---------------------------------------------------------------------------

def test_list_images_returns_boxes(yolo_root):
    split_id = api._dataset_entry(yolo_root)["splits"]["train"]["id"]
    res = api.list_images(split_id, None, 1, 60)

    assert res["format"] == "yolo"
    assert res["total"] == 5
    assert res["classes"] == CLASSES

    for item in res["items"]:
        assert item["boxes"]
        for box in item["boxes"]:
            assert box["class"] in CLASSES
            assert 0 <= box["x"] <= 1 and 0 <= box["y"] <= 1
            assert 0 < box["w"] <= 1 and 0 < box["h"] <= 1
            assert box["x"] + box["w"] <= 1.0001
            assert box["y"] + box["h"] <= 1.0001


def test_list_images_filters_by_class(yolo_root):
    split_id = api._dataset_entry(yolo_root)["splits"]["train"]["id"]
    all_items = api.list_images(split_id, None, 1, 60)["items"]

    for cls in CLASSES:
        expected = [it for it in all_items if any(b["class"] == cls for b in it["boxes"])]
        res = api.list_images(split_id, cls, 1, 60)
        assert res["total"] == len(expected)
        assert [it["path"] for it in res["items"]] == [it["path"] for it in expected]


def test_list_images_unknown_class(yolo_root):
    from fastapi import HTTPException

    split_id = api._dataset_entry(yolo_root)["splits"]["train"]["id"]
    with pytest.raises(HTTPException) as exc:
        api.list_images(split_id, "not-a-class", 1, 60)
    assert exc.value.status_code == 404


def test_classification_listing_unchanged(cls_root):
    split_id = api._dataset_entry(cls_root)["splits"]["train"]["id"]
    res = api.list_images(split_id, "circle", 1, 60)
    assert res["format"] == "classification"
    assert res["total"] == 2
    assert all(it["class"] == "circle" for it in res["items"])
    assert "boxes" not in res["items"][0]


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def test_delete_image_removes_label(yolo_root):
    split_id = api._dataset_entry(yolo_root)["splits"]["train"]["id"]
    item = api.list_images(split_id, None, 1, 1)["items"][0]

    api.delete_image(split_id, item["path"])

    stem = Path(item["path"]).stem
    assert not (yolo_root / "images" / "train" / f"{stem}.jpg").exists()
    assert not (yolo_root / "labels" / "train" / f"{stem}.txt").exists()
    assert api.list_images(split_id, None, 1, 60)["total"] == 4
