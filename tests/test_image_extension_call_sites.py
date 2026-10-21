"""Which files each call site treats as an image: trainable (TF-decodable) vs readable (PIL)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cvbench.classification import ClassificationTask
from cvbench.datasets.layout import classification_layout_errors
from cvbench.detection import DetectionTask
from cvbench.services import export, prediction

MIXED = ("a.JPG", "b.png", "c.webp", "d.tif", "notes.txt")


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def _solid(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_training_listing_keeps_only_tf_decodable_images(tmp_path):
    from cvbench.classification.data import list_labeled_images

    _touch(tmp_path / "cat", "a.JPG", "b.gif", "c.webp", "d.tif", "notes.txt")

    paths, labels = list_labeled_images(str(tmp_path), ["cat"])

    assert [Path(p).name for p in paths] == ["a.JPG", "b.gif"]
    assert labels == [0, 0]


def test_classification_count_images_counts_decodable_and_nested_images(tmp_path):
    _touch(tmp_path / "cat", *MIXED)
    _touch(tmp_path / "cat" / "nested", "e.png")

    assert ClassificationTask().count_images(str(tmp_path)) == 3


def test_detection_count_images_counts_only_decodable_images(tmp_path):
    _touch(tmp_path, *MIXED)

    assert DetectionTask().count_images(str(tmp_path)) == 2


@pytest.mark.parametrize("names", [("a.tif", "b.webp"), ("notes.txt",)], ids=["readable-only", "text-only"])
def test_layout_check_treats_untrainable_files_as_no_images(tmp_path, names):
    _touch(tmp_path / "train" / "cat", *names)
    _touch(tmp_path / "train" / "dog", "x.png", "y.png")

    errors = classification_layout_errors(tmp_path)

    assert len(errors) == 1
    assert "contain no images: cat" in errors[0]


def test_calibration_set_selects_decodable_images_case_insensitively(tmp_path):
    train = tmp_path / "train"
    _solid(train / "cat" / "x.JPG", (255, 0, 0))
    _solid(train / "cat" / "y.gif", (0, 255, 0))
    _solid(train / "cat" / "z.webp", (0, 0, 255))
    cfg = SimpleNamespace(
        data=SimpleNamespace(train_dir=str(train), val_dir=str(tmp_path / "val")),
        model=SimpleNamespace(input_size=4),
    )
    out = tmp_path / "calib.npy"

    export._build_calibration_set(cfg, out)

    calib = np.load(out)
    assert calib.shape == (2, 4, 4, 3)
    dominant = sorted(int(np.argmax(img.mean(axis=(0, 1)))) for img in calib)
    assert dominant == [0, 1]  # red (x.JPG) and green (y.gif); never the blue .webp


def test_predict_inputs_accept_the_readable_set(tmp_path):
    _touch(tmp_path, "a.PNG", "b.webp", "c.tif", "notes.txt")

    found = prediction._collect_images(str(tmp_path))

    assert [Path(p).name for p in found] == ["a.PNG", "b.webp", "c.tif"]
    assert [Path(p).name for p in prediction._collect_images(str(tmp_path / "b.webp"))] == ["b.webp"]
    assert prediction._collect_images(str(tmp_path / "notes.txt")) == []
