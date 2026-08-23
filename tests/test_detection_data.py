"""Tests for cvbench.detection.data — YOLO -> tf.data target encoding."""
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.tf

from cvbench.cli.generate import generate
from cvbench.datasets.shapes import CLASSES


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


# ---------------------------------------------------------------------------
# encode_target
# ---------------------------------------------------------------------------

def test_encode_target_shape():
    from cvbench.detection.data import encode_target

    target = encode_target([], num_classes=4, grid_size=16)
    assert target.shape == (16, 16, 4 + 5)
    assert target.dtype == np.float32


def test_encode_target_empty_boxes_is_all_zero():
    from cvbench.detection.data import encode_target

    target = encode_target([], num_classes=4, grid_size=16)
    assert np.all(target == 0)


def test_encode_target_places_box_in_correct_cell_with_correct_size_offset():
    from cvbench.detection.data import encode_target

    G, C = 16, 4
    # A box centered at (0.5, 0.5) with w=h=0.25 -> center cell (8, 8).
    boxes = [(2, (0.375, 0.375, 0.25, 0.25))]  # class 2, top-left xywh
    target = encode_target(boxes, num_classes=C, grid_size=G)

    cx_i, cy_i = 8, 8
    assert target[cy_i, cx_i, 2] == pytest.approx(1.0)  # heatmap peak at exact center
    assert target[cy_i, cx_i, C] == pytest.approx(0.25)      # width
    assert target[cy_i, cx_i, C + 1] == pytest.approx(0.25)  # height
    assert target[cy_i, cx_i, C + 4] == pytest.approx(1.0)   # mask

    # Sub-cell offset: true center in grid units is 8.0 exactly -> offset 0.
    assert target[cy_i, cx_i, C + 2] == pytest.approx(0.0, abs=1e-5)
    assert target[cy_i, cx_i, C + 3] == pytest.approx(0.0, abs=1e-5)

    # Other classes' heatmap channels stay untouched at this cell.
    for c in range(C):
        if c != 2:
            assert target[cy_i, cx_i, c] == 0.0


def test_encode_target_offset_is_fractional_part_of_center():
    from cvbench.detection.data import encode_target

    G, C = 16, 4
    # Center at grid-unit 8.3 in both axes.
    xc, yc = 8.3 / G, 8.3 / G
    w = h = 0.1
    boxes = [(0, (xc - w / 2, yc - h / 2, w, h))]
    target = encode_target(boxes, num_classes=C, grid_size=G)

    cx_i, cy_i = 8, 8
    assert target[cy_i, cx_i, C + 2] == pytest.approx(0.3, abs=1e-4)
    assert target[cy_i, cx_i, C + 3] == pytest.approx(0.3, abs=1e-4)


def test_encode_target_ignores_out_of_range_class_id():
    from cvbench.detection.data import encode_target

    boxes = [(99, (0.4, 0.4, 0.2, 0.2))]
    target = encode_target(boxes, num_classes=4, grid_size=16)
    assert np.all(target == 0)


# ---------------------------------------------------------------------------
# build_detection_dataset — end to end over a generated YOLO dataset
# ---------------------------------------------------------------------------

def test_build_detection_dataset_batch_shapes(yolo_root):
    from cvbench.core.config import CVBenchConfig
    from cvbench.detection.data import build_detection_dataset

    cfg = CVBenchConfig()
    cfg.model.input_size = 64
    cfg.data.batch_size = 2
    cfg.detection.grid_stride = 4

    ds = build_detection_dataset(
        str(yolo_root / "images" / "train"), str(yolo_root), CLASSES, cfg, training=False,
    )
    images, targets = next(iter(ds))
    G = 64 // 4
    assert images.shape == (2, 64, 64, 3)
    assert targets.shape == (2, G, G, len(CLASSES) + 5)


def test_build_detection_dataset_missing_label_is_hard_negative(tmp_path):
    """An image with no matching .txt file must yield an all-zero target,
    not an error — a valid hard negative per the design."""
    from PIL import Image

    from cvbench.core.config import CVBenchConfig
    from cvbench.detection.data import build_detection_dataset

    root = tmp_path / "yolo"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img_dir / "0000.jpg")
    # Deliberately no 0000.txt written.

    cfg = CVBenchConfig()
    cfg.model.input_size = 32
    cfg.data.batch_size = 1
    cfg.detection.grid_stride = 4

    ds = build_detection_dataset(str(img_dir), str(root), ["a", "b"], cfg, training=False)
    _images, targets = next(iter(ds))
    assert np.all(targets.numpy() == 0)


def test_build_detection_dataset_empty_label_file_is_hard_negative(tmp_path):
    from PIL import Image

    from cvbench.core.config import CVBenchConfig
    from cvbench.detection.data import build_detection_dataset

    root = tmp_path / "yolo"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img_dir / "0000.jpg")
    (lbl_dir / "0000.txt").write_text("")

    cfg = CVBenchConfig()
    cfg.model.input_size = 32
    cfg.data.batch_size = 1
    cfg.detection.grid_stride = 4

    ds = build_detection_dataset(str(img_dir), str(root), ["a", "b"], cfg, training=False)
    _images, targets = next(iter(ds))
    assert np.all(targets.numpy() == 0)


# ---------------------------------------------------------------------------
# DetectionTask
# ---------------------------------------------------------------------------

def test_detection_task_resolve_layout(yolo_root):
    from cvbench.core.config import build_config
    from cvbench.tasks import get_task

    cfg = build_config(str(yolo_root), task="detection", input_size=64, batch_size=2)
    task = get_task("detection")
    spec = task.resolve_layout(cfg)

    assert spec.class_names == CLASSES
    assert spec.train_dir == str(yolo_root / "images" / "train")
    assert spec.val_dir == str(yolo_root / "images" / "val")
    assert spec.test_dir == ""  # no test split was generated
    assert spec.num_train_images == 5
    assert cfg.data.classes == CLASSES
    assert cfg.model.num_classes == len(CLASSES)


def test_detection_task_validate_config_requires_val_split(tmp_path):
    from cvbench.core.config import build_config
    from cvbench.tasks import get_task

    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)

    cfg = build_config(str(root), task="detection")
    task = get_task("detection")
    errors = task.validate_config(cfg)
    assert any("images/val" in e for e in errors)


def test_detection_task_filter_transforms_drops_geometric_only():
    from cvbench.core.config import TransformConfig
    from cvbench.tasks import get_task

    task = get_task("detection")
    transforms = [
        TransformConfig(name="keras_flip"),
        TransformConfig(name="keras_rotation"),
        TransformConfig(name="keras_zoom"),
        TransformConfig(name="keras_translation"),
        TransformConfig(name="keras_crop"),
        TransformConfig(name="keras_brightness"),
        TransformConfig(name="aug_blur"),
    ]
    kept = task.filter_transforms(transforms)
    kept_names = {t.name for t in kept}
    assert kept_names == {"keras_brightness", "aug_blur"}


def test_detection_task_build_model_not_implemented_yet():
    from cvbench.core.config import CVBenchConfig
    from cvbench.tasks import get_task

    task = get_task("detection")
    with pytest.raises(NotImplementedError):
        task.build_model(CVBenchConfig())
