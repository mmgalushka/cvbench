"""Tests for cvbench.detection.data — YOLO -> tf.data target encoding."""
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.tf

from cvbench.cli.generate import generate
from cvbench.datasets.shapes import CLASSES

_STRIDES = [16, 32]
_ANCHORS = [
    [[0.15, 0.15], [0.25, 0.25], [0.35, 0.35]],
    [[0.45, 0.45], [0.55, 0.55], [0.7, 0.7]],
]


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
# encode_targets
# ---------------------------------------------------------------------------

def test_encode_targets_shapes():
    from cvbench.detection.data import encode_targets

    targets = encode_targets([], num_classes=4, anchors=_ANCHORS, strides=_STRIDES, input_size=64)
    assert len(targets) == 2
    assert targets[0].shape == (4, 4, 3, 4 + 6)   # G=64/16=4, A=3
    assert targets[1].shape == (2, 2, 3, 4 + 6)   # G=64/32=2, A=3
    assert all(t.dtype == np.float32 for t in targets)


def test_encode_targets_empty_boxes_is_all_zero():
    from cvbench.detection.data import encode_targets

    targets = encode_targets([], num_classes=4, anchors=_ANCHORS, strides=_STRIDES, input_size=64)
    assert all(np.all(t == 0) for t in targets)


def test_encode_targets_assigns_best_iou_anchor_and_scale():
    from cvbench.detection.data import encode_targets

    # w=h=0.25 is closest (by shape IoU) to the [0.25, 0.25] anchor on the
    # fine (stride-16) scale — box shapes only, not the anchor's magnitude
    # relative to the object's actual pixel size.
    boxes = [(2, (0.375, 0.375, 0.25, 0.25))]
    fine, coarse = encode_targets(
        boxes, num_classes=4, anchors=_ANCHORS, strides=_STRIDES, input_size=64,
    )
    assert np.all(coarse == 0)  # nothing assigned on the coarse scale

    G = 4
    cx_i, cy_i = int(0.5 * G), int(0.5 * G)  # center (0.5, 0.5) -> cell 2,2
    ai = 1  # index of the [0.25, 0.25] anchor
    assert fine[cy_i, cx_i, ai, 4] == pytest.approx(1.0)   # obj
    assert fine[cy_i, cx_i, ai, 5] == pytest.approx(0.0)   # not ignored
    assert fine[cy_i, cx_i, ai, 6 + 2] == pytest.approx(1.0)  # class one-hot
    # tw/th = log(box / anchor) = log(0.25/0.25) = 0
    assert fine[cy_i, cx_i, ai, 2] == pytest.approx(0.0, abs=1e-5)
    assert fine[cy_i, cx_i, ai, 3] == pytest.approx(0.0, abs=1e-5)


def test_encode_targets_offset_is_fractional_part_of_center():
    from cvbench.detection.data import encode_targets

    G = 4
    xc, yc = 2.3 / G, 2.3 / G
    w = h = 0.25
    boxes = [(0, (xc - w / 2, yc - h / 2, w, h))]
    fine, _coarse = encode_targets(
        boxes, num_classes=4, anchors=_ANCHORS, strides=_STRIDES, input_size=64,
    )
    cx_i, cy_i = 2, 2
    ai = 1
    assert fine[cy_i, cx_i, ai, 0] == pytest.approx(0.3, abs=1e-4)
    assert fine[cy_i, cx_i, ai, 1] == pytest.approx(0.3, abs=1e-4)


def test_encode_targets_ignores_out_of_range_class_id():
    from cvbench.detection.data import encode_targets

    boxes = [(99, (0.4, 0.4, 0.2, 0.2))]
    targets = encode_targets(boxes, num_classes=4, anchors=_ANCHORS, strides=_STRIDES, input_size=64)
    assert all(np.all(t == 0) for t in targets)


def test_encode_decode_round_trip_recovers_the_box():
    """The single highest-value correctness check for anchor-based encoding:
    a box run through encode -> (perfect-confidence) logits -> decode must
    come back at ~unit IoU. Catches anchor/stride/offset sign errors that a
    shape assertion alone would miss."""
    from cvbench.detection.data import encode_targets
    from cvbench.detection.decode import decode_batch
    from cvbench.detection.metrics import iou

    num_classes = 4
    box = (2, (0.375, 0.375, 0.2, 0.2))
    targets = encode_targets([box], num_classes, _ANCHORS, _STRIDES, input_size=64)

    def _logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    preds = []
    for si, (t, stride) in enumerate(zip(targets, _STRIDES, strict=True)):
        g = 64 // stride
        A = len(_ANCHORS[si])
        pred = np.zeros((1, g, g, A, 5 + num_classes), dtype=np.float32)
        for gy, gx, a in zip(*np.where(t[..., 4] == 1.0), strict=True):
            cell = t[gy, gx, a]
            pred[0, gy, gx, a, 0] = _logit(cell[0])
            pred[0, gy, gx, a, 1] = _logit(cell[1])
            pred[0, gy, gx, a, 2] = cell[2]
            pred[0, gy, gx, a, 3] = cell[3]
            pred[0, gy, gx, a, 4] = _logit(0.99)
            for c in range(num_classes):
                pred[0, gy, gx, a, 5 + c] = _logit(0.99 if cell[6 + c] == 1.0 else 0.01)
        preds.append(pred.reshape(1, g, g, -1))

    dets = decode_batch(
        preds, num_classes, _ANCHORS, _STRIDES,
        conf_threshold=0.1, max_detections=10, nms_iou_threshold=0.5,
    )
    best = max(dets[0], key=lambda d: d["confidence"])
    assert best["class_id"] == box[0]
    assert iou(box[1], (best["x"], best["y"], best["w"], best["h"])) > 0.99


# ---------------------------------------------------------------------------
# build_detection_dataset — end to end over a generated YOLO dataset
# ---------------------------------------------------------------------------

def test_build_detection_dataset_batch_shapes(yolo_root):
    from cvbench.core.config import CVBenchConfig
    from cvbench.detection.data import build_detection_dataset

    cfg = CVBenchConfig()
    cfg.model.input_size = 64
    cfg.data.batch_size = 2
    cfg.detection.strides = _STRIDES
    cfg.detection.anchors = _ANCHORS

    ds = build_detection_dataset(
        str(yolo_root / "images" / "train"), str(yolo_root), CLASSES, cfg, training=False,
    )
    images, (t_fine, t_coarse) = next(iter(ds))
    assert images.shape == (2, 64, 64, 3)
    assert t_fine.shape == (2, 4, 4, 3 * (6 + len(CLASSES)))
    assert t_coarse.shape == (2, 2, 2, 3 * (6 + len(CLASSES)))


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
    cfg.detection.strides = [16, 32]
    cfg.detection.anchors = _ANCHORS

    ds = build_detection_dataset(str(img_dir), str(root), ["a", "b"], cfg, training=False)
    _images, targets = next(iter(ds))
    assert all(np.all(t.numpy() == 0) for t in targets)


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
    cfg.detection.strides = [16, 32]
    cfg.detection.anchors = _ANCHORS

    ds = build_detection_dataset(str(img_dir), str(root), ["a", "b"], cfg, training=False)
    _images, targets = next(iter(ds))
    assert all(np.all(t.numpy() == 0) for t in targets)


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
    assert cfg.model.backbone == "resnet_18"  # detection default, not explicit
    assert len(cfg.detection.anchors) == len(cfg.detection.strides)


def test_detection_task_resolve_layout_respects_explicit_backbone(yolo_root):
    from cvbench.core.config import build_config
    from cvbench.tasks import get_task

    cfg = build_config(
        str(yolo_root), task="detection", backbone="efficientnet_b0", input_size=64, batch_size=2,
    )
    task = get_task("detection")
    task.resolve_layout(cfg)
    assert cfg.model.backbone == "efficientnet_b0"


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


def test_detection_task_build_model_delegates_to_detection_model():
    from cvbench.core.config import build_config
    from cvbench.tasks import get_task

    cfg = build_config(
        "data", task="detection", backbone="resnet_18", input_size=64, weights="none",
    )
    cfg.model.num_classes = 3
    cfg.detection.strides = _STRIDES
    cfg.detection.anchors = _ANCHORS
    task = get_task("detection")
    model = task.build_model(cfg)
    assert model.output_shape == [(None, 4, 4, 3 * (5 + 3)), (None, 2, 2, 3 * (5 + 3))]
