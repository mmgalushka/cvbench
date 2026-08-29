"""Tests for cvbench.detection.model/losses/decode."""
import numpy as np
import pytest

pytestmark = pytest.mark.tf

_STRIDES = [16, 32]
_ANCHORS = [
    [[0.15, 0.15], [0.25, 0.25], [0.35, 0.35]],
    [[0.45, 0.45], [0.55, 0.55], [0.7, 0.7]],
]


def _minimal_cfg(fine_tune_from_layer=0):
    from cvbench.core.config import CVBenchConfig

    cfg = CVBenchConfig()
    cfg.model.backbone = "resnet_18"
    cfg.model.weights = "none"  # fast, no download — shape/serialization tests don't need pretrained
    cfg.model.input_size = 64
    cfg.model.num_classes = 4
    cfg.model.dropout = 0.1
    cfg.model.fine_tune_from_layer = fine_tune_from_layer
    cfg.detection.strides = _STRIDES
    cfg.detection.anchors = _ANCHORS
    cfg.training.learning_rate = 1e-3
    return cfg


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def test_build_model_output_shapes():
    from cvbench.detection.model import build_model

    model = build_model(_minimal_cfg())
    assert model.output_shape == [
        (None, 4, 4, 3 * (5 + 4)),   # stride 16: G=4, A=3, 5+C=9
        (None, 2, 2, 3 * (5 + 4)),   # stride 32: G=2, A=3
    ]


def test_build_model_names_backbone_layer():
    from cvbench.detection.model import build_model

    model = build_model(_minimal_cfg())
    # Raises ValueError if no layer named "backbone" exists.
    backbone = model.get_layer("backbone")
    assert backbone is not None


def test_fit_one_step_on_two_images():
    from cvbench.detection.data import encode_targets
    from cvbench.detection.model import build_model

    cfg = _minimal_cfg()
    model = build_model(cfg)

    x = (np.random.rand(2, 64, 64, 3).astype("float32") * 255)
    boxes_per_image = [
        [(0, (0.3, 0.3, 0.2, 0.2))],
        [],  # hard negative
    ]
    per_scale = [[], []]
    for boxes in boxes_per_image:
        targets = encode_targets(boxes, num_classes=4, anchors=_ANCHORS, strides=_STRIDES, input_size=64)
        for i, t in enumerate(targets):
            per_scale[i].append(t.reshape(t.shape[0], t.shape[1], -1))
    y = tuple(np.stack(scale) for scale in per_scale)

    history = model.fit(x, y, epochs=1, verbose=0)
    loss = history.history["loss"][0]
    assert np.isfinite(loss)
    assert loss > 0


def test_overfit_one_batch_recovers_boxes():
    """Standard detector sanity gate: train on a handful of images until the
    model can recover each box's class and location — if this fails, suspect
    target encoding (anchor assignment, tw/th log-scaling) or the loss."""
    from cvbench.detection.data import encode_targets
    from cvbench.detection.decode import decode_batch
    from cvbench.detection.metrics import iou
    from cvbench.detection.model import build_model

    cfg = _minimal_cfg()
    cfg.model.num_classes = 3
    cfg.model.dropout = 0.0
    model = build_model(cfg)

    rng = np.random.RandomState(0)
    n = 6
    images = (rng.rand(n, 64, 64, 3) * 255).astype("float32")
    boxes_per_image = [[(i % 3, (0.3, 0.3, 0.2, 0.2))] for i in range(n)]

    per_scale = [[], []]
    for boxes in boxes_per_image:
        targets = encode_targets(boxes, num_classes=3, anchors=_ANCHORS, strides=_STRIDES, input_size=64)
        for i, t in enumerate(targets):
            per_scale[i].append(t.reshape(t.shape[0], t.shape[1], -1))
    y = tuple(np.stack(scale) for scale in per_scale)

    model.fit(images, y, epochs=200, verbose=0, batch_size=n)

    preds = model.predict(images, verbose=0)
    dets = decode_batch(
        list(preds), 3, _ANCHORS, _STRIDES,
        conf_threshold=0.3, max_detections=10, nms_iou_threshold=0.5,
    )
    for i, image_dets in enumerate(dets):
        assert image_dets, f"image {i}: no detection survived confidence threshold"
        best = max(image_dets, key=lambda d: d["confidence"])
        gt_cls, gt_box = boxes_per_image[i][0]
        assert best["class_id"] == gt_cls
        assert iou(gt_box, (best["x"], best["y"], best["w"], best["h"])) > 0.7


# ---------------------------------------------------------------------------
# YoloLoss serialization round-trip
# ---------------------------------------------------------------------------

def test_yolo_loss_get_config_round_trips():
    from cvbench.detection.losses import YoloLoss

    loss = YoloLoss(num_classes=5, num_anchors=3, noobj_weight=0.3)
    restored = YoloLoss.from_config(loss.get_config())
    assert restored.num_classes == 5
    assert restored.num_anchors == 3
    assert restored.noobj_weight == pytest.approx(0.3)
