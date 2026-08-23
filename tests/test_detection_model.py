"""Tests for cvbench.detection.model/losses/decode."""
import numpy as np
import pytest

pytestmark = pytest.mark.tf


def _minimal_cfg(fine_tune_from_layer=0):
    from cvbench.core.config import CVBenchConfig

    cfg = CVBenchConfig()
    cfg.model.backbone = "efficientnet_b0"
    cfg.model.input_size = 64
    cfg.model.num_classes = 4
    cfg.model.dropout = 0.1
    cfg.model.fine_tune_from_layer = fine_tune_from_layer
    cfg.detection.grid_stride = 4
    cfg.training.learning_rate = 1e-3
    return cfg


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def test_build_model_output_shape():
    from cvbench.detection.model import build_model

    model = build_model(_minimal_cfg())
    G = 64 // 4
    assert model.output_shape == (None, G, G, 4 + 4)


def test_build_model_names_backbone_layer():
    from cvbench.detection.model import build_model

    model = build_model(_minimal_cfg())
    # Raises ValueError if no layer named "backbone" exists.
    backbone = model.get_layer("backbone")
    assert backbone is not None


def test_fit_one_step_on_two_images():
    from cvbench.detection.data import encode_target
    from cvbench.detection.model import build_model

    cfg = _minimal_cfg()
    model = build_model(cfg)

    G = 64 // 4
    x = (np.random.rand(2, 64, 64, 3).astype("float32") * 255)
    y = np.stack([
        encode_target([(0, (0.3, 0.3, 0.2, 0.2))], num_classes=4, grid_size=G),
        encode_target([], num_classes=4, grid_size=G),  # hard negative
    ])

    history = model.fit(x, y, epochs=1, verbose=0)
    loss = history.history["loss"][0]
    assert np.isfinite(loss)
    assert loss > 0


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_round_trip_identical_predictions(tmp_path):
    from cvbench.detection.model import build_model
    from cvbench.tasks import get_task

    model = build_model(_minimal_cfg())
    x = np.random.rand(1, 64, 64, 3).astype("float32") * 255
    pred_before = model.predict(x, verbose=0)

    path = str(tmp_path / "model.keras")
    model.save(path)

    task = get_task("detection")
    reloaded = task.load_model(path)
    pred_after = reloaded.predict(x, verbose=0)

    np.testing.assert_allclose(pred_before, pred_after, atol=1e-6)


@pytest.mark.parametrize("ftl,expected_trainable", [(0, False), (-1, True), (3, True)])
def test_fine_tune_from_layer_backbone_trainable_state_survives_reload(tmp_path, ftl, expected_trainable):
    from cvbench.detection.model import build_model
    from cvbench.tasks import get_task

    model = build_model(_minimal_cfg(fine_tune_from_layer=ftl))
    path = str(tmp_path / f"model_{ftl}.keras")
    model.save(path)

    task = get_task("detection")
    reloaded = task.load_model(path)
    assert reloaded.get_layer("backbone").trainable is expected_trainable


# ---------------------------------------------------------------------------
# CenterNetLoss
# ---------------------------------------------------------------------------

def test_loss_registered_and_round_trips_config():
    from cvbench.detection.losses import CenterNetLoss

    loss = CenterNetLoss(num_classes=4, alpha=2.0, beta=4.0, size_weight=0.1, offset_weight=1.0)
    config = loss.get_config()
    restored = CenterNetLoss.from_config(config)
    assert restored.num_classes == 4
    assert restored.alpha == 2.0
    assert restored.size_weight == 0.1


def test_loss_is_zero_for_perfect_prediction():
    import keras

    from cvbench.detection.losses import CenterNetLoss
    from cvbench.detection.data import encode_target

    G, C = 8, 3
    target = encode_target([(1, (0.3, 0.3, 0.2, 0.2))], num_classes=C, grid_size=G)
    y_true = target[None]  # add batch dim

    # A "prediction" that matches the target's heatmap/size/offset exactly
    # (dropping the mask channel, which isn't part of the model's output).
    y_pred = np.clip(target[..., :C + 4], 1e-6, 1 - 1e-6)[None]

    loss_fn = CenterNetLoss(num_classes=C)
    value = float(loss_fn(y_true, y_pred))
    assert value == pytest.approx(0.0, abs=1e-3)


def test_loss_penalizes_wrong_prediction_more():
    from cvbench.detection.losses import CenterNetLoss
    from cvbench.detection.data import encode_target

    G, C = 8, 3
    target = encode_target([(1, (0.3, 0.3, 0.2, 0.2))], num_classes=C, grid_size=G)
    y_true = target[None]

    good_pred = np.clip(target[..., :C + 4], 1e-6, 1 - 1e-6)[None]
    bad_pred = np.full_like(good_pred, 0.5)

    loss_fn = CenterNetLoss(num_classes=C)
    good_loss = float(loss_fn(y_true, good_pred))
    bad_loss = float(loss_fn(y_true, bad_pred))
    assert bad_loss > good_loss


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def test_decode_recovers_a_known_peak():
    from cvbench.detection.decode import decode_batch
    from cvbench.detection.data import encode_target

    G, C = 16, 3
    boxes = [(1, (0.3, 0.3, 0.2, 0.2))]
    target = encode_target(boxes, num_classes=C, grid_size=G)
    # Feed the target's own heatmap/size/offset straight in as "predictions".
    preds = target[..., :C + 4][None]

    detections = decode_batch(preds, num_classes=C, conf_threshold=0.5, max_detections=10)
    assert len(detections) == 1
    dets = detections[0]
    assert len(dets) >= 1
    best = max(dets, key=lambda d: d["confidence"])
    assert best["class_id"] == 1
    assert best["x"] == pytest.approx(0.3, abs=0.05)
    assert best["y"] == pytest.approx(0.3, abs=0.05)
    assert best["w"] == pytest.approx(0.2, abs=0.05)
    assert best["h"] == pytest.approx(0.2, abs=0.05)


def test_decode_respects_confidence_threshold():
    from cvbench.detection.decode import decode_batch

    G, C = 8, 2
    preds = np.zeros((1, G, G, C + 4), dtype="float32")
    preds[0, 4, 4, 0] = 0.1  # below threshold
    detections = decode_batch(preds, num_classes=C, conf_threshold=0.25, max_detections=10)
    assert detections == [[]]


def test_decode_suppresses_non_local_maxima():
    from cvbench.detection.decode import decode_batch

    G, C = 8, 1
    preds = np.zeros((1, G, G, C + 4), dtype="float32")
    preds[0, 3, 3, 0] = 0.9
    preds[0, 3, 4, 0] = 0.5  # adjacent, lower — should be suppressed by peak-picking
    preds[0, 3, 3, C] = 0.2
    preds[0, 3, 3, C + 1] = 0.2
    detections = decode_batch(preds, num_classes=C, conf_threshold=0.25, max_detections=10)[0]
    assert len(detections) == 1
    assert detections[0]["confidence"] == pytest.approx(0.9)


def test_decode_respects_max_detections():
    from cvbench.detection.decode import decode_batch

    G, C = 8, 1
    preds = np.zeros((1, G, G, C + 4), dtype="float32")
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            preds[0, i, j, 0] = 0.5 + 0.01 * (i + j)
            preds[0, i, j, C] = 0.1
            preds[0, i, j, C + 1] = 0.1
    detections = decode_batch(preds, num_classes=C, conf_threshold=0.1, max_detections=3)[0]
    assert len(detections) == 3
