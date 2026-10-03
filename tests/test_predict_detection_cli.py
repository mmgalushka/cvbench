"""`predict` on detection runs — decode, ordering and printing, without TensorFlow.

The model-inference helpers are stubbed with raw head outputs, so these run in
the `-m "not tf"` CI job; the real train → export → predict path is covered by
`test_prediction_service.py`.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from cvbench.cli.predict import predict
from cvbench.services import prediction

ANCHORS = [[[0.2, 0.2]], [[0.5, 0.5]]]  # one anchor per scale
CLASSES = ["cat", "dog"]


def _cfg(task="detection", anchors=ANCHORS):
    return SimpleNamespace(
        task=task,
        data=SimpleNamespace(classes=CLASSES),
        detection=SimpleNamespace(
            anchors=anchors, strides=[16, 32], conf_threshold=0.5,
            max_detections=10, iou_threshold=0.5,
        ),
    )


def _head(grid: int, cell=None, cls=0):
    """Raw (G, G, 1*(5+C)) head output: all logits very negative, except one
    confident box at ``cell`` (gy, gx) for class ``cls``."""
    out = np.full((grid, grid, 5 + len(CLASSES)), -10.0, dtype=np.float32)
    if cell is not None:
        gy, gx = cell
        out[gy, gx, :4] = 0.0          # box at cell centre, anchor-sized
        out[gy, gx, 4] = 10.0          # objectness
        out[gy, gx, 5 + cls] = 10.0    # class score
    return out


@pytest.fixture
def det_run(tmp_path, monkeypatch):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"")
    monkeypatch.setattr(prediction, "resolve_run_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(prediction, "load_config", lambda path: _cfg())
    monkeypatch.setattr(prediction, "_collect_images", lambda p: [str(img)])
    monkeypatch.setattr(prediction, "_get_run_info", lambda run_dir, f: (64, CLASSES))
    monkeypatch.setattr(prediction, "_model_path", lambda run_dir, f: tmp_path / f)
    return img


def _stub_outputs(monkeypatch, outputs, fmts=("keras", "onnx", "tflite")):
    for name in fmts:
        monkeypatch.setattr(prediction, f"_infer_{name}", lambda p, i, s: [outputs])


def test_decode_reorders_scales_and_applies_conf(det_run, monkeypatch):
    fine, coarse = _head(4, (1, 2), cls=1), _head(2, (0, 0), cls=0)
    # Exported graphs may emit the coarse head first — decode must still pair
    # each tensor with the right anchors/stride.
    _stub_outputs(monkeypatch, [coarse, fine], fmts=("keras",))

    result = prediction.run_experiment_prediction("run", str(det_run), "keras")

    assert result["task"] == "detection"
    dets = result["formats_run"][0]["results"][0]["detections"]
    assert {d["class_name"] for d in dets} == {"cat", "dog"}
    dog = next(d for d in dets if d["class_name"] == "dog")
    assert dog["confidence"] > 0.9
    assert dog["x"] == pytest.approx((2 + 0.5) / 4 - 0.2 / 2)  # fine-head anchor

    strict = prediction.run_experiment_prediction("run", str(det_run), "keras", conf=1.0)
    assert strict["formats_run"][0]["results"][0]["detections"] == []


def test_conf_rejected_for_classification(det_run, monkeypatch):
    monkeypatch.setattr(prediction, "load_config", lambda path: _cfg("classification"))
    with pytest.raises(ValueError, match="--conf only applies to detection runs"):
        prediction.run_experiment_prediction("run", str(det_run), "keras", conf=0.3)


def test_legacy_run_without_anchors_is_rejected(det_run, monkeypatch):
    monkeypatch.setattr(prediction, "load_config", lambda path: _cfg(anchors=None))
    _stub_outputs(monkeypatch, [_head(4), _head(2)], fmts=("keras",))
    with pytest.raises(ValueError, match="predates anchor-based decoding"):
        prediction.run_experiment_prediction("run", str(det_run), "keras")


def test_cli_prints_boxes_for_single_format(det_run, monkeypatch):
    _stub_outputs(monkeypatch, [_head(4, (1, 2), cls=1), _head(2)], fmts=("keras",))

    result = CliRunner().invoke(predict, ["run", str(det_run)])

    assert result.exit_code == 0, result.output
    assert "1 detection" in result.output
    assert "dog" in result.output and "x=" in result.output


def test_cli_conf_option_filters_boxes(det_run, monkeypatch):
    _stub_outputs(monkeypatch, [_head(4, (1, 2)), _head(2)], fmts=("keras",))

    result = CliRunner().invoke(predict, ["run", str(det_run), "--conf", "1"])

    assert result.exit_code == 0, result.output
    assert "0 detections" in result.output


def test_cli_conf_out_of_range_is_usage_error(det_run):
    result = CliRunner().invoke(predict, ["run", str(det_run), "--conf", "2"])
    assert result.exit_code == 2


def test_cli_all_formats_flags_disagreement(det_run, monkeypatch):
    agree = [_head(4, (1, 2)), _head(2)]
    drift = [_head(4), _head(2)]  # tflite lost the box
    monkeypatch.setattr(prediction, "_infer_keras", lambda p, i, s: [agree])
    monkeypatch.setattr(prediction, "_infer_onnx", lambda p, i, s: [agree])
    monkeypatch.setattr(prediction, "_infer_tflite", lambda p, i, s: [drift])

    result = CliRunner().invoke(predict, ["run", str(det_run), "--format", "all"])

    assert result.exit_code == 0, result.output
    rows = [ln for ln in result.output.splitlines() if ln.startswith(" a.jpg")]
    assert len(rows) == 1
    assert rows[0].count("⚠️") == 1
    assert "1 box (top" in rows[0] and "0 boxes" in rows[0]
