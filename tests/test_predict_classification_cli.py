"""`predict` for classification runs and the prediction helpers, without TensorFlow.

Model inference (`_infer_*`) is stubbed with synthetic probabilities, so these
run in the `-m "not tf"` CI job; the real train → export → predict path is
covered by `test_prediction_service.py`.
"""
import base64
import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from cvbench.cli.predict import predict
from cvbench.services import prediction

CLASSES = ["cat", "dog", "bird"]


def _png_bytes(size=8, color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def _probs(top: int, p: float = 0.9) -> np.ndarray:
    out = np.full(len(CLASSES), (1 - p) / (len(CLASSES) - 1), dtype=np.float32)
    out[top] = p
    return out


def _cfg(task="classification"):
    return SimpleNamespace(
        task=task,
        data=SimpleNamespace(classes=CLASSES),
        model=SimpleNamespace(input_size=16),
    )


@pytest.fixture
def images(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for name in ("b.png", "a.jpg"):
        (d / name).write_bytes(_png_bytes())
    (d / "notes.txt").write_text("not an image")
    return d


@pytest.fixture
def cls_run(tmp_path, monkeypatch):
    """A classification run whose model files all 'exist'."""
    monkeypatch.setattr(prediction, "resolve_run_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(prediction, "load_config", lambda path: _cfg())
    monkeypatch.setattr(prediction, "_get_run_info", lambda run_dir, f: (16, CLASSES))
    monkeypatch.setattr(prediction, "_model_path", lambda run_dir, f: tmp_path / f)
    return tmp_path


def _stub(monkeypatch, fmt, tops):
    """Make ``_infer_<fmt>`` return one probability vector per image."""
    monkeypatch.setattr(
        prediction, f"_infer_{fmt}",
        lambda p, imgs, s: [[_probs(tops[i % len(tops)])] for i in range(len(imgs))],
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_single_format_prints_top_class_per_image(cls_run, images, monkeypatch):
    _stub(monkeypatch, "keras", [1, 0])

    result = CliRunner().invoke(predict, ["run", str(images)])

    assert result.exit_code == 0, result.output
    lines = [ln for ln in result.output.splitlines() if ln.startswith((" a.jpg", " b.png"))]
    assert [ln.split()[0] for ln in lines] == ["a.jpg", "b.png"]  # sorted, txt ignored
    assert "dog" in lines[0] and "90.0%" in lines[0]
    assert "cat" in lines[1]
    assert " 2 images" in result.output


def test_cli_single_image_uses_singular_count(cls_run, images, monkeypatch):
    _stub(monkeypatch, "keras", [2])

    result = CliRunner().invoke(predict, ["run", str(images / "a.jpg")])

    assert result.exit_code == 0, result.output
    assert " 1 image\n" in result.output
    assert "bird" in result.output


def test_cli_all_formats_flags_disagreement_and_lists_skipped(cls_run, images, monkeypatch):
    monkeypatch.setattr(
        prediction, "_model_path",
        lambda run_dir, f: None if f == "onnx" else run_dir / f,
    )
    _stub(monkeypatch, "keras", [0])
    _stub(monkeypatch, "tflite", [1])  # drifted after conversion

    result = CliRunner().invoke(predict, ["run", str(images / "a.jpg"), "--format", "all"])

    assert result.exit_code == 0, result.output
    row = next(ln for ln in result.output.splitlines() if ln.startswith(" a.jpg"))
    assert row.count("⚠️") == 1
    assert "cat (90.0%)" in row and "dog (90.0%)" in row
    assert "skipped: onnx" in result.output
    assert "runs export" in result.output
    assert "predict --format plan" in result.output


def test_cli_all_formats_agreement_has_no_warning(cls_run, images, monkeypatch):
    for fmt in prediction.FORMATS:
        _stub(monkeypatch, fmt, [0])

    result = CliRunner().invoke(predict, ["run", str(images / "a.jpg"), "--format", "all"])

    assert result.exit_code == 0, result.output
    assert "⚠️" not in result.output


def test_cli_reports_when_nothing_could_run(cls_run, images, monkeypatch):
    monkeypatch.setattr(prediction, "_model_path", lambda run_dir, f: None)

    result = CliRunner().invoke(predict, ["run", str(images / "a.jpg"), "--format", "onnx"])

    assert result.exit_code == 0, result.output
    assert "No models available" in result.output
    assert "not exported" in result.output


def test_cli_runtime_error_becomes_skipped_format(cls_run, images, monkeypatch):
    def boom(p, imgs, s):
        raise RuntimeError("onnxruntime is required")

    monkeypatch.setattr(prediction, "_infer_onnx", boom)

    result = CliRunner().invoke(predict, ["run", str(images / "a.jpg"), "--format", "onnx"])

    assert result.exit_code == 0, result.output
    assert "skipped: onnx" in result.output and "onnxruntime is required" in result.output


def test_cli_folder_without_images_is_an_error(cls_run, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = CliRunner().invoke(predict, ["run", str(empty)])

    assert result.exit_code == 1
    assert "No images found" in result.output


def test_cli_conf_rejected_for_classification(cls_run, images):
    result = CliRunner().invoke(predict, ["run", str(images), "--conf", "0.3"])

    assert result.exit_code == 1
    assert "--conf only applies to detection runs" in result.output


@pytest.mark.parametrize("args, missing", [
    ([], "EXPERIMENT"),
    (["run"], "INPUT"),
])
def test_cli_requires_experiment_and_input(args, missing):
    result = CliRunner().invoke(predict, args)

    assert result.exit_code == 2
    assert f"{missing} is required" in result.output


def test_cli_plan_needs_no_run_or_images():
    result = CliRunner().invoke(predict, ["--format", "plan"])

    assert result.exit_code == 0, result.output
    assert "[plan]" in result.output
    assert "infer.py" in result.output and "tensorrt" in result.output
    assert "runs export  --format plan" in result.output


def test_cli_plan_names_the_run_when_given(tmp_path):
    result = CliRunner().invoke(predict, ["my_run", "--format", "plan"])

    assert result.exit_code == 0, result.output
    assert "runs export my_run --format plan" in result.output


def test_service_plan_short_circuits(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "resolve_run_dir", lambda name: str(tmp_path))

    result = prediction.run_experiment_prediction("run", "ignored", "plan")

    assert result == {"plan_only": True, "experiment": tmp_path.name, "run_dir": tmp_path}


# ── file discovery and run metadata ──────────────────────────────────────────

def test_collect_images_filters_extensions_case_insensitively(tmp_path):
    (tmp_path / "sub").mkdir()
    for rel in ("x.PNG", "sub/y.webp", "z.txt"):
        (tmp_path / rel).write_bytes(b"")

    found = prediction._collect_images(str(tmp_path))

    assert [p.rsplit("/", 1)[-1] for p in found] == ["y.webp", "x.PNG"]  # sorted by full path


def test_collect_images_single_file(tmp_path):
    img, txt = tmp_path / "a.jpeg", tmp_path / "a.txt"
    img.write_bytes(b"")
    txt.write_bytes(b"")

    assert prediction._collect_images(str(img)) == [str(img)]
    assert prediction._collect_images(str(txt)) == []


def test_model_path_per_format(tmp_path):
    assert all(prediction._model_path(tmp_path, f) is None for f in prediction.FORMATS)

    (tmp_path / "best.keras").write_bytes(b"")
    (tmp_path / "export" / "onnx").mkdir(parents=True)
    (tmp_path / "export" / "onnx" / "model.onnx").write_bytes(b"")
    (tmp_path / "export" / "tflite_int8").mkdir(parents=True)
    (tmp_path / "export" / "tflite_int8" / "model_int8.tflite").write_bytes(b"")

    assert prediction._model_path(tmp_path, "keras") == tmp_path / "best.keras"
    assert prediction._model_path(tmp_path, "onnx").name == "model.onnx"
    assert prediction._model_path(tmp_path, "tflite").name == "model_int8.tflite"
    with pytest.raises(ValueError, match="Unknown format"):
        prediction._model_path(tmp_path, "plan")


def test_tflite_prefers_float32_over_quantized_variants(tmp_path):
    for sub, name in (("tflite", "model.tflite"), ("tflite_float16", "model_float16.tflite")):
        (tmp_path / "export" / sub).mkdir(parents=True)
        (tmp_path / "export" / sub / name).write_bytes(b"")

    assert prediction._model_path(tmp_path, "tflite").name == "model.tflite"


def test_get_run_info_prefers_export_info(tmp_path, monkeypatch):
    info_dir = tmp_path / "export" / "onnx"
    info_dir.mkdir(parents=True)
    (info_dir / "export_info.json").write_text(
        json.dumps({"input_shape": [1, 32, 32, 3], "classes": ["x", "y"]})
    )
    monkeypatch.setattr(prediction, "load_config", lambda p: pytest.fail("config used"))

    assert prediction._get_run_info(tmp_path, "onnx") == (32, ["x", "y"])


def test_get_run_info_falls_back_to_config(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "load_config", lambda p: _cfg())

    assert prediction._get_run_info(tmp_path, "keras") == (16, CLASSES)
    assert prediction._get_run_info(tmp_path, "onnx") == (16, CLASSES)


def test_export_info_path_ignores_keras(tmp_path):
    assert prediction._export_info_path(tmp_path, "keras") is None
    assert prediction._export_info_path(tmp_path, "tflite") is None


# ── image loading and result shaping ─────────────────────────────────────────

def test_load_image_resizes_and_batches(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png_bytes(size=5, color=(255, 0, 0)))

    arr = prediction._load_image(str(p), 12)

    assert arr.shape == (1, 12, 12, 3) and arr.dtype == np.float32
    assert arr[0, 0, 0].tolist() == [255.0, 0.0, 0.0]  # raw RGB, no rescaling


def test_load_image_converts_grayscale_to_rgb(tmp_path):
    p = tmp_path / "g.png"
    Image.new("L", (4, 4), 7).save(p)

    assert prediction._load_image(str(p), 4).shape == (1, 4, 4, 3)


def test_build_results_names_classes_and_falls_back_to_index():
    probs = [_probs(1), np.array([0.1, 0.1, 0.1, 0.7], dtype=np.float32)]

    named = prediction._build_results(["d/a.jpg", "d/b.jpg"], probs, CLASSES)
    bare = prediction._build_results(["d/a.jpg"], [_probs(2)], None)

    assert named[0] == {
        "filename": "a.jpg", "class_index": 1, "confidence": pytest.approx(0.9),
        "class_name": "dog",
    }
    assert named[1]["class_name"] == "3"  # index beyond the class list
    assert bare[0]["class_name"] == "2"


def test_build_result_sorts_top_k():
    result = prediction._build_result(np.array([0.2, 0.5, 0.3]), CLASSES)

    assert result["class_name"] == "dog" and result["class_index"] == 1
    assert [e["class_name"] for e in result["top_k"]] == ["dog", "bird", "cat"]


def test_bytes_helpers_roundtrip():
    data = _png_bytes(size=6, color=(1, 2, 3))

    batched = prediction._bytes_to_input(data, 10)
    plain = prediction._bytes_to_numpy(data, 10)
    b64 = prediction._numpy_to_base64_png(plain)

    assert batched.shape == (1, 10, 10, 3) and batched.dtype == np.float32
    assert plain.shape == (10, 10, 3)
    decoded = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
    assert decoded.tolist() == plain.tolist()


# ── inference backends ───────────────────────────────────────────────────────

def test_infer_onnx_without_runtime_raises_runtime_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # makes the import fail

    with pytest.raises(RuntimeError, match="onnxruntime is required"):
        prediction._infer_onnx(tmp_path / "m.onnx", [], 8)


def test_infer_onnx_runs_each_image_through_the_session(monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(_png_bytes())
    seen = []

    class FakeSession:
        def __init__(self, path):
            seen.append(path)

        def get_inputs(self):
            return [SimpleNamespace(name="input_1")]

        def run(self, _, feed):
            assert feed["input_1"].shape == (1, 8, 8, 3)
            return [np.array([[0.1, 0.9]], dtype=np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))

    out = prediction._infer_onnx(tmp_path / "m.onnx", [str(img), str(img)], 8)

    assert seen == [str(tmp_path / "m.onnx")]
    assert len(out) == 2 and out[0][0].tolist() == pytest.approx([0.1, 0.9])


def test_infer_tflite_without_tensorflow_raises_runtime_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "tensorflow", None)

    with pytest.raises(RuntimeError, match="tensorflow is required"):
        prediction._infer_tflite(tmp_path / "m.tflite", [], 8)


def test_infer_tflite_collects_every_output_tensor(monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(_png_bytes())

    class FakeInterpreter:
        def __init__(self, model_path):
            self.set = None

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [{"index": 0}]

        def get_output_details(self):
            return [{"index": 1}, {"index": 2}]

        def set_tensor(self, index, arr):
            assert index == 0 and arr.shape == (1, 8, 8, 3)

        def invoke(self):
            pass

        def get_tensor(self, index):
            return np.full((1, 2), index, dtype=np.float32)

    fake_tf = SimpleNamespace(lite=SimpleNamespace(Interpreter=FakeInterpreter))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    out = prediction._infer_tflite(tmp_path / "m.tflite", [str(img)], 8)

    assert [o.tolist() for o in out[0]] == [[1.0, 1.0], [2.0, 2.0]]


# ── web inference shaping ────────────────────────────────────────────────────

def test_predict_from_input_dispatches_on_task():
    class Model:
        def predict(self, arr, verbose=0):
            return np.array([[0.1, 0.8, 0.1]], dtype=np.float32)

    arr = np.zeros((1, 4, 4, 3), dtype=np.float32)
    result = prediction._predict_from_input(_cfg(), Model(), arr)

    assert result["class_name"] == "dog" and len(result["top_k"]) == 3


def test_detection_result_for_web_picks_top_box():
    det = SimpleNamespace(
        anchors=[[[0.2, 0.2]]], strides=[16], conf_threshold=0.5,
        max_detections=10, iou_threshold=0.5,
    )
    cfg = SimpleNamespace(task="detection", data=SimpleNamespace(classes=CLASSES), detection=det)
    head = np.full((1, 4, 4, 5 + len(CLASSES)), -10.0, dtype=np.float32)
    head[0, 1, 2, :4] = 0.0
    head[0, 1, 2, 4] = 10.0
    head[0, 1, 2, 5 + 2] = 10.0

    result = prediction._build_detection_result([head], cfg)

    assert result["task"] == "detection" and len(result["detections"]) == 1
    assert result["class_name"] == "bird" and result["class_index"] == 2

    empty = prediction._build_detection_result([np.full_like(head, -10.0)], cfg)
    assert empty["detections"] == [] and empty["class_name"] is None

    det.anchors = None
    with pytest.raises(ValueError, match="predates anchor-based decoding"):
        prediction._build_detection_result([head], cfg)
