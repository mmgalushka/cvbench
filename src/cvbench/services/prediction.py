from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path

import numpy as np

from cvbench.core.config import load_config
from cvbench.core.exp_store import resolve_run_dir

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

FORMATS = ("keras", "onnx", "tflite")

_TFLITE_VARIANTS = [
    ("tflite", "model.tflite"),
    ("tflite_float16", "model_float16.tflite"),
    ("tflite_int8", "model_int8.tflite"),
]


def _collect_images(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in _IMAGE_EXTS else []
    return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in _IMAGE_EXTS)


def _supported_extensions() -> str:
    return ", ".join(sorted(_IMAGE_EXTS))


def _unreadable_reason(img_path: str) -> str | None:
    """Return why an image can't be decoded, or None when it is readable."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(img_path) as img:
            img.load()
    except UnidentifiedImageError:
        return "not a valid image file"
    except (OSError, ValueError) as e:
        return str(e) or type(e).__name__
    return None


def _model_path(run_dir: Path, fmt: str) -> Path | None:
    if fmt == "keras":
        p = run_dir / "best.keras"
        return p if p.exists() else None
    if fmt == "onnx":
        p = run_dir / "export" / "onnx" / "model.onnx"
        return p if p.exists() else None
    if fmt == "tflite":
        for subfolder, filename in _TFLITE_VARIANTS:
            p = run_dir / "export" / subfolder / filename
            if p.exists():
                return p
        # Relocated or custom-named exports still live somewhere under export/.
        return next(iter(sorted((run_dir / "export").rglob("*.tflite"))), None)
    raise ValueError(f"Unknown format: {fmt}")


def _export_info_path(run_dir: Path, fmt: str) -> Path | None:
    if fmt == "onnx":
        p = run_dir / "export" / "onnx" / "export_info.json"
        return p if p.exists() else None
    if fmt == "tflite":
        for subfolder, _ in _TFLITE_VARIANTS:
            p = run_dir / "export" / subfolder / "export_info.json"
            if p.exists():
                return p
    return None


def _get_run_info(run_dir: Path, fmt: str) -> tuple[int, list[str] | None]:
    """Return (input_size, class_names) for a run and format."""
    info_path = _export_info_path(run_dir, fmt)
    if info_path:
        info = json.loads(info_path.read_text())
        size = info["input_shape"][1]
        classes = info.get("classes")
        return size, classes
    cfg = load_config(str(run_dir))
    return cfg.model.input_size, cfg.data.classes


def _load_image(img_path: str, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float32)
    return arr[None]  # (1, H, W, 3) RGB


def _infer_keras(model_path: Path, images: list[str], size: int) -> list[list[np.ndarray]]:
    import keras
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model = keras.saving.load_model(str(model_path))
    results = []
    for p in images:
        out = model.predict(_load_image(p, size), verbose=0)
        outs = out if isinstance(out, (list, tuple)) else [out]
        results.append([np.asarray(o[0], dtype=np.float32) for o in outs])
    return results


def _infer_onnx(model_path: Path, images: list[str], size: int) -> list[list[np.ndarray]]:
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime is required for ONNX inference. "
            "Install it with: pip install onnxruntime"
        ) from None
    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    return [
        [
            np.asarray(o[0], dtype=np.float32)
            for o in sess.run(None, {input_name: _load_image(p, size)})
        ]
        for p in images
    ]


def _infer_tflite(model_path: Path, images: list[str], size: int) -> list[list[np.ndarray]]:
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("tensorflow is required for TFLite inference.") from None
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    results = []
    for p in images:
        interp.set_tensor(inp[0]["index"], _load_image(p, size))
        interp.invoke()
        results.append([
            np.asarray(interp.get_tensor(o["index"])[0], dtype=np.float32) for o in out
        ])
    return results


def _class_name(class_names: list[str] | None, idx: int) -> str:
    return class_names[idx] if class_names and idx < len(class_names) else str(idx)


def build_classification_result(
    probs, class_names: list[str] | None, top_k: int | None = 1
) -> dict:
    """Shape one image's class probabilities; ``top_k=None`` keeps every class.

    The single classification result builder — `predict` and the WebUI both use it,
    so a given image yields the same fields from either.
    """
    order = [int(i) for i in np.argsort(-np.asarray(probs), kind="stable")]
    top_idx = order[0]
    return {
        "class_index": top_idx,
        "confidence": float(probs[top_idx]),
        "class_name": _class_name(class_names, top_idx),
        "top_k": [
            {"class_name": _class_name(class_names, i), "confidence": float(probs[i])}
            for i in order[: top_k or len(order)]
        ],
        "probs": [float(x) for x in probs],
    }


def _build_results(
    images: list[str],
    probs_list: list[np.ndarray],
    class_names: list[str] | None,
    top_k: int | None = 1,
) -> list[dict]:
    """Per-image classification results, tagged with the file name."""
    return [
        {"filename": Path(img_path).name, **build_classification_result(probs, class_names, top_k)}
        for img_path, probs in zip(images, probs_list, strict=True)
    ]


def build_detection_result(outputs: list[np.ndarray], cfg, conf: float | None) -> dict:
    """Decode one image's raw head outputs (no batch axis) into detections.

    The single detection result builder. Exported models don't guarantee output
    order (TFLite sorts tensors by name), so scales are re-ordered to match
    ``cfg.detection.strides`` — finest stride first means largest grid first.
    ``conf`` overrides the run's ``detection.conf_threshold``.
    """
    from cvbench.detection.decode import decode_batch

    det = cfg.detection
    if not det.anchors or not det.strides:
        raise ValueError(
            "This detection run predates anchor-based decoding and cannot be "
            "used for inference — retrain with the current detection head."
        )
    class_names = cfg.data.classes
    scales = sorted(outputs, key=lambda o: o.shape[0], reverse=True)
    dets = decode_batch(
        [o[None] for o in scales],
        num_classes=len(class_names),
        anchors=det.anchors,
        strides=det.strides,
        conf_threshold=det.conf_threshold if conf is None else conf,
        max_detections=det.max_detections,
        nms_iou_threshold=det.iou_threshold,
    )[0]
    return {
        "detections": [
            {
                "class_index": d["class_id"],
                "class_name": _class_name(class_names, d["class_id"]),
                "confidence": d["confidence"],
                "x": d["x"],
                "y": d["y"],
                "w": d["w"],
                "h": d["h"],
            }
            for d in dets
        ],
    }


def _build_detection_results(
    images: list[str],
    outputs_list: list[list[np.ndarray]],
    cfg,
    conf: float | None,
) -> list[dict]:
    """Per-image detection results, tagged with the file name."""
    return [
        {"filename": Path(img_path).name, **build_detection_result(outs, cfg, conf)}
        for img_path, outs in zip(images, outputs_list, strict=True)
    ]


def run_experiment_prediction(
    experiment: str,
    input_path: str,
    fmt: str,
    conf: float | None = None,
    top_k: int | None = 1,
    model_path: str | None = None,
) -> dict:
    """Run inference for an experiment across one or all model formats.

    fmt: "keras" | "onnx" | "tflite" | "plan" | "all"
    conf: detection runs only — minimum score for a detection to be reported
        (default: the run's ``detection.conf_threshold``).
    top_k: classification runs only — classes reported per image (None = all).
    model_path: explicit model file; the format is taken from ``fmt``, which
        must be one of keras/onnx/tflite.

    Returns dict with keys:
        plan_only       – True when fmt == "plan"
        task            – "classification" | "detection"
        experiment      – resolved run name
        run_dir         – Path to run directory
        formats_run     – list of {format, model_path, results}
        formats_skipped – list of {format, reason}
        inputs_skipped  – list of {filename, reason} for unreadable images
    """
    run_dir = Path(resolve_run_dir(experiment))

    if fmt == "plan":
        return {"plan_only": True, "experiment": run_dir.name, "run_dir": run_dir}

    run_cfg = load_config(str(run_dir))
    task = getattr(run_cfg, "task", "classification")
    if conf is not None and task != "detection":
        raise ValueError("--conf only applies to detection runs")
    if task == "detection" and top_k != 1:
        raise ValueError("--top-k and --probs only apply to classification runs")

    found = _collect_images(input_path)
    if not found:
        raise ValueError(
            f"No images found at: {input_path} "
            f"(supported extensions: {_supported_extensions()})"
        )
    images: list[str] = []
    inputs_skipped: list[dict] = []
    for img in found:
        reason = _unreadable_reason(img)
        if reason is None:
            images.append(img)
        else:
            inputs_skipped.append({"filename": Path(img).name, "reason": reason})
    if not images:
        first = inputs_skipped[0]
        detail = (
            f"{first['filename']} ({first['reason']})"
            if len(inputs_skipped) == 1
            else f"all {len(inputs_skipped)} files at {input_path} are unreadable"
        )
        raise ValueError(f"Cannot read image: {detail}")

    if model_path is not None:
        if fmt not in FORMATS:
            raise ValueError("--model needs a single format (keras, onnx or tflite)")
        if not Path(model_path).is_file():
            raise ValueError(f"Model file not found: {model_path}")

    formats_to_run = list(FORMATS) if fmt == "all" else [fmt]
    formats_run: list[dict] = []
    formats_skipped: list[dict] = []

    for f in formats_to_run:
        path = Path(model_path) if model_path is not None else _model_path(run_dir, f)
        if path is None:
            formats_skipped.append({
                "format": f,
                "reason": (
                    f"not exported — run: runs export {run_dir.name} --format {f}"
                    " (or pass --model PATH)"
                ),
            })
            continue
        try:
            size, class_names = _get_run_info(run_dir, f)
            if f == "keras":
                outputs_list = _infer_keras(path, images, size)
            elif f == "onnx":
                outputs_list = _infer_onnx(path, images, size)
            else:
                outputs_list = _infer_tflite(path, images, size)
            if task == "detection":
                results = _build_detection_results(images, outputs_list, run_cfg, conf)
            else:
                results = _build_results(
                    images, [o[0] for o in outputs_list], class_names, top_k
                )
            formats_run.append({"format": f, "model_path": str(path), "results": results})
        except RuntimeError as e:
            formats_skipped.append({"format": f, "reason": str(e)})

    return {
        "plan_only": False,
        "task": task,
        "experiment": run_dir.name,
        "run_dir": run_dir,
        "formats_run": formats_run,
        "formats_skipped": formats_skipped,
        "inputs_skipped": inputs_skipped,
    }


# ---------------------------------------------------------------------------
# Legacy single-model inference (kept for the top-level `predict` command)
# ---------------------------------------------------------------------------

def run_prediction(checkpoint: str, input_path: str) -> list[dict]:
    """Run inference on an image or folder using a .keras checkpoint."""
    import keras

    model = keras.saving.load_model(checkpoint)
    size = model.input_shape[1]

    paths = _collect_images(input_path)
    if not paths:
        raise ValueError(f"No images found at: {input_path}")

    results = []
    for img_path in paths:
        arr = _load_image(img_path, size)
        probs = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        results.append({
            "filename": Path(img_path).name,
            "class_index": top_idx,
            "confidence": float(probs[top_idx]),
        })
    return results


# ---------------------------------------------------------------------------
# Web inference — single image supplied as raw bytes
# ---------------------------------------------------------------------------

def _load_run_model(run_name: str):
    """Return (cfg, model, input_size) for a run's keras checkpoint."""
    from cvbench.tasks import resolve_task

    run_dir = Path(resolve_run_dir(run_name))
    cfg = load_config(str(run_dir))
    checkpoint = _model_path(run_dir, "keras")
    if checkpoint is None:
        raise ValueError(f"No keras checkpoint found for run '{run_name}'")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model = resolve_task(cfg).load_model(str(checkpoint))
    return cfg, model, model.input_shape[1]


WEB_TOP_K = 5  # classes the WebUI lists by default


def _check_options(task: str, conf: float | None) -> None:
    if conf is not None and task != "detection":
        raise ValueError("--conf only applies to detection runs")


def _predict_from_input(
    cfg, model, arr: np.ndarray, conf: float | None = None, top_k: int | None = WEB_TOP_K
) -> dict:
    """Run the model on a batched ``(1, H, W, 3)`` float array and shape the
    result with the same builders the CLI uses."""
    task = getattr(cfg, "task", "classification")
    _check_options(task, conf)
    preds = model.predict(arr, verbose=0)
    outputs = [np.asarray(o[0], dtype=np.float32) for o in (preds if isinstance(preds, (list, tuple)) else [preds])]
    if task == "detection":
        result = build_detection_result(outputs, cfg, conf)
        # Boxes are normalized top-left xywh in [0, 1] — what shared.js::buildBoxLayer() consumes.
        top = max(result["detections"], key=lambda d: d["confidence"], default=None)
        return {
            "task": "detection",
            **result,
            "class_name": top["class_name"] if top else None,
            "class_index": top["class_index"] if top else None,
            "confidence": top["confidence"] if top else None,
        }
    return {"task": "classification", **build_classification_result(outputs[0], cfg.data.classes, top_k)}


def predict_image(
    run_name: str, image_bytes: bytes, conf: float | None = None, top_k: int | None = WEB_TOP_K
) -> dict:
    cfg, model, size = _load_run_model(run_name)
    arr = _bytes_to_input(image_bytes, size)
    return _predict_from_input(cfg, model, arr, conf, top_k)


def predict_augmented(
    run_name: str,
    image_bytes: bytes,
    augmentations: list[dict],
    conf: float | None = None,
    top_k: int | None = WEB_TOP_K,
) -> dict:
    import cvbench.augmentations as aug_mod

    cfg, model, size = _load_run_model(run_name)

    img_arr = _bytes_to_numpy(image_bytes, size)

    for aug in augmentations:
        fn = getattr(aug_mod, aug["name"], None)
        if fn is None:
            raise ValueError(f"Unknown augmentation: {aug['name']}")
        img_arr = fn(img_arr, **aug.get("params", {}))

    augmented_b64 = _numpy_to_base64_png(img_arr)

    arr = img_arr.astype(np.float32)[None]
    result = _predict_from_input(cfg, model, arr, conf, top_k)
    result["augmented_image_b64"] = augmented_b64
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bytes_to_input(image_bytes: bytes, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img, dtype=np.float32)
    return arr[None]  # (1, H, W, 3) RGB


def _bytes_to_numpy(image_bytes: bytes, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size))
    return np.array(img)


def _numpy_to_base64_png(arr: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
