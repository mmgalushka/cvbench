from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path

import numpy as np

from cvbench.core.config import load_config
from cvbench.core.runs import resolve_run_dir

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
        return None
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


def _get_input_size(run_dir: Path, fmt: str) -> int:
    info_path = _export_info_path(run_dir, fmt)
    if info_path:
        return json.loads(info_path.read_text())["input_shape"][1]
    return load_config(str(run_dir)).model.input_size


def _get_class_names(run_dir: Path, fmt: str) -> list[str] | None:
    info_path = _export_info_path(run_dir, fmt)
    if info_path:
        return json.loads(info_path.read_text()).get("classes")
    try:
        return load_config(str(run_dir)).data.classes
    except Exception:
        return None


def _load_image(img_path: str, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.float32)[None]  # (1, H, W, 3)


def _infer_keras(model_path: Path, images: list[str], size: int) -> list[np.ndarray]:
    import keras
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model = keras.saving.load_model(str(model_path))
    return [model.predict(_load_image(p, size), verbose=0)[0] for p in images]


def _infer_onnx(model_path: Path, images: list[str], size: int) -> list[np.ndarray]:
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime is required for ONNX inference. "
            "Install it with: pip install onnxruntime"
        )
    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    return [
        np.array(sess.run(None, {input_name: _load_image(p, size)})[0][0], dtype=np.float32)
        for p in images
    ]


def _infer_tflite(model_path: Path, images: list[str], size: int) -> list[np.ndarray]:
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("tensorflow is required for TFLite inference.")
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    results = []
    for p in images:
        interp.set_tensor(inp[0]["index"], _load_image(p, size))
        interp.invoke()
        results.append(np.array(interp.get_tensor(out[0]["index"])[0], dtype=np.float32))
    return results


def _build_results(
    images: list[str],
    probs_list: list[np.ndarray],
    class_names: list[str] | None,
) -> list[dict]:
    results = []
    for img_path, probs in zip(images, probs_list):
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        class_name = (
            class_names[top_idx]
            if class_names and top_idx < len(class_names)
            else str(top_idx)
        )
        results.append({
            "filename": Path(img_path).name,
            "class_index": top_idx,
            "confidence": confidence,
            "class_name": class_name,
        })
    return results


def run_experiment_prediction(
    experiment: str,
    input_path: str,
    fmt: str,
) -> dict:
    """Run inference for an experiment across one or all model formats.

    fmt: "keras" | "onnx" | "tflite" | "plan" | "all"

    Returns dict with keys:
        plan_only       – True when fmt == "plan"
        experiment      – resolved run name
        run_dir         – Path to run directory
        formats_run     – list of {format, results}
        formats_skipped – list of {format, reason}
    """
    run_dir = Path(resolve_run_dir(experiment))

    if fmt == "plan":
        return {"plan_only": True, "experiment": run_dir.name, "run_dir": run_dir}

    images = _collect_images(input_path)
    if not images:
        raise ValueError(f"No images found at: {input_path}")

    formats_to_run = list(FORMATS) if fmt == "all" else [fmt]
    formats_run: list[dict] = []
    formats_skipped: list[dict] = []

    for f in formats_to_run:
        path = _model_path(run_dir, f)
        if path is None:
            formats_skipped.append({
                "format": f,
                "reason": f"not exported — run: cvbench runs export {run_dir.name} --format {f}",
            })
            continue
        try:
            size = _get_input_size(run_dir, f)
            class_names = _get_class_names(run_dir, f)
            if f == "keras":
                probs_list = _infer_keras(path, images, size)
            elif f == "onnx":
                probs_list = _infer_onnx(path, images, size)
            else:
                probs_list = _infer_tflite(path, images, size)
            formats_run.append({"format": f, "results": _build_results(images, probs_list, class_names)})
        except RuntimeError as e:
            formats_skipped.append({"format": f, "reason": str(e)})

    return {
        "plan_only": False,
        "experiment": run_dir.name,
        "run_dir": run_dir,
        "formats_run": formats_run,
        "formats_skipped": formats_skipped,
    }


# ---------------------------------------------------------------------------
# Legacy single-model inference (kept for the top-level `predict` command)
# ---------------------------------------------------------------------------

def run_prediction(checkpoint: str, input_path: str) -> list[dict]:
    """Run inference on an image or folder using a .keras checkpoint."""
    import keras
    import tensorflow as tf

    model = keras.saving.load_model(checkpoint)
    size = model.input_shape[1]

    paths = _collect_images(input_path)
    if not paths:
        raise ValueError(f"No images found at: {input_path}")

    results = []
    for img_path in paths:
        img = tf.keras.utils.load_img(img_path, target_size=(size, size))
        arr = tf.keras.utils.img_to_array(img)[None]
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

def predict_image(run_name: str, image_bytes: bytes) -> dict:
    import keras

    run_dir = Path(resolve_run_dir(run_name))
    class_names = load_config(str(run_dir)).data.classes

    checkpoint = _model_path(run_dir, "keras")
    if checkpoint is None:
        raise ValueError(f"No keras checkpoint found for run '{run_name}'")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model = keras.saving.load_model(str(checkpoint))
    size = model.input_shape[1]

    arr = _bytes_to_input(image_bytes, size)
    probs = model.predict(arr, verbose=0)[0]
    return _build_result(probs, class_names)


def predict_augmented(run_name: str, image_bytes: bytes, augmentations: list[dict]) -> dict:
    import cvbench.augmentations as aug_mod
    import keras

    run_dir = Path(resolve_run_dir(run_name))
    class_names = load_config(str(run_dir)).data.classes

    checkpoint = _model_path(run_dir, "keras")
    if checkpoint is None:
        raise ValueError(f"No keras checkpoint found for run '{run_name}'")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model = keras.saving.load_model(str(checkpoint))
    size = model.input_shape[1]

    img_arr = _bytes_to_numpy(image_bytes, size)

    for aug in augmentations:
        fn = getattr(aug_mod, aug["name"], None)
        if fn is None:
            raise ValueError(f"Unknown augmentation: {aug['name']}")
        img_arr = fn(img_arr, **aug.get("params", {}))

    augmented_b64 = _numpy_to_base64_png(img_arr)

    arr = img_arr.astype(np.float32)[None]
    probs = model.predict(arr, verbose=0)[0]

    result = _build_result(probs, class_names)
    result["augmented_image_b64"] = augmented_b64
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_result(probs: np.ndarray, class_names: list[str]) -> dict:
    top_k = sorted(
        [{"class_name": class_names[i], "confidence": float(probs[i])} for i in range(len(probs))],
        key=lambda x: -x["confidence"],
    )
    return {
        "class_name": top_k[0]["class_name"],
        "class_index": int(np.argmax(probs)),
        "confidence": top_k[0]["confidence"],
        "top_k": top_k,
    }


def _bytes_to_input(image_bytes: bytes, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.float32)[None]


def _bytes_to_numpy(image_bytes: bytes, size: int) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.uint8)


def _numpy_to_base64_png(arr: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
