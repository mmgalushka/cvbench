from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cvbench.core.config import load_config
from cvbench.core.errors import ConfigError, DatasetError, ExportError, NotFoundError
from cvbench.core.exp_store import resolve_run_dir
from cvbench.datasets.layout import READABLE_IMAGE_EXTS, list_images
from cvbench.tasks import check_predict_options, resolve_task

if TYPE_CHECKING:
    from cvbench.core.task import Task

FORMATS = ("keras", "onnx", "tflite")

_TFLITE_VARIANTS = [
    ("tflite", "model.tflite"),
    ("tflite_float16", "model_float16.tflite"),
    ("tflite_int8", "model_int8.tflite"),
]


def _collect_images(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in READABLE_IMAGE_EXTS else []
    return [str(f) for f in list_images(p, recursive=True, exts=READABLE_IMAGE_EXTS)]


def _supported_extensions() -> str:
    return ", ".join(sorted(READABLE_IMAGE_EXTS))


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
    raise ConfigError(f"Unknown format: {fmt}")


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


def _infer_keras(model_path: Path, images: list[str], size: int, task: Task) -> list[list[np.ndarray]]:
    model = task.load_model(str(model_path))
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
        raise ExportError(
            "onnxruntime is required for ONNX inference. "
            "Install it with: pip install onnxruntime"
        ) from None
    try:
        sess = ort.InferenceSession(str(model_path))
        input_name = sess.get_inputs()[0].name
        return [
            [
                np.asarray(o[0], dtype=np.float32)
                for o in sess.run(None, {input_name: _load_image(p, size)})
            ]
            for p in images
        ]
    except RuntimeError as e:  # onnxruntime raises RuntimeError subclasses for a broken model
        raise ExportError(f"ONNX inference failed: {e}") from e


def _infer_tflite(model_path: Path, images: list[str], size: int) -> list[list[np.ndarray]]:
    try:
        import tensorflow as tf
    except ImportError:
        raise ExportError("tensorflow is required for TFLite inference.") from None
    try:
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
    except RuntimeError as e:  # the TFLite interpreter raises RuntimeError for a broken model
        raise ExportError(f"TFLite inference failed: {e}") from e


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
    conf: only for tasks that threshold detections (see ``Task.predict_options``) — minimum
        score for a detection to be reported (default: the run's ``detection.conf_threshold``).
    top_k: only for tasks that rank classes — classes reported per image (None = all).
    model_path: explicit model file; the format is taken from ``fmt``, which
        must be one of keras/onnx/tflite.

    Returns dict with keys:
        plan_only       – True when fmt == "plan"
        task            – the run's task name
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
    task = resolve_task(run_cfg)
    check_predict_options(task, conf, top_k)

    found = _collect_images(input_path)
    if not found:
        raise DatasetError(
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
        raise DatasetError(f"Cannot read image: {detail}")

    if model_path is not None:
        if fmt not in FORMATS:
            raise ConfigError("--model needs a single format (keras, onnx or tflite)")
        if not Path(model_path).is_file():
            raise NotFoundError(f"Model file not found: {model_path}")

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
                outputs_list = _infer_keras(path, images, size, task)
            elif f == "onnx":
                outputs_list = _infer_onnx(path, images, size)
            else:
                outputs_list = _infer_tflite(path, images, size)
            results = task.predict(images, outputs_list, run_cfg, class_names, conf, top_k)
            formats_run.append({"format": f, "model_path": str(path), "results": results})
        except ExportError as e:
            formats_skipped.append({"format": f, "reason": str(e)})

    return {
        "plan_only": False,
        "task": task.name,
        "experiment": run_dir.name,
        "run_dir": run_dir,
        "formats_run": formats_run,
        "formats_skipped": formats_skipped,
        "inputs_skipped": inputs_skipped,
    }


# ---------------------------------------------------------------------------
# Web inference — single image supplied as raw bytes
# ---------------------------------------------------------------------------

def _load_run_model(run_name: str):
    """Return (cfg, model, input_size) for a run's keras checkpoint."""
    run_dir = Path(resolve_run_dir(run_name))
    cfg = load_config(str(run_dir))
    checkpoint = _model_path(run_dir, "keras")
    if checkpoint is None:
        raise NotFoundError(f"No keras checkpoint found for run '{run_name}'")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model = resolve_task(cfg).load_model(str(checkpoint))
    return cfg, model, model.input_shape[1]


WEB_TOP_K = 5  # classes the WebUI lists by default


def _predict_from_input(
    cfg, model, arr: np.ndarray, conf: float | None = None, top_k: int | None = WEB_TOP_K
) -> dict:
    """Run the model on a batched ``(1, H, W, 3)`` float array and shape the
    result with the same builders the CLI uses."""
    task = resolve_task(cfg)
    check_predict_options(task, conf)
    preds = model.predict(arr, verbose=0)
    outputs = [np.asarray(o[0], dtype=np.float32) for o in (preds if isinstance(preds, (list, tuple)) else [preds])]
    return task.predict_one(outputs, cfg, conf, top_k)


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
            raise ConfigError(f"Unknown augmentation: {aug['name']}")
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
