from __future__ import annotations

import base64
import io
from pathlib import Path

import keras
import numpy as np


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def run_prediction(checkpoint: str, input_path: str) -> list[dict]:
    """Run inference on an image or folder of images.

    Returns a list of dicts with keys:
        filename    – image file name (not full path)
        class_index – predicted class index
        confidence  – probability of the predicted class (0.0–1.0)

    Raises ValueError when no images are found at input_path.
    """
    import tensorflow as tf

    model = keras.saving.load_model(checkpoint)
    size = model.input_shape[1]

    paths = _collect_images(input_path)
    if not paths:
        raise ValueError(f"No images found at: {input_path}")

    results = []
    for img_path in paths:
        img = tf.keras.utils.load_img(img_path, target_size=(size, size))
        arr = tf.keras.utils.img_to_array(img)[None]  # (1, H, W, 3)
        probs = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        results.append({
            "filename": Path(img_path).name,
            "class_index": top_idx,
            "confidence": confidence,
        })

    return results


def predict_image(run_name: str, image_bytes: bytes) -> dict:
    """Run inference on a single image supplied as raw bytes.

    Returns:
        class_name  – predicted class label
        class_index – predicted class index
        confidence  – probability of the top class
        top_k       – list of {class_name, confidence} sorted by confidence desc
    """
    from cvbench.core.runs import resolve_run_dir
    from cvbench.core.config import load_config

    run_dir = resolve_run_dir(run_name)
    cfg = load_config(run_dir)
    class_names = cfg.data.classes

    checkpoint = _find_checkpoint(Path(run_dir))
    if checkpoint is None:
        raise ValueError(f"No model checkpoint found for run '{run_name}'")

    model = keras.saving.load_model(str(checkpoint))
    size = model.input_shape[1]

    arr = _bytes_to_input(image_bytes, size)
    probs = model.predict(arr, verbose=0)[0]

    return _build_result(probs, class_names)


def predict_augmented(run_name: str, image_bytes: bytes, augmentations: list[dict]) -> dict:
    """Apply a sequence of augmentations to an image then run inference.

    Args:
        run_name:     experiment run name (used to locate checkpoint + class list)
        image_bytes:  raw image bytes (any PIL-readable format)
        augmentations: list of {name: str, params: dict} applied in order

    Returns same keys as predict_image plus:
        augmented_image_b64 – base64 PNG of the augmented image
    """
    import cvbench.augmentations as aug_mod
    from cvbench.core.runs import resolve_run_dir
    from cvbench.core.config import load_config

    run_dir = resolve_run_dir(run_name)
    cfg = load_config(run_dir)
    class_names = cfg.data.classes

    checkpoint = _find_checkpoint(Path(run_dir))
    if checkpoint is None:
        raise ValueError(f"No model checkpoint found for run '{run_name}'")

    model = keras.saving.load_model(str(checkpoint))
    size = model.input_shape[1]

    img_arr = _bytes_to_numpy(image_bytes, size)  # (H, W, C) uint8

    for aug in augmentations:
        fn = getattr(aug_mod, aug["name"], None)
        if fn is None:
            raise ValueError(f"Unknown augmentation: {aug['name']}")
        img_arr = fn(img_arr, **aug.get("params", {}))

    augmented_b64 = _numpy_to_base64_png(img_arr)

    arr = img_arr.astype(np.float32)[None]  # (1, H, W, C)
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
    """Load image bytes → (1, H, W, C) float32 array ready for model.predict()."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.float32)[None]


def _bytes_to_numpy(image_bytes: bytes, size: int) -> np.ndarray:
    """Load image bytes → (H, W, C) uint8 numpy array."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.uint8)


def _numpy_to_base64_png(arr: np.ndarray) -> str:
    """Convert (H, W, C) uint8 numpy array → base64-encoded PNG string."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _find_checkpoint(run_dir: Path) -> Path | None:
    for pattern in ("best_model.keras", "*.keras", "best_model.h5", "*.h5"):
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _collect_images(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in _IMAGE_EXTS else []
    return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in _IMAGE_EXTS)
