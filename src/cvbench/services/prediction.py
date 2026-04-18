from __future__ import annotations

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


def _collect_images(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in _IMAGE_EXTS else []
    return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in _IMAGE_EXTS)
