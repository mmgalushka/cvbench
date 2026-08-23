"""Decode raw CenterNet-style model output into detections.

Peak-picking uses a 3x3 max-pool "soft NMS" — a cell survives only if its
score equals the pooled max in its neighborhood — via ``keras.ops.max_pool``.
Top-k selection and box construction happen in plain Python/NumPy afterward,
so no NMS op ever enters the exported graph (keeping the TFLite/ONNX/Hailo
export path clean).
"""
from __future__ import annotations

import keras
import numpy as np


def _peak_mask(heatmap, pool_size: int = 3):
    """Zero out every score that isn't a local max in its POOL_SIZE neighborhood."""
    pooled = keras.ops.max_pool(heatmap, pool_size=pool_size, strides=1, padding="same")
    keep = keras.ops.cast(keras.ops.equal(heatmap, pooled), heatmap.dtype)
    return heatmap * keep


def decode_batch(
    preds,
    num_classes: int,
    conf_threshold: float = 0.25,
    max_detections: int = 100,
) -> list[list[dict]]:
    """Decode a batch of raw model outputs into detections.

    Args:
        preds: (batch, G, G, C+4) raw model output — heatmap channels already
            sigmoid-activated (as the model produces them).
        num_classes: C.
        conf_threshold: minimum heatmap score to keep a detection.
        max_detections: top-k cap per image.

    Returns:
        A list of length ``batch``; each entry is a list of dicts
        ``{"class_id", "confidence", "x", "y", "w", "h"}`` with box
        coordinates normalized top-left xywh in [0, 1].
    """
    preds = keras.ops.convert_to_tensor(preds)
    heatmap = preds[..., :num_classes]
    size = preds[..., num_classes:num_classes + 2]
    offset = preds[..., num_classes + 2:num_classes + 4]

    peaks = _peak_mask(heatmap)

    peaks_np = keras.ops.convert_to_numpy(peaks)
    size_np = keras.ops.convert_to_numpy(size)
    offset_np = keras.ops.convert_to_numpy(offset)

    batch, grid, _, num_classes = peaks_np.shape
    results: list[list[dict]] = []

    for b in range(batch):
        scores = peaks_np[b].reshape(-1)
        k = min(max_detections, scores.size)
        # argpartition for the top-k, then sort just those descending.
        top_idx = np.argpartition(-scores, k - 1)[:k] if k < scores.size else np.arange(scores.size)
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        dets = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < conf_threshold:
                break  # sorted descending: everything after this also fails
            cy, cx, cls_id = np.unravel_index(idx, (grid, grid, num_classes))
            w, h = size_np[b, cy, cx]
            if w <= 0 or h <= 0:
                continue
            dx, dy = offset_np[b, cy, cx]
            cx_norm = (cx + dx) / grid
            cy_norm = (cy + dy) / grid
            x = float(np.clip(cx_norm - w / 2, 0.0, 1.0))
            y = float(np.clip(cy_norm - h / 2, 0.0, 1.0))
            dets.append({
                "class_id": int(cls_id),
                "confidence": score,
                "x": x,
                "y": y,
                "w": float(min(max(w, 0.0), 1.0 - x)),
                "h": float(min(max(h, 0.0), 1.0 - y)),
            })
        results.append(dets)

    return results
