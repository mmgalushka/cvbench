"""Decode raw YOLO-style model output into detections.

The model emits raw logits (see ``model.py``) — sigmoid/exp are applied here,
in plain NumPy, along with per-anchor class-argmax scoring, confidence
thresholding, and greedy per-class NMS. None of this enters the exported
graph, keeping the TFLite/ONNX/Hailo export path clean.
"""
from __future__ import annotations

import numpy as np

from cvbench.detection.metrics import iou


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms(dets: list[dict], iou_threshold: float) -> list[dict]:
    """Greedy per-class NMS. DETS must already be sorted by descending
    confidence — each box is kept unless it overlaps a higher-confidence,
    already-kept box of the same class above IOU_THRESHOLD."""
    kept: list[dict] = []
    for d in dets:
        box = (d["x"], d["y"], d["w"], d["h"])
        if any(
            k["class_id"] == d["class_id"] and iou(box, (k["x"], k["y"], k["w"], k["h"])) > iou_threshold
            for k in kept
        ):
            continue
        kept.append(d)
    return kept


def _decode_scale(
    preds: np.ndarray, anchors: list, num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one scale's raw ``(B, G, G, A*(5+C))`` output into per-anchor
    ``(cx, cy, w, h, score, class_id)`` arrays, all shaped ``(B, G, G, A)``
    (``cx``/``cy``/``w``/``h``/``score`` float, ``class_id`` int)."""
    B, G, _, _ = preds.shape
    A = len(anchors)
    C = num_classes
    p = preds.reshape(B, G, G, A, 5 + C)

    tx, ty = _sigmoid(p[..., 0]), _sigmoid(p[..., 1])
    tw, th = p[..., 2], p[..., 3]
    obj = _sigmoid(p[..., 4])
    cls = _sigmoid(p[..., 5:])

    class_id = np.argmax(cls, axis=-1)
    cls_score = np.take_along_axis(cls, class_id[..., None], axis=-1)[..., 0]
    score = obj * cls_score

    rows, cols = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    anchor_w = np.array([a[0] for a in anchors])
    anchor_h = np.array([a[1] for a in anchors])

    cx = (tx + cols[None, :, :, None]) / G
    cy = (ty + rows[None, :, :, None]) / G
    w = np.exp(tw) * anchor_w[None, None, None, :]
    h = np.exp(th) * anchor_h[None, None, None, :]

    return cx, cy, w, h, score, class_id


def decode_batch(
    preds: list,
    num_classes: int,
    anchors: list,
    strides: list[int],
    conf_threshold: float = 0.25,
    max_detections: int = 100,
    nms_iou_threshold: float = 0.5,
) -> list[list[dict]]:
    """Decode a batch of raw model output into detections.

    Args:
        preds: list of per-scale raw model outputs (one per entry in
            STRIDES), each ``(batch, G, G, A*(5+C))``.
        num_classes: C.
        anchors: ``cfg.detection.anchors`` — list of length ``len(strides)``,
            each a list of ``[w, h]`` pairs.
        strides: ``cfg.detection.strides`` — zipped against PREDS/ANCHORS;
            the actual grid size is read from each tensor's own shape.
        conf_threshold: minimum ``obj * class`` score to keep a detection.
        max_detections: top-k cap per image (across all scales combined),
            applied before NMS.
        nms_iou_threshold: greedy per-class NMS threshold — a lower-confidence
            box is dropped once it overlaps a kept, higher-confidence box of
            the same class above this IoU.

    Returns:
        A list of length ``batch``; each entry is a list of dicts
        ``{"class_id", "confidence", "x", "y", "w", "h"}`` with box
        coordinates normalized top-left xywh in [0, 1].
    """
    batch = np.asarray(preds[0]).shape[0]
    per_image: list[list[dict]] = [[] for _ in range(batch)]

    for p, scale_anchors, _stride in zip(preds, anchors, strides):
        p = np.asarray(p)
        cx, cy, w, h, score, class_id = _decode_scale(p, scale_anchors, num_classes)

        for b in range(batch):
            for gy, gx, a in np.argwhere(score[b] >= conf_threshold):
                bw, bh = float(w[b, gy, gx, a]), float(h[b, gy, gx, a])
                x = float(np.clip(cx[b, gy, gx, a] - bw / 2, 0.0, 1.0))
                y = float(np.clip(cy[b, gy, gx, a] - bh / 2, 0.0, 1.0))
                per_image[b].append({
                    "class_id": int(class_id[b, gy, gx, a]),
                    "confidence": float(score[b, gy, gx, a]),
                    "x": x,
                    "y": y,
                    "w": float(min(max(bw, 0.0), 1.0 - x)),
                    "h": float(min(max(bh, 0.0), 1.0 - y)),
                })

    results = []
    for dets in per_image:
        dets.sort(key=lambda d: -d["confidence"])
        results.append(_nms(dets[:max_detections], nms_iou_threshold))
    return results
