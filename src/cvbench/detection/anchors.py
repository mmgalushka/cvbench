"""Anchor-box derivation for the YOLO-style detection head.

Anchors are derived once per dataset (via IoU-distance k-means over the
training split's ground-truth box shapes, as in the YOLOv2 paper — plain
Euclidean k-means on (w, h) is biased toward large boxes, since a large box's
squared-distance error dwarfs a small box's even when their IoUs with a given
anchor are identical) and then persisted into ``cfg.detection.anchors`` so
every subsequent build/resume/evaluate of a run reuses the exact same
anchors. Anchors must never be silently re-derived — that would invalidate
whatever the model already learned relative to the old ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cvbench.datasets.layout import list_images, read_yolo_boxes, yolo_label_dir


def _iou_wh(box: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """IoU of one (w, h) box against each row of CLUSTERS (both centered at
    a common origin, so IoU depends only on shape, not position)."""
    w, h = box
    cw, ch = clusters[:, 0], clusters[:, 1]
    inter = np.minimum(w, cw) * np.minimum(h, ch)
    union = w * h + cw * ch - inter
    return inter / np.maximum(union, 1e-12)


def kmeans_anchors(
    boxes: np.ndarray, k: int, seed: int = 42, max_iter: int = 300,
) -> np.ndarray:
    """IoU-distance k-means over (w, h) box shapes.

    Args:
        boxes: (N, 2) array of normalized (w, h) box shapes.
        k: number of anchors to return.
        seed: RNG seed — deterministic for a given BOXES/K/SEED.
        max_iter: safety cap; the loop also stops on assignment convergence.

    Returns:
        (k, 2) array of (w, h) anchors, sorted by ascending area.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    n = len(boxes)
    if n == 0:
        raise ValueError("Cannot derive anchors from zero boxes.")
    k = min(k, n)

    rng = np.random.RandomState(seed)
    clusters = boxes[rng.choice(n, size=k, replace=False)]

    prev_assignments = None
    for _ in range(max_iter):
        distances = np.stack([1.0 - _iou_wh(b, clusters) for b in boxes])
        assignments = distances.argmin(axis=1)
        if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
            break
        prev_assignments = assignments
        for c in range(k):
            members = boxes[assignments == c]
            if len(members) > 0:
                clusters[c] = np.median(members, axis=0)

    order = np.argsort(clusters[:, 0] * clusters[:, 1])
    return clusters[order]


def _collect_box_shapes(train_dir: str, ds_root: str) -> np.ndarray:
    """(w, h) of every ground-truth box in the training split's labels."""
    split_path = Path(train_dir)
    label_dir = yolo_label_dir(split_path, Path(ds_root))
    shapes = []
    for img_path in list_images(split_path):
        label_path = label_dir / f"{img_path.stem}.txt"
        for _cls_id, (_x, _y, w, h) in read_yolo_boxes(label_path):
            if w > 0 and h > 0:
                shapes.append((w, h))
    return np.array(shapes, dtype=np.float64) if shapes else np.zeros((0, 2))


def resolve_anchors(cfg) -> list:
    """Return ``cfg.detection.anchors`` as-is if already set, else derive and
    persist it into CFG.

    Anchors are split across ``cfg.detection.strides`` by area — the
    smaller-area anchors go to the smallest stride (highest-resolution
    feature map, best suited to small objects) and the largest-area anchors
    to the largest stride, matching the standard YOLO convention.

    Returns:
        A list of length ``len(cfg.detection.strides)``, each element a list
        of ``[w, h]`` pairs (``cfg.detection.anchors_per_scale`` per scale).
    """
    if cfg.detection.anchors:
        return cfg.detection.anchors

    strides = cfg.detection.strides
    per_scale = cfg.detection.anchors_per_scale
    total = per_scale * len(strides)

    shapes = _collect_box_shapes(cfg.data.train_dir, cfg.data.data_dir)
    if len(shapes) == 0:
        raise ValueError(
            f"No ground-truth boxes found under {cfg.data.train_dir} — "
            "cannot derive anchors."
        )

    clusters = kmeans_anchors(shapes, total, seed=cfg.training.seed or 42)
    if len(clusters) < total:
        # Fewer distinct box shapes than anchors requested (tiny dataset) —
        # pad by repeating the largest cluster rather than erroring.
        pad = np.repeat(clusters[-1:], total - len(clusters), axis=0)
        clusters = np.concatenate([clusters, pad], axis=0)
    # kmeans_anchors sorts by ascending area already; slice into per-scale
    # groups in stride order (smallest stride = smallest objects first).
    anchors = [
        clusters[i * per_scale:(i + 1) * per_scale].tolist()
        for i in range(len(strides))
    ]
    cfg.detection.anchors = anchors
    return anchors
