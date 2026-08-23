"""YOLO dataset -> tf.data pipeline for the detection task.

Targets are encoded to a single ``(G, G, C + 5)`` tensor per image (G = grid
size, C = number of classes) so ``model.fit`` stays a plain single-input call:

    [..., :C]      heatmap   — per-class Gaussian-splatted object-center peaks
    [..., C]       width     — normalized box width  (0..1), at the center cell
    [..., C + 1]   height    — normalized box height (0..1), at the center cell
    [..., C + 2]   offset x  — sub-cell center offset (0..1), at the center cell
    [..., C + 3]   offset y  — sub-cell center offset (0..1), at the center cell
    [..., C + 4]   mask      — 1.0 at object-center cells, 0.0 elsewhere

The heatmap uses the standard CornerNet/CenterNet Gaussian-radius formula so a
missed-by-one-pixel peak still contributes a partial loss signal instead of a
hard 0/1 miss. Size/offset/mask are exact-cell-only (no splatting) since they
are regression targets, masked out everywhere but the object's own cell.

Images are stretch-resized to the model's input size — no aspect-ratio
preservation. Letterboxing is deferred (see issue #50 design notes). An empty
or missing label file is a valid hard negative: every image under
``images/<split>`` is included even when its ``.txt`` has no boxes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from cvbench.core.config import CVBenchConfig
from cvbench.datasets.layout import list_images, read_yolo_boxes, yolo_label_dir


def _gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CornerNet/CenterNet radius: the largest radius whose Gaussian keeps IoU
    with the ground-truth box >= min_overlap under three overlap scenarios."""
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / 2

    a2 = 4.0
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))
    r2 = (b2 + sq2) / 2

    a3 = 4.0 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def _gaussian2d(shape: tuple[int, int], sigma: float) -> np.ndarray:
    m, n = [(s - 1.0) / 2.0 for s in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def _draw_gaussian(channel: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Max-splat a Gaussian peak of the given radius onto CHANNEL, in place."""
    diameter = 2 * radius + 1
    gaussian = _gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    height, width = channel.shape
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)

    masked_channel = channel[cy - top: cy + bottom, cx - left: cx + right]
    masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
    if min(masked_channel.shape) > 0 and min(masked_gaussian.shape) > 0:
        np.maximum(masked_channel, masked_gaussian, out=masked_channel)


def encode_target(
    boxes: list[tuple[int, tuple[float, float, float, float]]],
    num_classes: int,
    grid_size: int,
) -> np.ndarray:
    """Encode BOXES (as returned by ``read_yolo_boxes``) into a (G, G, C+5) target.

    BOXES entries are ``(class_id, (x, y, w, h))`` — normalized, top-left
    origin, exactly the shape ``cvbench.datasets.layout.read_yolo_boxes``
    returns. An empty list encodes to an all-zero target (a valid hard
    negative — every heatmap cell is a true negative and the mask channel is
    all zero, so no size/offset loss is computed for this image).
    """
    G, C = grid_size, num_classes
    target = np.zeros((G, G, C + 5), dtype=np.float32)
    for cls_id, (x, y, w, h) in boxes:
        if not (0 <= cls_id < C) or w <= 0 or h <= 0:
            continue
        cx, cy = (x + w / 2) * G, (y + h / 2) * G
        cx_i, cy_i = int(cx), int(cy)
        if not (0 <= cx_i < G and 0 <= cy_i < G):
            continue
        radius = max(0, int(_gaussian_radius(h * G, w * G)))
        _draw_gaussian(target[:, :, cls_id], cx_i, cy_i, radius)
        target[cy_i, cx_i, C] = w
        target[cy_i, cx_i, C + 1] = h
        target[cy_i, cx_i, C + 2] = cx - cx_i
        target[cy_i, cx_i, C + 3] = cy - cy_i
        target[cy_i, cx_i, C + 4] = 1.0
    return target


def build_detection_dataset(
    split_dir: str,
    ds_root: str,
    class_names: list[str],
    cfg: CVBenchConfig,
    training: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data pipeline yielding (image, target) pairs for one YOLO split.

    Args:
        split_dir: e.g. ``<ds_root>/images/train``.
        ds_root: the YOLO dataset root (contains images/, labels/, data.yaml).
        class_names: ordered class list (index = heatmap channel).
        cfg: resolved experiment config.
        training: if True, shuffle and repeat.

    Returns:
        Batched, prefetched tf.data.Dataset yielding
        (image (size, size, 3) float32, target (G, G, C+5) float32).
    """
    split_path = Path(split_dir)
    label_dir = yolo_label_dir(split_path, Path(ds_root))

    image_paths = [str(p) for p in list_images(split_path)]
    label_paths = [str(label_dir / f"{Path(p).stem}.txt") for p in image_paths]

    size = cfg.model.input_size
    grid_size = size // cfg.detection.grid_stride
    num_classes = len(class_names)
    batch = cfg.data.batch_size

    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    if training:
        ds = ds.shuffle(
            max(len(image_paths), 1), seed=cfg.training.seed, reshuffle_each_iteration=True
        )

    def _encode(label_path_bytes):
        boxes = read_yolo_boxes(Path(label_path_bytes.decode("utf-8")))
        return encode_target(boxes, num_classes, grid_size)

    def _load(img_path, label_path):
        img_bytes = tf.io.read_file(img_path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))  # stretch resize; no letterboxing
        img = tf.cast(img, tf.float32)
        img.set_shape((size, size, 3))

        target = tf.numpy_function(_encode, [label_path], tf.float32)
        target.set_shape((grid_size, grid_size, num_classes + 5))
        return img, target

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch)
    if training:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)
