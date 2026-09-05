"""YOLO dataset -> tf.data pipeline for the detection task.

Targets are encoded to one ``(Gs, Gs, A * (6 + C))`` tensor per scale
(``Gs`` = grid size at that scale, ``A`` = anchors per scale, ``C`` = number
of classes) — one tensor per entry in ``cfg.detection.strides`` — so
``model.fit`` sees ``y`` as a tuple matching the model's list of per-scale
outputs. Per anchor, per cell, the 6 + C target channels are:

    [0]        tx        — sigmoid-space sub-cell center offset x (0..1)
    [1]        ty        — sigmoid-space sub-cell center offset y (0..1)
    [2]        tw        — log(box_w / anchor_w)
    [3]        th        — log(box_h / anchor_h)
    [4]        obj       — 1.0 at the assigned (anchor, cell), else 0.0
    [5]        ignore    — 1.0 at a non-assigned anchor whose *shape* overlaps
                            a ground-truth box above
                            ``cfg.detection.ignore_iou_threshold`` — excluded
                            from the objectness loss entirely (neither a
                            positive nor a negative), so a near-miss anchor
                            isn't punished for not being the chosen one.
    [6:6+C]    class      — one-hot at the assigned (anchor, cell)

The predicted tensor (``model.py``'s output) is one channel narrower per
anchor (5 + C — no ``ignore`` channel, since that's a training-time-only
target concept); see ``losses.py``/``decode.py`` for the reshape boundary.

Each ground-truth box is assigned to the single best-IoU anchor across *all*
scales (shape-only IoU — see ``anchors.py``), the standard YOLOv2/v3 scheme.

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
from cvbench.detection.anchors import _iou_wh


def encode_targets(
    boxes: list[tuple[int, tuple[float, float, float, float]]],
    num_classes: int,
    anchors: list,
    strides: list[int],
    input_size: int,
    ignore_iou_threshold: float = 0.5,
) -> list[np.ndarray]:
    """Encode BOXES into one ``(Gs, Gs, A, 6 + C)`` target array per scale.

    BOXES entries are ``(class_id, (x, y, w, h))`` — normalized, top-left
    origin, exactly the shape ``cvbench.datasets.layout.read_yolo_boxes``
    returns. ANCHORS is ``cfg.detection.anchors``: a list of length
    ``len(strides)``, each a list of ``[w, h]`` pairs. An empty BOXES list
    encodes to all-zero targets (a valid hard negative).
    """
    C = num_classes
    grids = [input_size // s for s in strides]
    targets = [
        np.zeros((g, g, len(anchors[si]), 6 + C), dtype=np.float32)
        for si, g in enumerate(grids)
    ]

    # Flatten anchors across every scale for a single global best-IoU search.
    flat = [
        (si, ai, aw, ah)
        for si, scale_anchors in enumerate(anchors)
        for ai, (aw, ah) in enumerate(scale_anchors)
    ]
    anchor_arr = np.array([(aw, ah) for _, _, aw, ah in flat], dtype=np.float64)

    for cls_id, (x, y, w, h) in boxes:
        if not (0 <= cls_id < C) or w <= 0 or h <= 0:
            continue
        cx, cy = x + w / 2, y + h / 2
        ious = _iou_wh(np.array([w, h]), anchor_arr)
        best_idx = int(np.argmax(ious))
        best_si, best_ai, aw, ah = flat[best_idx]

        g = grids[best_si]
        gx_i = min(max(int(cx * g), 0), g - 1)
        gy_i = min(max(int(cy * g), 0), g - 1)
        tx, ty = cx * g - gx_i, cy * g - gy_i
        tw, th = float(np.log(w / aw)), float(np.log(h / ah))

        t = targets[best_si]
        t[gy_i, gx_i, best_ai, 0] = tx
        t[gy_i, gx_i, best_ai, 1] = ty
        t[gy_i, gx_i, best_ai, 2] = tw
        t[gy_i, gx_i, best_ai, 3] = th
        t[gy_i, gx_i, best_ai, 4] = 1.0  # obj
        t[gy_i, gx_i, best_ai, 5] = 0.0  # not ignored — this is the positive
        t[gy_i, gx_i, best_ai, 6 + cls_id] = 1.0

        # Ignore mask: every other anchor (any scale) whose *shape* overlaps
        # this box above the threshold gets excluded from the objectness
        # loss at this box's center cell in that scale — unless that cell is
        # already someone else's positive assignment.
        for idx, (si, ai, _aw2, _ah2) in enumerate(flat):
            if idx == best_idx or ious[idx] <= ignore_iou_threshold:
                continue
            g2 = grids[si]
            gx2_i = min(max(int(cx * g2), 0), g2 - 1)
            gy2_i = min(max(int(cy * g2), 0), g2 - 1)
            if targets[si][gy2_i, gx2_i, ai, 4] == 0.0:
                targets[si][gy2_i, gx2_i, ai, 5] = 1.0

    return targets


def build_detection_dataset(
    split_dir: str,
    ds_root: str,
    class_names: list[str],
    cfg: CVBenchConfig,
    training: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data pipeline yielding (image, targets) pairs for one YOLO
    split, where TARGETS is a tuple of per-scale target tensors (see
    ``encode_targets``), one per ``cfg.detection.strides`` entry.

    Args:
        split_dir: e.g. ``<ds_root>/images/train``.
        ds_root: the YOLO dataset root (contains images/, labels/, data.yaml).
        class_names: ordered class list (index = one-hot class channel).
        cfg: resolved experiment config — ``cfg.detection.anchors`` must
            already be resolved (see ``detection/anchors.py::resolve_anchors``).
        training: if True, shuffle and repeat.

    Returns:
        Batched, prefetched tf.data.Dataset yielding
        (image (size, size, 3) float32, tuple of per-scale target tensors).
    """
    split_path = Path(split_dir)
    label_dir = yolo_label_dir(split_path, Path(ds_root))

    image_paths = [str(p) for p in list_images(split_path)]
    label_paths = [str(label_dir / f"{Path(p).stem}.txt") for p in image_paths]

    size = cfg.model.input_size
    num_classes = len(class_names)
    strides = cfg.detection.strides
    anchors = cfg.detection.anchors
    ignore_iou = cfg.detection.ignore_iou_threshold
    grids = [size // s for s in strides]
    channels = [len(anchors[i]) * (6 + num_classes) for i in range(len(strides))]
    batch = cfg.data.batch_size

    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    if training:
        ds = ds.shuffle(
            max(len(image_paths), 1), seed=cfg.training.seed, reshuffle_each_iteration=True
        )

    def _encode(label_path_bytes):
        boxes = read_yolo_boxes(Path(label_path_bytes.decode("utf-8")))
        targets = encode_targets(boxes, num_classes, anchors, strides, size, ignore_iou)
        return [t.reshape(t.shape[0], t.shape[1], -1) for t in targets]

    def _load(img_path, label_path):
        img_bytes = tf.io.read_file(img_path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))  # stretch resize; no letterboxing
        img = tf.cast(img, tf.float32)
        img.set_shape((size, size, 3))

        raw_targets = tf.numpy_function(_encode, [label_path], [tf.float32] * len(strides))
        targets = []
        for t, g, ch in zip(raw_targets, grids, channels):
            t.set_shape((g, g, ch))
            targets.append(t)
        return img, tuple(targets)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch)
    if training:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)
