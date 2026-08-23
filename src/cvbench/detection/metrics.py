"""Detection metrics: IoU, per-class AP, mAP@50, precision/recall at a threshold.

Pure NumPy — no TensorFlow/Keras dependency, so this module's tests run in the
fast (non-``tf``-marked) lane.

Two distinct matching passes are used, deliberately:

* **Per-class matching** (``compute_detection_metrics``) — the standard mAP
  definition: a prediction only matches ground truth of its *own* class. This
  is what ``ap``/``map50`` are computed from.
* **Cross-class matching** (``bucket_samples``) — a separate, simplified pass
  used only to build the human-browsable sample gallery. It matches
  predictions to the nearest ground-truth box *regardless of class* so a
  spatially-correct-but-wrong-class prediction can be labeled "confusion"
  instead of being silently split into an unmatched FP and an unmatched FN.
  It does not feed into the AP/mAP numbers.
"""
from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]  # normalized top-left (x, y, w, h)

_MAX_SAMPLES_PER_BUCKET = 20


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two top-left xywh boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """All-point interpolated AP — area under the monotone precision envelope."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match(
    gt_boxes_by_image: dict[int, list[Box]],
    preds: list[tuple[int, float, Box]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily match PREDS (already in the caller's desired order) against
    GT_BOXES_BY_IMAGE, one class at a time. Returns (tp, fp) boolean-ish
    arrays aligned to PREDS. Each ground-truth box can be claimed at most once.
    """
    claimed = {img_idx: [False] * len(boxes) for img_idx, boxes in gt_boxes_by_image.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (img_idx, _conf, box) in enumerate(preds):
        boxes = gt_boxes_by_image.get(img_idx, [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(boxes):
            if claimed[img_idx][j]:
                continue
            cur = iou(box, gt_box)
            if cur > best_iou:
                best_iou, best_j = cur, j
        if best_j >= 0 and best_iou >= iou_threshold:
            claimed[img_idx][best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def compute_detection_metrics(
    ground_truths: list[list[dict]],
    predictions: list[list[dict]],
    class_names: list[str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Compute per-class AP/P/R/F1, mAP@50, and aggregate TP/FP/FN counts.

    Args:
        ground_truths: per image, a list of ``{"class_id", "box"}``.
        predictions: per image, a list of ``{"class_id", "confidence", "box"}``
            — the full ranked candidate list (not pre-filtered by threshold;
            AP needs the whole precision-recall curve).
        class_names: ordered class list.
        iou_threshold: IoU at which a prediction counts as matching a GT box.
        conf_threshold: minimum confidence for the fixed operating-point
            precision/recall/F1/counts (AP itself is threshold-independent).

    Returns:
        {"per_class": {cls: {"ap", "precision", "recall", "f1", "support"}},
         "map50": float | None, "counts": {"tp", "fp", "fn"}}
    """
    per_class: dict[str, dict] = {}
    aps: list[float] = []
    total_tp = total_fp = total_fn = 0

    for cls_id, cls_name in enumerate(class_names):
        gt_by_image = {
            img_idx: [g["box"] for g in gts if g["class_id"] == cls_id]
            for img_idx, gts in enumerate(ground_truths)
        }
        n_gt = sum(len(b) for b in gt_by_image.values())

        all_preds = [
            (img_idx, d["confidence"], d["box"])
            for img_idx, dets in enumerate(predictions)
            for d in dets if d["class_id"] == cls_id
        ]
        all_preds.sort(key=lambda p: -p[1])

        ap = None
        if n_gt > 0:
            tp, fp = _match(gt_by_image, all_preds, iou_threshold)
            tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
            recalls = tp_cum / n_gt
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = _average_precision(recalls, precisions) if len(all_preds) else 0.0
            aps.append(ap)

        # Separate matching pass restricted to the fixed operating threshold,
        # for the summary precision/recall/F1/counts.
        thr_preds = [p for p in all_preds if p[1] >= conf_threshold]
        thr_tp, thr_fp = (
            _match(gt_by_image, thr_preds, iou_threshold) if thr_preds else (np.array([]), np.array([]))
        )
        tp_sum, fp_sum = int(thr_tp.sum()), int(thr_fp.sum())
        fn_sum = max(n_gt - tp_sum, 0)

        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls_name] = {
            "ap": round(ap, 4) if ap is not None else None,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": n_gt,
        }
        total_tp += tp_sum
        total_fp += fp_sum
        total_fn += fn_sum

    map50 = round(float(np.mean(aps)), 4) if aps else None

    return {
        "per_class": per_class,
        "map50": map50,
        "counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    }


def _box_dict(class_id: int, class_names: list[str], box: Box) -> dict:
    x, y, w, h = box
    return {
        "class_id": class_id,
        "class": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
        "x": x, "y": y, "w": w, "h": h,
    }


def bucket_samples(
    image_paths: list[str],
    ground_truths: list[list[dict]],
    predictions: list[list[dict]],
    class_names: list[str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    max_per_bucket: int = _MAX_SAMPLES_PER_BUCKET,
) -> list[dict]:
    """Bucket each image's predictions/ground-truth into TP / FP / FN / confusion
    for the WebUI's clickable gallery — see the module docstring for why this
    is a class-agnostic pass, separate from the AP-driving per-class matching.
    """
    bucket_counts = {"tp": 0, "fp": 0, "fn": 0, "confusion": 0}
    samples: list[dict] = []

    def _add_sample(kind: str, path: str, gt: list[dict], pred: list[dict]) -> None:
        bucket_counts[kind] += 1
        if bucket_counts[kind] <= max_per_bucket:
            samples.append({"path": path, "bucket": kind, "gt": gt, "pred": pred})

    for img_idx, path in enumerate(image_paths):
        gts = ground_truths[img_idx]
        dets = sorted(
            (d for d in predictions[img_idx] if d["confidence"] >= conf_threshold),
            key=lambda d: -d["confidence"],
        )
        claimed = [False] * len(gts)
        gt_dicts = [_box_dict(g["class_id"], class_names, g["box"]) for g in gts]

        for d in dets:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if claimed[j]:
                    continue
                cur = iou(d["box"], g["box"])
                if cur > best_iou:
                    best_iou, best_j = cur, j
            if best_j >= 0 and best_iou >= iou_threshold:
                claimed[best_j] = True
                kind = "tp" if gts[best_j]["class_id"] == d["class_id"] else "confusion"
            else:
                kind = "fp"
            _add_sample(kind, path, gt_dicts, [_box_dict(d["class_id"], class_names, d["box"])])

        for j, g in enumerate(gts):
            if not claimed[j]:
                _add_sample("fn", path, [_box_dict(g["class_id"], class_names, g["box"])], [])

    return samples
