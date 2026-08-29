"""Detection metrics, framed as two decomposed sub-problems.

Pure NumPy — no TensorFlow/Keras dependency, so this module's tests run in the
fast (non-``tf``-marked) lane.

A detector has to get two things right, and they are evaluated separately:

* **Localization** — are the boxes in the right place? Measured with IoU:
  ``localization_metrics`` reports the mean IoU of matched boxes plus AP at
  IoU 0.50 / 0.75 and a recall-vs-IoU sweep.
* **Classification** — given a box is in the right place, is the label right?
  Measured with an ``(N+1)x(N+1)`` confusion matrix over the classes plus a
  ``background`` row/column (``detection_confusion``), exactly like the
  classification task's confusion matrix. Missed ground truth lands in the
  ``background`` column, spurious predictions in the ``background`` row.

Both views, and the WebUI sample gallery, are derived from **one** greedy
class-agnostic prediction->ground-truth matching pass (``match_detections``) at
a fixed IoU floor and confidence threshold. The only thing that keeps its own
matching is per-class AP (``_per_class_ap``) — a ranked, threshold-free,
same-class-only match is what the AP definition requires.

mAP@50 is still computed and reported, but as a secondary benchmark number
rather than the headline.
"""
from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]  # normalized top-left (x, y, w, h)

_MAX_SAMPLES_PER_CELL = 20
_BACKGROUND = "background"


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


# ---------------------------------------------------------------------------
# Shared matching pass
# ---------------------------------------------------------------------------

def match_detections(
    image_paths: list[str],
    ground_truths: list[list[dict]],
    predictions: list[list[dict]],
    class_names: list[str],
    iou_floor: float = 0.5,
    conf_threshold: float = 0.25,
) -> list[dict]:
    """Greedy class-agnostic prediction->ground-truth matching, one image at a time.

    For each image: keep predictions with ``confidence >= conf_threshold``,
    sort them by descending confidence, and greedily assign each to the
    highest-IoU *unclaimed* ground-truth box in that image (regardless of
    class). A prediction whose best IoU is ``>= iou_floor`` is *matched*;
    otherwise it is a background false positive. Each ground-truth box is
    claimed at most once; unclaimed ground truth is a miss.

    Returns one dict per image (every image, including empty ones — callers
    that only want non-empty images filter downstream)::

        {"path": str,
         "gt":   [{"class_id", "box", "matched_pred": int | None}],
         "pred": [{"class_id", "confidence", "box",
                   "matched_gt": int | None, "iou": float}]}

    ``pred[i]["iou"]`` is the IoU with the box it was compared against (0.0 when
    there was no unclaimed box left); ``matched_gt`` / ``matched_pred`` are the
    index of the counterpart within the same image's list, or ``None``.
    """
    matches: list[dict] = []

    for img_idx, path in enumerate(image_paths):
        gts = ground_truths[img_idx]
        dets = sorted(
            (d for d in predictions[img_idx] if d["confidence"] >= conf_threshold),
            key=lambda d: -d["confidence"],
        )

        claimed = [False] * len(gts)
        gt_out = [
            {"class_id": g["class_id"], "box": g["box"], "matched_pred": None}
            for g in gts
        ]
        pred_out: list[dict] = []

        for pred_idx, d in enumerate(dets):
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if claimed[j]:
                    continue
                cur = iou(d["box"], g["box"])
                if cur > best_iou:
                    best_iou, best_j = cur, j

            matched_gt = None
            if best_j >= 0 and best_iou >= iou_floor:
                claimed[best_j] = True
                matched_gt = best_j
                gt_out[best_j]["matched_pred"] = pred_idx

            pred_out.append({
                "class_id": d["class_id"],
                "confidence": d["confidence"],
                "box": d["box"],
                "matched_gt": matched_gt,
                "iou": best_iou,
            })

        matches.append({"path": path, "gt": gt_out, "pred": pred_out})

    return matches


# ---------------------------------------------------------------------------
# Localization: IoU, AP@50 / AP@75, recall-vs-IoU
# ---------------------------------------------------------------------------

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


def _per_class_ap(
    ground_truths: list[list[dict]],
    predictions: list[list[dict]],
    class_names: list[str],
    iou_threshold: float,
) -> tuple[dict[str, float | None], float | None]:
    """All-point interpolated AP per class at a single IoU threshold, plus the
    mean over classes that have ground truth (``None`` when none do).

    Classes with no ground truth anywhere get ``ap = None`` and are excluded
    from the mean — this is the standard mAP convention.
    """
    aps_by_class: dict[str, float | None] = {}
    aps: list[float] = []

    for cls_id, cls_name in enumerate(class_names):
        gt_by_image = {
            img_idx: [g["box"] for g in gts if g["class_id"] == cls_id]
            for img_idx, gts in enumerate(ground_truths)
        }
        n_gt = sum(len(b) for b in gt_by_image.values())
        if n_gt == 0:
            aps_by_class[cls_name] = None
            continue

        all_preds = [
            (img_idx, d["confidence"], d["box"])
            for img_idx, dets in enumerate(predictions)
            for d in dets if d["class_id"] == cls_id
        ]
        all_preds.sort(key=lambda p: -p[1])

        if not all_preds:
            aps_by_class[cls_name] = 0.0
            aps.append(0.0)
            continue

        tp, fp = _match(gt_by_image, all_preds, iou_threshold)
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recalls = tp_cum / n_gt
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = _average_precision(recalls, precisions)
        aps_by_class[cls_name] = ap
        aps.append(ap)

    mean_ap = float(np.mean(aps)) if aps else None
    return aps_by_class, mean_ap


def localization_metrics(
    image_matches: list[dict],
    ground_truths: list[list[dict]],
    predictions: list[list[dict]],
    class_names: list[str],
) -> dict:
    """The localization view — how well-placed are the boxes.

    Returns::

        {"mean_iou": float | None,          # mean IoU over matched pred/GT pairs
         "ap50": float | None,              # mAP at IoU 0.50 (== the report's map50)
         "ap75": float | None,              # mAP at IoU 0.75 (strict localization)
         "recall_sweep": {"0.5": r, "0.75": r, "0.9": r}}

    ``recall_sweep`` is confidence-order-independent: for every ground-truth box
    it takes the best IoU to *any* above-threshold prediction in that image
    (class-agnostic), then reports the fraction ``>= t``.
    """
    ious = [
        p["iou"]
        for m in image_matches
        for p in m["pred"]
        if p["matched_gt"] is not None
    ]
    mean_iou = float(np.mean(ious)) if ious else None

    _, ap50 = _per_class_ap(ground_truths, predictions, class_names, 0.5)
    _, ap75 = _per_class_ap(ground_truths, predictions, class_names, 0.75)

    best_by_gt: list[float] = []
    for m in image_matches:
        pred_boxes = [p["box"] for p in m["pred"]]
        for g in m["gt"]:
            best_by_gt.append(max((iou(g["box"], pb) for pb in pred_boxes), default=0.0))

    def _recall(t: float) -> float | None:
        if not best_by_gt:
            return None
        return float(np.mean([1.0 if v >= t else 0.0 for v in best_by_gt]))

    return {
        "mean_iou": mean_iou,
        "ap50": ap50,
        "ap75": ap75,
        "recall_sweep": {"0.5": _recall(0.5), "0.75": _recall(0.75), "0.9": _recall(0.9)},
    }


# ---------------------------------------------------------------------------
# Classification: (N+1)x(N+1) confusion matrix with a background row/column
# ---------------------------------------------------------------------------

def detection_confusion(image_matches: list[dict], class_names: list[str]) -> dict:
    """Build the detection confusion matrix from the shared matching pass.

    An ``(N+1)x(N+1)`` grid; index ``N`` is ``"background"``:

    * matched pair  -> ``cm[gt_class][pred_class] += 1``
      (the diagonal is "located AND classified correctly"; off-diagonal within
      the first N rows/cols is class confusion on a well-placed box)
    * missed ground truth -> ``cm[gt_class][background] += 1``
    * spurious prediction -> ``cm[background][pred_class] += 1``

    Returns ``{"classes": [...class_names, "background"], "matrix": [[...]],
    "matrix_normalized": [[...]]}`` where ``matrix_normalized`` is row-normalized
    (rows that sum to zero stay zero).
    """
    n = len(class_names)
    cm = np.zeros((n + 1, n + 1), dtype=int)

    for m in image_matches:
        for p in m["pred"]:
            if p["matched_gt"] is None:
                cm[n, p["class_id"]] += 1
            else:
                gt_cls = m["gt"][p["matched_gt"]]["class_id"]
                cm[gt_cls, p["class_id"]] += 1
        for g in m["gt"]:
            if g["matched_pred"] is None:
                cm[g["class_id"], n] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm, row_sums, out=np.zeros((n + 1, n + 1), dtype=float), where=row_sums > 0
    )

    return {
        "classes": [*class_names, _BACKGROUND],
        "matrix": cm.tolist(),
        "matrix_normalized": [[round(v, 4) for v in row] for row in normalized.tolist()],
    }


def _per_class_from_confusion(cm: np.ndarray, class_names: list[str]) -> dict[str, dict]:
    """Precision / recall / F1 / support per class, Ultralytics-style, from the
    ``(N+1)x(N+1)`` confusion matrix (the trailing background row/column
    contribute FP / FN but are not themselves a class)."""
    out: dict[str, dict] = {}
    for c, name in enumerate(class_names):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(cm[c, :].sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------

def compute_detection_metrics(
    ground_truths: list[list[dict]],
    predictions: list[list[dict]],
    class_names: list[str],
    image_matches: list[dict] | None = None,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> dict:
    """Assemble the localization + classification views into one dict.

    Args:
        ground_truths: per image, a list of ``{"class_id", "box"}``.
        predictions: per image, a list of ``{"class_id", "confidence", "box"}``
            — the full ranked candidate list (not pre-filtered by threshold;
            AP needs the whole precision-recall curve).
        class_names: ordered class list.
        image_matches: the output of ``match_detections`` — the single greedy
            pass everything but AP shares. When ``None`` it is computed here
            from ``(iou_threshold, conf_threshold)``.
        iou_threshold / conf_threshold: the matching thresholds; used to build
            ``image_matches`` when it is not supplied.

    Returns::

        {"per_class": {cls: {ap, ap75, precision, recall, f1, support}},
         "map50": float | None,             # secondary benchmark number
         "localization": {...},             # see localization_metrics
         "confusion_matrix": {...},         # see detection_confusion
         "counts": {"tp", "fp", "fn"}}      # confusion-matrix margins,
                                            # class-agnostic (matched pairs /
                                            # spurious preds / missed GT)
    """
    if image_matches is None:
        image_matches = match_detections(
            [str(i) for i in range(len(ground_truths))],
            ground_truths, predictions, class_names,
            iou_floor=iou_threshold, conf_threshold=conf_threshold,
        )

    loc = localization_metrics(image_matches, ground_truths, predictions, class_names)
    confusion = detection_confusion(image_matches, class_names)
    cm = np.array(confusion["matrix"])
    n = len(class_names)

    ap50_by_class, map50 = _per_class_ap(ground_truths, predictions, class_names, 0.5)
    ap75_by_class, _ = _per_class_ap(ground_truths, predictions, class_names, 0.75)

    prf = _per_class_from_confusion(cm, class_names)
    per_class = {
        name: {
            "ap": round(ap50_by_class[name], 4) if ap50_by_class[name] is not None else None,
            "ap75": round(ap75_by_class[name], 4) if ap75_by_class[name] is not None else None,
            **prf[name],
        }
        for name in class_names
    }

    counts = {
        "tp": int(cm[:n, :n].sum()),
        "fp": int(cm[n, :n].sum()),
        "fn": int(cm[:n, n].sum()),
    }

    return {
        "per_class": per_class,
        "map50": round(map50, 4) if map50 is not None else None,
        "localization": loc,
        "confusion_matrix": confusion,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# WebUI sample gallery
# ---------------------------------------------------------------------------

def _box_dict(
    class_id: int,
    class_names: list[str],
    box: Box,
    confidence: float | None = None,
    match: str | None = None,
    counterpart: str | None = None,
    iou_value: float | None = None,
) -> dict:
    x, y, w, h = box
    d = {
        "class_id": class_id,
        "class": class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id),
        "x": x, "y": y, "w": w, "h": h,
    }
    if confidence is not None:
        d["confidence"] = round(confidence, 4)
    if match is not None:
        d["match"] = match
    if counterpart is not None:
        d["counterpart"] = counterpart
    if iou_value is not None:
        d["iou"] = round(iou_value, 4)
    return d


def build_detection_samples(
    image_matches: list[dict],
    class_names: list[str],
    max_per_cell: int = _MAX_SAMPLES_PER_CELL,
) -> list[dict]:
    """Turn the shared matching pass into the WebUI's clickable gallery.

    One dict per image that has at least one ground-truth box or one
    above-threshold prediction::

        {"path",
         "cells":  [[true_class, pred_class], ...],   # confusion-matrix cells present
         "counts": {"matched", "fp", "fn"},
         "gt":   [{..., "match": "matched"|"background", "counterpart"?}],
         "pred": [{..., "confidence", "match": "matched"|"background",
                   "counterpart"?, "iou"?}]}

    ``cells`` uses the ``"background"`` sentinel for missed GT (``[cls,
    "background"]``) and spurious predictions (``["background", cls]``) so the
    confusion-matrix cell click on the UI side can filter directly.

    Images are capped per confusion-matrix cell (``max_per_cell``, counted in
    images); an image spanning several cells is kept once and consumes each of
    those cell counters.
    """
    kept_per_cell: dict[tuple[str, str], int] = {}
    samples: list[dict] = []

    for m in image_matches:
        gts, preds = m["gt"], m["pred"]
        if not gts and not preds:
            continue

        def _name(cid: int) -> str:
            return class_names[cid] if 0 <= cid < len(class_names) else str(cid)

        cells: set[tuple[str, str]] = set()
        counts = {"matched": 0, "fp": 0, "fn": 0}

        gt_boxes, pred_boxes = [], []

        for p in preds:
            if p["matched_gt"] is None:
                counts["fp"] += 1
                cells.add((_BACKGROUND, _name(p["class_id"])))
                pred_boxes.append(_box_dict(
                    p["class_id"], class_names, p["box"],
                    confidence=p["confidence"], match=_BACKGROUND,
                ))
            else:
                counts["matched"] += 1
                gt_cls = gts[p["matched_gt"]]["class_id"]
                cells.add((_name(gt_cls), _name(p["class_id"])))
                pred_boxes.append(_box_dict(
                    p["class_id"], class_names, p["box"],
                    confidence=p["confidence"], match="matched",
                    counterpart=_name(gt_cls), iou_value=p["iou"],
                ))

        for g in gts:
            if g["matched_pred"] is None:
                counts["fn"] += 1
                cells.add((_name(g["class_id"]), _BACKGROUND))
                gt_boxes.append(_box_dict(
                    g["class_id"], class_names, g["box"], match=_BACKGROUND,
                ))
            else:
                pred_cls = preds[g["matched_pred"]]["class_id"]
                gt_boxes.append(_box_dict(
                    g["class_id"], class_names, g["box"], match="matched",
                    counterpart=_name(pred_cls),
                ))

        cell_list = sorted(cells)
        if not any(kept_per_cell.get(c, 0) < max_per_cell for c in cell_list):
            continue
        for c in cell_list:
            kept_per_cell[c] = kept_per_cell.get(c, 0) + 1

        samples.append({
            "path": m["path"],
            "cells": [list(c) for c in cell_list],
            "counts": counts,
            "gt": gt_boxes,
            "pred": pred_boxes,
        })

    return samples
