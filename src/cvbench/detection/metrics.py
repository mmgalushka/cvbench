"""Detection metrics, framed as two decomposed sub-problems.

Pure NumPy — no TensorFlow/Keras dependency, so this module's tests run in the
fast (non-``tf``-marked) lane.

A detector has to get two things right, and they are evaluated separately:

* **Localization** — are the boxes in the right place? Measured with IoU:
  ``localization_metrics`` reports the mean IoU of matched boxes plus AP at
  IoU 0.50 / 0.75 and a recall-vs-IoU sweep.
* **Classification** — given a box is in the right place, is the label right?
  The headline view is a per-class *outcome* table (``detection_class_breakdown``):
  every ground-truth box is matched / class-confused / mis-located / missed, and
  spurious predictions are tallied by predicted class. The older
  ``(N+1)x(N+1)`` confusion matrix (``detection_confusion``) is still built —
  missed GT in the ``background`` column, spurious preds in the ``background``
  row — for datasets where class-for-class confusion is actually non-trivial.

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


# ---------------------------------------------------------------------------
# Per-class outcome breakdown (the detection tab's headline view)
# ---------------------------------------------------------------------------

# Minimum IoU for an unmatched same-class prediction sitting on a ground-truth
# box to read as "aimed at that object" rather than a box in empty space —
# separates ``mislocated`` / ``duplicate`` from ``spurious``.
_LOC_FLOOR = 0.1

# The four ways a ground-truth box can turn out. ``matched`` is the only good one.
_GT_OUTCOMES = ("matched", "confused", "mislocated", "missed")

# The two ways a leftover (unmatched, non-confused) prediction can turn out:
# ``duplicate`` lands on a real same-class object that another box already
# covers; ``spurious`` is a box in the background.
_EXTRA_PRED_OUTCOMES = ("duplicate", "spurious")

# A ``duplicate`` whose IoU with the kept (matched) box on the same object is
# ``>= _DUP_SUPPRESS_IOU`` would be removed by ordinary NMS at that threshold;
# below it the two boxes barely overlap, so it is a genuine second bad box
# (a regression problem, not a suppression one).
_DUP_SUPPRESS_IOU = 0.5


def _classify_outcomes(m: dict, loc_floor: float = _LOC_FLOOR) -> tuple[list[dict], list[dict]]:
    """Label every box in one matched image by what actually happened to it.

    Consumes a single ``match_detections`` image dict and returns
    ``(gt_outcomes, pred_outcomes)``, lists parallel to ``m["gt"]`` /
    ``m["pred"]``:

    * gt   -> ``{"outcome": "matched"|"confused"|"mislocated"|"missed",
                 "as_class": int | None}``  (the predicted class for
      ``confused``; the GT's own class for ``mislocated``; ``None`` for
      ``missed``)
    * pred -> ``{"outcome": "matched"|"confused"|"mislocated"|"duplicate"
                 |"spurious", "gt_class": int | None, "iou": float,
                 "rival_iou": float}``
      (``iou`` is the IoU with the box this prediction is really about — its
      matched/paired/overlapped GT — not the class-agnostic pass's
      unclaimed-only figure, which is 0 for a duplicate. ``rival_iou``, only
      meaningful for ``duplicate``, is the IoU with the kept box on the same
      object: high => ordinary NMS would drop this; low => a real second box.)

    The class-agnostic matching pass splits some real situations across two
    boxes; this reunites them:

    * ``mislocated`` — an unmatched GT and an unmatched same-class prediction
      overlapping it at ``>= loc_floor`` but below the match IoU floor.
    * ``duplicate`` — an unmatched same-class prediction overlapping *any* GT
      (already matched or not) at ``>= loc_floor``: the object is real and
      already accounted for, this is just a redundant box.

    Only a prediction that overlaps no same-class GT at all is ``spurious``.
    """
    gts, preds = m["gt"], m["pred"]
    gt_out: list[dict | None] = [None] * len(gts)
    pred_out: list[dict | None] = [None] * len(preds)
    consumed = [False] * len(preds)

    for pi, p in enumerate(preds):
        if p["matched_gt"] is not None:
            gc = gts[p["matched_gt"]]["class_id"]
            pred_out[pi] = {
                "outcome": "matched" if gc == p["class_id"] else "confused",
                "gt_class": gc,
                "iou": p["iou"],
            }

    for gj, g in enumerate(gts):
        mp = g["matched_pred"]
        if mp is not None:
            pc = preds[mp]["class_id"]
            gt_out[gj] = {
                "outcome": "matched" if pc == g["class_id"] else "confused",
                "as_class": pc,
            }
            continue

        best_iou, best_pi = loc_floor, -1
        for pi, p in enumerate(preds):
            if p["matched_gt"] is not None or consumed[pi]:
                continue
            if p["class_id"] != g["class_id"]:
                continue
            v = iou(p["box"], g["box"])
            if v >= best_iou:
                best_iou, best_pi = v, pi

        if best_pi >= 0:
            consumed[best_pi] = True
            gt_out[gj] = {"outcome": "mislocated", "as_class": g["class_id"]}
            pred_out[best_pi] = {
                "outcome": "mislocated", "gt_class": g["class_id"], "iou": best_iou,
            }
        else:
            gt_out[gj] = {"outcome": "missed", "as_class": None}

    for pi, p in enumerate(preds):
        if pred_out[pi] is not None:
            continue
        best, best_gj = 0.0, -1
        for gj, g in enumerate(gts):
            if g["class_id"] != p["class_id"]:
                continue
            v = iou(p["box"], g["box"])
            if v > best:
                best, best_gj = v, gj
        if best >= loc_floor:
            rival = 0.0
            if best_gj >= 0 and gts[best_gj]["matched_pred"] is not None:
                rival = iou(p["box"], preds[gts[best_gj]["matched_pred"]]["box"])
            pred_out[pi] = {
                "outcome": "duplicate", "gt_class": p["class_id"],
                "iou": best, "rival_iou": rival,
            }
        else:
            pred_out[pi] = {"outcome": "spurious", "gt_class": None, "iou": best}

    return gt_out, pred_out  # type: ignore[return-value]


def detection_class_breakdown(
    image_matches: list[dict],
    class_names: list[str],
    loc_floor: float = _LOC_FLOOR,
) -> dict:
    """Aggregate ``_classify_outcomes`` over every image into a per-class table.

    Returns::

        {"classes": [...class_names],
         "rows": {cls: {"instances", "matched", "confused", "mislocated",
                        "missed", "confused_as": {other_cls: n}}},
         "duplicate": {cls: n},              # redundant box on an already-covered object
         "duplicate_suppressible": {cls: n}, # ... subset NMS would drop (rival IoU high)
         "spurious": {cls: n},               # box in the background, no object there
         "loc_floor": float,
         "dup_suppress_iou": float}

    ``instances`` == ``matched + confused + mislocated + missed`` (every GT box
    of that class). ``duplicate`` / ``spurious`` are keyed by *predicted* class
    and are disjoint from the rows — an extra prediction has no ground truth of
    its own to belong to. ``duplicate_suppressible`` is the part of
    ``duplicate`` that overlaps the kept box on its object at
    ``>= dup_suppress_iou`` — i.e. plain NMS would remove it; the remainder is
    a genuine second box the regressor produced.
    """
    def _name(cid: int) -> str:
        return class_names[cid] if 0 <= cid < len(class_names) else str(cid)

    rows = {
        n: {k: 0 for k in ("instances", *_GT_OUTCOMES)} | {"confused_as": {}}
        for n in class_names
    }
    extra = {k: {n: 0 for n in class_names} for k in _EXTRA_PRED_OUTCOMES}
    dup_suppressible = {n: 0 for n in class_names}

    for m in image_matches:
        gt_out, pred_out = _classify_outcomes(m, loc_floor)
        for g, o in zip(m["gt"], gt_out, strict=True):
            row = rows.setdefault(
                _name(g["class_id"]),
                {k: 0 for k in ("instances", *_GT_OUTCOMES)} | {"confused_as": {}},
            )
            row["instances"] += 1
            row[o["outcome"]] += 1
            if o["outcome"] == "confused":
                as_name = _name(o["as_class"])
                row["confused_as"][as_name] = row["confused_as"].get(as_name, 0) + 1
        for p, o in zip(m["pred"], pred_out, strict=True):
            if o["outcome"] in extra:
                name = _name(p["class_id"])
                extra[o["outcome"]][name] = extra[o["outcome"]].get(name, 0) + 1
                if o["outcome"] == "duplicate" and o.get("rival_iou", 0.0) >= _DUP_SUPPRESS_IOU:
                    dup_suppressible[name] = dup_suppressible.get(name, 0) + 1

    return {
        "classes": list(class_names),
        "rows": rows,
        "duplicate": extra["duplicate"],
        "duplicate_suppressible": dup_suppressible,
        "spurious": extra["spurious"],
        "loc_floor": loc_floor,
        "dup_suppress_iou": _DUP_SUPPRESS_IOU,
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
    loc_floor: float = _LOC_FLOOR,
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
         "class_breakdown": {...},          # see detection_class_breakdown
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
    breakdown = detection_class_breakdown(image_matches, class_names, loc_floor)
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
        "class_breakdown": breakdown,
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
    outcome: str | None = None,
    rival_iou: float | None = None,
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
    if outcome is not None:
        d["outcome"] = outcome
    if rival_iou is not None:
        d["rival_iou"] = round(rival_iou, 4)
    return d


def build_detection_samples(
    image_matches: list[dict],
    class_names: list[str],
    max_per_cell: int = _MAX_SAMPLES_PER_CELL,
    loc_floor: float = _LOC_FLOOR,
) -> list[dict]:
    """Turn the shared matching pass into the WebUI's clickable gallery.

    One dict per image that has at least one ground-truth box or one
    above-threshold prediction::

        {"path",
         "cells":  [[true_class, pred_class], ...],   # confusion-matrix cells present
         "tags":   ["circle:matched", "square:missed", "triangle:duplicate", ...],
         "counts": {"matched", "fp", "fn"},
         "gt":   [{..., "match": "matched"|"background", "counterpart"?,
                   "outcome": "matched"|"confused"|"mislocated"|"missed"}],
         "pred": [{..., "confidence", "match": "matched"|"background",
                   "counterpart"?, "iou"?, "rival_iou"?, "outcome":
                   "matched"|"confused"|"mislocated"|"duplicate"|"spurious"}]}

    ``cells`` drives the (legacy) confusion-matrix cell click; ``tags`` drives
    the per-class breakdown table — a cell there filters the gallery to samples
    whose ``tags`` contains ``"<class>:<outcome>"`` (``<outcome>`` being any GT
    outcome, or ``duplicate`` / ``spurious`` for the extra-prediction strips).
    ``match`` / ``counterpart`` are kept verbatim for the confusion-matrix view;
    ``outcome`` is the finer label the breakdown and box styling read.

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

        gt_out, pred_out = _classify_outcomes(m, loc_floor)
        cells: set[tuple[str, str]] = set()
        tags: set[str] = set()
        counts = {"matched": 0, "fp": 0, "fn": 0}

        gt_boxes, pred_boxes = [], []

        for p, o in zip(preds, pred_out, strict=True):
            if p["matched_gt"] is None:
                counts["fp"] += 1
                cells.add((_BACKGROUND, _name(p["class_id"])))
                pred_boxes.append(_box_dict(
                    p["class_id"], class_names, p["box"],
                    confidence=p["confidence"], match=_BACKGROUND,
                    iou_value=o["iou"], outcome=o["outcome"],
                    rival_iou=o.get("rival_iou") if o["outcome"] == "duplicate" else None,
                ))
            else:
                counts["matched"] += 1
                gt_cls = gts[p["matched_gt"]]["class_id"]
                cells.add((_name(gt_cls), _name(p["class_id"])))
                pred_boxes.append(_box_dict(
                    p["class_id"], class_names, p["box"],
                    confidence=p["confidence"], match="matched",
                    counterpart=_name(gt_cls), iou_value=o["iou"],
                    outcome=o["outcome"],
                ))
            if o["outcome"] in ("spurious", "duplicate"):
                tags.add(f"{_name(p['class_id'])}:{o['outcome']}")

        for g, o in zip(gts, gt_out, strict=True):
            tags.add(f"{_name(g['class_id'])}:{o['outcome']}")
            if g["matched_pred"] is None:
                counts["fn"] += 1
                cells.add((_name(g["class_id"]), _BACKGROUND))
                gt_boxes.append(_box_dict(
                    g["class_id"], class_names, g["box"], match=_BACKGROUND,
                    outcome=o["outcome"],
                ))
            else:
                pred_cls = preds[g["matched_pred"]]["class_id"]
                gt_boxes.append(_box_dict(
                    g["class_id"], class_names, g["box"], match="matched",
                    counterpart=_name(pred_cls), outcome=o["outcome"],
                ))

        cell_list = sorted(cells)
        if not any(kept_per_cell.get(c, 0) < max_per_cell for c in cell_list):
            continue
        for c in cell_list:
            kept_per_cell[c] = kept_per_cell.get(c, 0) + 1

        samples.append({
            "path": m["path"],
            "cells": [list(c) for c in cell_list],
            "tags": sorted(tags),
            "counts": counts,
            "gt": gt_boxes,
            "pred": pred_boxes,
        })

    return samples
