from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tqdm

from cvbench.core import _fmt
from cvbench.core._confusion import print_confusion_matrix
from cvbench.core.report import report_envelope, write_report
from cvbench.datasets.layout import list_images, read_yolo_boxes, yolo_label_dir
from cvbench.detection.decode import decode_batch
from cvbench.detection.metrics import (
    build_detection_samples,
    compute_detection_metrics,
    match_detections,
)

# Decode every candidate above a tiny epsilon (not the operating conf_threshold)
# so compute_detection_metrics sees the full ranked list it needs for AP.
_AP_DECODE_THRESHOLD = 1e-3


def evaluate(
    model: keras.Model,
    test_ds,
    class_names: list[str],
    run_dir: str,
    test_dir: str,
    ds_root: str,
    anchors: list,
    strides: list[int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
    output_dir: str | None = None,
) -> dict:
    """Run detection evaluation, print report, write eval_report.json.

    Ground truth is read directly from the YOLO label files rather than
    decoded back out of the encoded target tensor, in the same image order
    ``detection/data.py::build_detection_dataset`` uses (sorted
    ``list_images``) — the two must stay in lockstep for predictions and
    ground truth to line up.

    Returns the report dict.
    """
    out_dir = Path(output_dir or run_dir)
    num_classes = len(class_names)

    test_path = Path(test_dir)
    label_dir = yolo_label_dir(test_path, Path(ds_root))
    image_paths = list_images(test_path)

    # model.predict on a multi-output model returns a list of per-scale
    # arrays — accumulate each scale separately across batches.
    all_preds: list[list] = [[] for _ in strides]
    n_batches = test_ds.cardinality().numpy()
    total = int(n_batches) if n_batches > 0 else None
    for images, _targets in tqdm.tqdm(test_ds, total=total, desc=" Evaluating", unit="batch"):
        batch_preds = model.predict(images, verbose=0)
        for scale_preds, acc in zip(batch_preds, all_preds):
            acc.append(scale_preds)
    all_preds_np = [np.concatenate(scale_preds, axis=0) for scale_preds in all_preds]

    n = len(image_paths)
    if len(all_preds_np[0]) != n:
        raise RuntimeError(
            f"Decoded {len(all_preds_np[0])} predictions but found {n} images in {test_dir} — "
            "the eval dataset and the ground-truth file listing have gone out of sync."
        )

    # decode_batch/read_yolo_boxes use the flat {class_id, x, y, w, h} shape
    # the rest of the codebase's box-consuming code expects (matching
    # web/api/datasets.py's _yolo_item / buildBoxLayer() on the WebUI side);
    # metrics.py works with a (class_id, confidence?, (x, y, w, h)) shape
    # internally, so translate at this boundary.
    raw_predictions = decode_batch(
        all_preds_np, num_classes, anchors, strides,
        conf_threshold=_AP_DECODE_THRESHOLD, max_detections=max_detections,
        nms_iou_threshold=iou_threshold,
    )
    predictions = [
        [{"class_id": d["class_id"], "confidence": d["confidence"], "box": (d["x"], d["y"], d["w"], d["h"])}
         for d in dets]
        for dets in raw_predictions
    ]
    ground_truths = [
        [{"class_id": cid, "box": box} for cid, box in read_yolo_boxes(label_dir / f"{p.stem}.txt")]
        for p in image_paths
    ]

    # One greedy class-agnostic matching pass feeds the localization view, the
    # confusion matrix, and the sample gallery. Per-class AP keeps its own
    # ranked same-class match inside compute_detection_metrics — that is the
    # AP definition, not a second interpretation of the data.
    rel_paths = [str(p.relative_to(test_path)) for p in image_paths]
    image_matches = match_detections(
        rel_paths, ground_truths, predictions, class_names,
        iou_floor=iou_threshold, conf_threshold=conf_threshold,
    )

    metrics = compute_detection_metrics(
        ground_truths, predictions, class_names, image_matches,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )
    samples = build_detection_samples(image_matches, class_names)

    report = report_envelope(
        task="detection",
        split="test",
        n_images=n,
        overall_metric="map50",
        overall_value=metrics["map50"],
        overall_label="mAP@50",
        per_class=metrics["per_class"],
        samples=samples,
        detection={
            "map50": metrics["map50"],
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "counts": metrics["counts"],
            "localization": metrics["localization"],
            "class_breakdown": metrics["class_breakdown"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        # Top-level mirror so the generic WebUI confusion-matrix path
        # (report.confusion_matrix) works for detection too.
        confusion_matrix=metrics["confusion_matrix"],
    )

    write_report(report, out_dir)
    _print_report(report, run_dir, out_dir)
    return report


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "n/a"


def _print_class_outcomes(breakdown: dict, counts: dict) -> None:
    """Per-class outcome table + FP/FN reconciliation + extra-prediction strips.

    Mirrors the WebUI's 'Per-class outcomes' card so the terminal and the
    browser tell the same story.
    """
    classes = breakdown["classes"]
    rows = breakdown["rows"]
    dup, dup_sup = breakdown["duplicate"], breakdown.get("duplicate_suppressible", {})
    spur = breakdown["spurious"]

    hdr = ("class", "instances", "matched", "mis-loc", "confused", "missed")
    name_w = max(len(hdr[0]), *(len(c) for c in classes))
    cells = [
        (
            c,
            str(rows[c]["instances"]),
            str(rows[c]["matched"]),
            str(rows[c]["mislocated"]),
            str(rows[c]["confused"]),
            str(rows[c]["missed"]),
        )
        for c in classes
    ]
    widths = [name_w] + [
        max(len(hdr[i]), *(len(row[i]) for row in cells)) for i in range(1, len(hdr))
    ]

    def _fmt_row(row: tuple[str, ...]) -> str:
        return "   " + "  ".join(
            v.ljust(widths[0]) if i == 0 else v.rjust(widths[i])
            for i, v in enumerate(row)
        )

    print(f" {_fmt.bold('Per-class outcomes')}  {_fmt.dim('(what happened to every ground-truth box)')}")
    print(_fmt.dim(_fmt_row(hdr)))
    for c, row in zip(classes, cells):
        line = _fmt_row(row)
        confused_as = rows[c].get("confused_as") or {}
        if confused_as:
            tail = ", ".join(f"→{k} {n}" for k, n in confused_as.items())
            line += _fmt.dim(f"   ({tail})")
        print(line)
    print()

    misloc = sum(rows[c]["mislocated"] for c in classes)
    missed = sum(rows[c]["missed"] for c in classes)
    nw = max(len(str(counts["fp"])), len(str(counts["fn"])))
    print(_fmt.dim(
        f"   FP {counts['fp']:>{nw}}  =  {misloc} mis-located  +  {sum(dup.values())} duplicate"
        f"  +  {sum(spur.values())} spurious"
    ))
    print(_fmt.dim(
        f"   FN {counts['fn']:>{nw}}  =  {missed} missed  +  {misloc} mis-located"
    ))
    print()

    print(f" {_fmt.bold('Extra predictions')}  {_fmt.dim('(boxes matched to no ground truth)')}")
    dup_line = " · ".join(f"{c} {dup.get(c, 0)}" for c in classes)
    print(f"   duplicate {sum(dup.values()):<5} {dup_line}")
    n_sup, thr = sum(dup_sup.values()), breakdown.get("dup_suppress_iou", 0.5)
    print(_fmt.dim(
        f"             {'':<5} {n_sup} of {sum(dup.values())} removable by NMS at IoU {thr}"
        f" — the rest are separate boxes"
    ))
    spur_line = " · ".join(f"{c} {spur.get(c, 0)}" for c in classes)
    print(f"   spurious  {sum(spur.values()):<5} {spur_line}")
    print()


def _print_report(report: dict, run_dir: str, out_dir: Path) -> None:
    run_name = Path(run_dir).name
    n_images = report["n_images"]
    det = report["detection"]
    loc = det.get("localization", {})
    sweep = loc.get("recall_sweep", {})

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — evaluate')}  {_fmt.dim('|')}  {_fmt.dim('run: ' + run_name)}")
    print(_fmt.rule())
    print(_fmt.dim(" Split             : test"))
    print(_fmt.dim(f" Images evaluated  : {n_images}"))
    print()

    print(f" {_fmt.bold('Localization')}  {_fmt.dim('(how well-placed are the boxes)')}")
    print(f"   Mean IoU (matched)     : {_fmt.bold(_pct(loc.get('mean_iou')))}")
    print(f"   AP@50 / AP@75          : {_pct(loc.get('ap50'))} / {_pct(loc.get('ap75'))}")
    print(
        "   Recall @ IoU .5/.75/.9 : "
        f"{_pct(sweep.get('0.5'))} / {_pct(sweep.get('0.75'))} / {_pct(sweep.get('0.9'))}"
    )
    print(_fmt.dim(f"   mAP@50                 : {_pct(det['map50'])}"))
    print()

    counts = det["counts"]
    breakdown = det.get("class_breakdown")
    if breakdown:
        _print_class_outcomes(breakdown, counts)

    print(f" {_fmt.bold('Detection quality')}  {_fmt.dim('(per-class AP / precision / recall)')}")
    per_class = report["per_class"]
    max_cls = max((len(cls) for cls in per_class), default=10)
    for cls, m in per_class.items():
        ap_str = f"{m['ap']:.4f}" if m["ap"] is not None else "n/a"
        p = f"{m['precision']:.4f}"
        r = f"{m['recall']:.4f}"
        f1 = f"{m['f1']:.4f}"
        support = f"({m['support']} instances)"
        print(
            f"   {cls:<{max_cls}}  AP: {_fmt.bold(ap_str)}  P: {p}  R: {r}  F1: {f1}  {_fmt.dim(support)}"
        )
    print()

    cm = np.array(det["confusion_matrix"]["matrix"])
    cm_classes = det["confusion_matrix"]["classes"]
    print(f" {_fmt.bold('Class-confusion matrix')}  {_fmt.dim('(rows = true, cols = predicted)')}")
    print_confusion_matrix(cm, cm_classes, title="")
    print(_fmt.dim(
        f" TP {counts['tp']} · FP {counts['fp']} · FN {counts['fn']}"
        f"  (class-agnostic, conf ≥ {det['conf_threshold']}, IoU ≥ {det['iou_threshold']})"
    ))
    print()

    print(f" {_fmt.bold('Saved:')}")
    print(f"   {_fmt.dim(str(out_dir / 'eval_report.json'))}")
    print(_fmt.rule())
