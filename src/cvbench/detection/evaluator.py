from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tqdm

from cvbench.core import _console
from cvbench.core.report import report_envelope, write_report
from cvbench.core.report_print import print_detection_body as print_body
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
        for scale_preds, acc in zip(batch_preds, all_preds, strict=True):
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


def _print_report(report: dict, run_dir: str, out_dir: Path) -> None:
    run_name = Path(run_dir).name
    print(_console.rule())
    print(f" {_console.bold('CVBench — evaluate')}  {_console.dim('|')}  {_console.dim('run: ' + run_name)}")
    print(_console.rule())
    print_body(report)
    print(f" {_console.bold('Saved:')}")
    print(f"   {_console.dim(str(out_dir / 'eval_report.json'))}")
    print(_console.rule())
