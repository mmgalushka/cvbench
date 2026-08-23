from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tqdm

from cvbench.core import _fmt
from cvbench.core.report import report_envelope, write_report
from cvbench.datasets.layout import list_images, read_yolo_boxes, yolo_label_dir
from cvbench.detection.decode import decode_batch
from cvbench.detection.metrics import bucket_samples, compute_detection_metrics

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

    all_preds = []
    n_batches = test_ds.cardinality().numpy()
    total = int(n_batches) if n_batches > 0 else None
    for images, _targets in tqdm.tqdm(test_ds, total=total, desc=" Evaluating", unit="batch"):
        all_preds.append(model.predict(images, verbose=0))
    all_preds_np = np.concatenate(all_preds, axis=0)

    n = len(image_paths)
    if len(all_preds_np) != n:
        raise RuntimeError(
            f"Decoded {len(all_preds_np)} predictions but found {n} images in {test_dir} — "
            "the eval dataset and the ground-truth file listing have gone out of sync."
        )

    # decode_batch/read_yolo_boxes use the flat {class_id, x, y, w, h} shape
    # the rest of the codebase's box-consuming code expects (matching
    # web/api/datasets.py's _yolo_item / buildBoxLayer() on the WebUI side);
    # metrics.py works with a (class_id, confidence?, (x, y, w, h)) shape
    # internally, so translate at this boundary.
    raw_predictions = decode_batch(
        all_preds_np, num_classes, conf_threshold=_AP_DECODE_THRESHOLD, max_detections=max_detections
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

    metrics = compute_detection_metrics(
        ground_truths, predictions, class_names,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )
    samples = bucket_samples(
        [str(p.relative_to(test_path)) for p in image_paths],
        ground_truths, predictions, class_names,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )

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
        },
    )

    write_report(report, out_dir)
    _print_report(report, run_dir, out_dir)
    return report


def _print_report(report: dict, run_dir: str, out_dir: Path) -> None:
    run_name = Path(run_dir).name
    n_images = report["n_images"]
    map50 = report["overall"]["value"]
    map50_str = f"{map50 * 100:.1f}%" if map50 is not None else "n/a"

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — evaluate')}  {_fmt.dim('|')}  {_fmt.dim('run: ' + run_name)}")
    print(_fmt.rule())
    print(_fmt.dim(" Split             : test"))
    print(_fmt.dim(f" Images evaluated  : {n_images}"))
    print(f" {_fmt.bold('mAP@50')}           : {_fmt.bold(map50_str)}")
    det = report["detection"]
    print(_fmt.dim(
        f" conf ≥ {det['conf_threshold']}, IoU ≥ {det['iou_threshold']}"
        f"  →  TP {det['counts']['tp']}  FP {det['counts']['fp']}  FN {det['counts']['fn']}"
    ))
    print()
    print(f" {_fmt.bold('Per-class breakdown:')}")
    per_class = report["per_class"]
    max_cls = max((len(cls) for cls in per_class), default=10)
    for cls, m in per_class.items():
        ap_str = f"{m['ap']:.4f}" if m["ap"] is not None else "n/a"
        p = f"{m['precision']:.4f}"
        r = f"{m['recall']:.4f}"
        support = f"({m['support']} instances)"
        print(
            f"   {cls:<{max_cls}}  AP: {_fmt.bold(ap_str)}  P: {p}  R: {r}  {_fmt.dim(support)}"
        )
    print()
    print(f" {_fmt.bold('Saved:')}")
    print(f"   {_fmt.dim(str(out_dir / 'eval_report.json'))}")
    print(_fmt.rule())
