from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import tqdm

from cvbench.core import _console
from cvbench.core._confusion import print_confusion_matrix
from cvbench.core.report import report_envelope, write_report

_MAX_SAMPLES_PER_CELL = 20


def _collect_test_paths(test_dir: str, class_names: list[str]) -> list[tuple[str, int]]:
    """Return (relative_path, class_idx) pairs in the same order image_dataset_from_directory uses.

    Order matches: sorted class_names, then sorted filenames within each class.
    Paths are relative to test_dir.
    """
    result = []
    test_root = Path(test_dir)
    for idx, cls in enumerate(class_names):
        cls_dir = test_root / cls
        if cls_dir.is_dir():
            for f in sorted(cls_dir.iterdir()):
                if f.is_file():
                    result.append((str(f.relative_to(test_root)), idx))
    return result


def _collect_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    all_preds_np: np.ndarray,
    paths: list[tuple[str, int]],
    class_names: list[str],
) -> list[dict]:
    """Collect up to _MAX_SAMPLES_PER_CELL samples per confusion matrix cell.

    Returns a flat list of dicts with path (relative to test_dir), true_class,
    predicted_class, and confidence (score for the predicted class).
    """
    n_cls = len(class_names)
    cells: dict[tuple[int, int], list[int]] = {
        (t, p): [] for t in range(n_cls) for p in range(n_cls)
    }
    for i in range(len(y_true)):
        cells[(int(y_true[i]), int(y_pred[i]))].append(i)

    samples = []
    for (t, p), indices in cells.items():
        for i in indices[:_MAX_SAMPLES_PER_CELL]:
            path, _ = paths[i]
            samples.append({
                "path": path,
                "true_class": class_names[t],
                "predicted_class": class_names[p],
                "confidence": round(float(all_preds_np[i][p]), 4),
            })
    return samples


def evaluate(
    model: keras.Model,
    test_ds,
    class_names: list[str],
    run_dir: str,
    test_dir: str,
    output_dir: str | None = None,
) -> dict:
    """Run evaluation, print report, write eval_report.json.

    Returns the report dict.
    """
    out_dir = Path(output_dir or run_dir)

    # Single pass: collect predictions, ground truth, and raw scores
    n_batches = test_ds.cardinality().numpy()
    total = int(n_batches) if n_batches > 0 else None

    y_true_list, y_pred_list, all_preds = [], [], []
    for images, labels in tqdm.tqdm(test_ds, total=total, desc=" Evaluating", unit="batch"):
        preds = model.predict(images, verbose=0)
        all_preds.append(preds)
        y_pred_list.extend(np.argmax(preds, axis=1))
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    all_preds_np = np.concatenate(all_preds, axis=0)

    n = len(y_true)
    overall_acc = float(np.mean(y_true == y_pred))

    # Top-3 accuracy (if num_classes >= 3)
    top3_acc = None
    if model.output_shape[-1] >= 3:
        top3 = np.argsort(all_preds_np, axis=1)[:, -3:]
        top3_acc = float(np.mean([y_true[i] in top3[i] for i in range(n)]))

    # Per-class P / R / F1
    per_class = {}
    for idx, cls in enumerate(class_names):
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        support = int(np.sum(y_true == idx))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Confusion matrix
    n_cls = len(class_names)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1

    # Samples per confusion matrix cell
    paths = _collect_test_paths(test_dir, class_names)
    samples = _collect_samples(y_true, y_pred, all_preds_np, paths, class_names)

    rounded_top3 = round(top3_acc, 4) if top3_acc is not None else None
    confusion_matrix = {"classes": class_names, "matrix": cm.tolist()}

    report = report_envelope(
        task="classification",
        split="test",
        n_images=n,
        overall_metric="accuracy",
        overall_value=round(overall_acc, 4),
        overall_label="Overall Accuracy",
        per_class=per_class,
        samples=samples,
        classification={
            "top3_accuracy": rounded_top3,
            "confusion_matrix": confusion_matrix,
        },
        # Legacy top-level mirrors for readers written before the envelope
        # existed (core/runs.py, web/api/runs.py fall back to these).
        overall_accuracy=round(overall_acc, 4),
        top3_accuracy=rounded_top3,
        confusion_matrix=confusion_matrix,
    )

    write_report(report, out_dir)

    _print_report(report, class_names, run_dir, out_dir, cm)
    return report


def _print_report(report: dict, class_names: list[str], run_dir: str, out_dir: Path,
                  cm: np.ndarray | None = None):
    run_name = Path(run_dir).name
    n_images = report["n_images"]
    overall_acc = f"{report['overall_accuracy'] * 100:.1f}%"
    print(_console.rule())
    print(f" {_console.bold('CVBench — evaluate')}  {_console.dim('|')}  {_console.dim('run: ' + run_name)}")
    print(_console.rule())
    print(_console.dim(" Split             : test"))
    print(_console.dim(f" Images evaluated  : {n_images}"))
    print(f" {_console.bold('Overall accuracy')}  : {_console.bold(overall_acc)}")
    if report["top3_accuracy"] is not None:
        top3_acc = f"{report['top3_accuracy'] * 100:.1f}%"
        print(f" {_console.bold('Top-3 accuracy')}    : {_console.bold(top3_acc)}")
    print()
    print(f" {_console.bold('Per-class breakdown:')}")
    _console.table(
        ["class", ("P", "right"), ("R", "right"), ("F1", "right"), ("support", "right")],
        [
            (cls, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", f"{m['support']} samples")
            for cls, m in report["per_class"].items()
        ],
    )
    print()
    if cm is not None:
        print_confusion_matrix(cm, class_names)
    print(f" {_console.bold('Saved:')}")
    print(f"   {_console.dim(str(out_dir / 'eval_report.json'))}")
    print(_console.rule())
