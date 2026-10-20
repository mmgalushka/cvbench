from __future__ import annotations

from pathlib import Path

import keras
import numpy as np

from cvbench.classification.view import render_report as print_body
from cvbench.core import _console, present
from cvbench.core.events import Listener, track_batches
from cvbench.core.report import report_envelope, write_report

_MAX_SAMPLES_PER_CELL = 20
# Top-3 only says something once there are clearly more than three classes to pick from.
_TOP3_MIN_CLASSES = 6


def _collect_test_paths(test_dir: str, class_names: list[str]) -> list[tuple[str, int]]:
    """Return (relative_path, class_idx) pairs in the order the test dataset yields them.

    Shares ``list_labeled_images`` with ``data.build_dataset``, so the two
    orders cannot drift apart. Paths are relative to test_dir.
    """
    from cvbench.classification.data import list_labeled_images

    paths, labels = list_labeled_images(test_dir, class_names)
    test_root = Path(test_dir)
    return [(str(Path(p).relative_to(test_root)), idx) for p, idx in zip(paths, labels, strict=True)]


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
    val_accuracy: float | None = None,
    listener: Listener | None = None,
) -> dict:
    """Run evaluation, print report, write eval_report.json.

    ``val_accuracy`` (the run's validation score) is only printed beside the test score.

    Returns the report dict.
    """
    out_dir = Path(output_dir or run_dir)

    # Single pass: collect predictions, ground truth, and raw scores
    y_true_list, y_pred_list, all_preds = [], [], []
    for images, labels in track_batches(test_ds, listener):
        preds = model.predict(images, verbose=0)
        all_preds.append(preds)
        y_pred_list.extend(np.argmax(preds, axis=1))
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    all_preds_np = np.concatenate(all_preds, axis=0)

    n = len(y_true)
    overall_acc = float(np.mean(y_true == y_pred))

    # Top-3 accuracy (only when there are more than 5 classes)
    top3_acc = None
    if model.output_shape[-1] >= _TOP3_MIN_CLASSES:
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
        # existed (core/exp_store.py, web/api/runs.py fall back to these).
        overall_accuracy=round(overall_acc, 4),
        top3_accuracy=rounded_top3,
        confusion_matrix=confusion_matrix,
    )

    write_report(report, out_dir)

    _print_report(report, run_dir, out_dir, val_accuracy)
    return report


def _print_report(report: dict, run_dir: str, out_dir: Path, val_accuracy: float | None = None) -> None:
    run_name = Path(run_dir).name
    print(_console.rule())
    print(f" {_console.bold('CVBench — evaluate')}  {_console.dim('|')}  {_console.dim('run: ' + run_name)}")
    print(_console.rule())
    print_body(report, val_accuracy, present.run_advisories(run_dir, report))
    print(f" {_console.bold('Saved:')}")
    print(f"   {_console.dim(str(out_dir / 'eval_report.json'))}")
    print(_console.rule())
