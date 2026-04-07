from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import tqdm


_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _device_status_line() -> str:
    """Return a coloured GPU/CPU availability line for the evaluation header."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        names = ", ".join(g.name for g in gpus)
        return f"{_GREEN}🟢 GPU      : {len(gpus)} device(s) — {names}{_RESET}"
    return f"{_YELLOW}⚠️  GPU      : not available — evaluating on CPU{_RESET}"


def evaluate(
    model: keras.Model,
    test_ds,
    class_names: list[str],
    run_dir: str,
    output_dir: str | None = None,
) -> dict:
    """Run evaluation, print report, write eval_report.json and confusion_matrix.png.

    Returns the report dict.
    """
    out_dir = Path(output_dir or run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single pass: collect predictions, ground truth, and raw scores for top-3
    n_batches = test_ds.cardinality().numpy()
    total = int(n_batches) if n_batches > 0 else None

    y_true, y_pred, all_preds = [], [], []
    for images, labels in tqdm.tqdm(test_ds, total=total, desc=" Evaluating", unit="batch"):
        preds = model.predict(images, verbose=0)
        all_preds.append(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)
    overall_acc = float(np.mean(y_true == y_pred))

    # Top-3 accuracy (if num_classes >= 3) — reuse already-collected scores
    top3_acc = None
    if model.output_shape[-1] >= 3:
        all_preds_np = np.concatenate(all_preds, axis=0)
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
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }

    # Confusion matrix PNG
    _save_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")

    report = {
        "split": "test",
        "n_images": n,
        "overall_accuracy": round(overall_acc, 4),
        "top3_accuracy": round(top3_acc, 4) if top3_acc is not None else None,
        "per_class": per_class,
    }

    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report, class_names, run_dir, out_dir)
    return report


def _save_confusion_matrix(y_true, y_pred, class_names, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im)
    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _print_report(report: dict, class_names: list[str], run_dir: str, out_dir: Path):
    run_name = Path(run_dir).name
    w = 55
    print("━" * w)
    print(f" CVBench — evaluate  |  run: {run_name}")
    print("━" * w)
    print(f" {_device_status_line()}")
    print(f" Split             : test")
    print(f" Images evaluated  : {report['n_images']}")
    print(f" Overall accuracy  : {report['overall_accuracy'] * 100:.1f}%")
    if report["top3_accuracy"] is not None:
        print(f" Top-3 accuracy    : {report['top3_accuracy'] * 100:.1f}%")
    print()
    print(" Per-class breakdown:")
    for cls, m in report["per_class"].items():
        print(f"   {cls:<10} P: {m['precision']:.2f}  R: {m['recall']:.2f}  "
              f"F1: {m['f1']:.2f}  ({m['support']} samples)")
    print()
    print(f" Saved:")
    print(f"   {out_dir / 'eval_report.json'}")
    print(f"   {out_dir / 'confusion_matrix.png'}")
    print("━" * w)
