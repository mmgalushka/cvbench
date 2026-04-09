from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import tqdm


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
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single pass: collect predictions, ground truth, and raw scores
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
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }

    # Confusion matrix
    n_cls = len(class_names)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Samples per confusion matrix cell
    paths = _collect_test_paths(test_dir, class_names)
    samples = _collect_samples(y_true, y_pred, all_preds_np, paths, class_names)

    report = {
        "split": "test",
        "n_images": n,
        "overall_accuracy": round(overall_acc, 4),
        "top3_accuracy": round(top3_acc, 4) if top3_acc is not None else None,
        "per_class": per_class,
        "confusion_matrix": {
            "classes": class_names,
            "matrix": cm.tolist(),
        },
        "samples": samples,
    }

    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report, class_names, run_dir, out_dir, cm)
    return report


def _print_confusion_matrix(cm: np.ndarray, class_names: list[str]):
    """Print a colour-coded confusion matrix to the terminal using ANSI 256 colours.

    Uses the normal layout (horizontal column headers) when the matrix fits the
    terminal width, otherwise falls back to the staircase layout where column
    labels are right-aligned and connected by L-shaped pseudo-graphic lines.
    """
    import shutil

    n = len(class_names)
    label_w = max(len(cls) for cls in class_names)
    max_val = int(cm.max()) if cm.max() > 0 else 1
    term_w = shutil.get_terminal_size((80, 24)).columns

    # ANSI 256-colour blue ramp: white → pure blue (no cyan tint)
    # 231=#ffffff  189=#d7d7ff  147=#afafff  105=#8787ff  63=#5f5fff  21=#0000ff
    _BLUE_RAMP = [231, 189, 147, 105, 63, 21]
    _RESET = "\033[0m"

    def _fmt_cell(val: int, is_diag: bool, cell_w: int) -> str:
        idx = min(int(val / max_val * (len(_BLUE_RAMP) - 1)), len(_BLUE_RAMP) - 1)
        bg = f"\033[48;5;{_BLUE_RAMP[idx]}m"
        fg = "\033[30m" if idx < 3 else "\033[97m"
        bold = "\033[1m" if is_diag else ""
        return f"{bg}{fg}{bold}{val:^{cell_w}}{_RESET}"

    # --- measure whether normal layout fits ---
    num_w = len(str(max_val)) + 2
    normal_cell_w = max(num_w, label_w)
    row_prefix_w = 3 + label_w + 3
    grid_w = n * normal_cell_w + (n - 1)
    normal_fits = (row_prefix_w + grid_w) <= term_w

    print(" Confusion matrix (rows = true, cols = predicted):")

    if normal_fits:
        # ── normal layout ──────────────────────────────────────────────
        pad = " " * (label_w + 3)
        col_header = " ".join(f"{cls:^{normal_cell_w}}" for cls in class_names)
        print(f"   {pad}{col_header}")
        for i, true_cls in enumerate(class_names):
            cells = " ".join(_fmt_cell(int(cm[i, j]), i == j, normal_cell_w) for j in range(n))
            print(f"   {true_cls:<{label_w}} | {cells}")
    else:
        # ── staircase layout ───────────────────────────────────────────
        cell_w = num_w

        col_offsets = [row_prefix_w + j * (cell_w + 1) + cell_w // 2
                       for j in range(n)]

        align_col = col_offsets[-1] + 3

        for i, cls in enumerate(class_names):
            line = " " * col_offsets[0]
            for j in range(i):
                gap = col_offsets[j + 1] - col_offsets[j] - 1
                line += "│" + " " * gap
            n_dashes = align_col - col_offsets[i] - 2
            line += "┌" + "─" * n_dashes + " " + cls
            print(line)

        vert_row = ""
        for j in range(n):
            vert_row = vert_row.ljust(col_offsets[j]) + "│"
        print(vert_row)

        for i, true_cls in enumerate(class_names):
            cells = " ".join(_fmt_cell(int(cm[i, j]), i == j, cell_w) for j in range(n))
            print(f"   {true_cls:<{label_w}} | {cells}")

    print()


def _print_report(report: dict, class_names: list[str], run_dir: str, out_dir: Path,
                  cm: np.ndarray | None = None):
    run_name = Path(run_dir).name
    w = 55
    print("━" * w)
    print(f" CVBench — evaluate  |  run: {run_name}")
    print("━" * w)
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
    if cm is not None:
        _print_confusion_matrix(cm, class_names)
    print(f" Saved:")
    print(f"   {out_dir / 'eval_report.json'}")
    print("━" * w)
