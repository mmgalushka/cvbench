"""Terminal and table rendering for classification predictions and eval reports (TensorFlow-free)."""
from __future__ import annotations

import numpy as np

from cvbench.core import _console, present
from cvbench.core._confusion import print_confusion_matrix
from cvbench.core.present import fmt_pct
from cvbench.core.report_print import print_advisories, print_per_class, print_summary

_CLASS_COLUMNS = ["Class", "Precision", "Recall", "F1", "Support"]


def eval_summary(report: dict, val_accuracy: float | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Split", str(report.get("split") or "test")),
        ("Images evaluated", str(report["n_images"])),
    ]
    acc = report.get("overall_accuracy")
    if acc is None:
        acc = (report.get("overall") or {}).get("value")
    overall = fmt_pct(acc)
    if val_accuracy is not None:
        overall += f"  (val {fmt_pct(val_accuracy)})"
    rows.append(("Overall accuracy", overall))
    if report.get("top3_accuracy") is not None:
        rows.append(("Top-3 accuracy", fmt_pct(report["top3_accuracy"])))
    return rows


def per_class_table(report: dict) -> dict[str, list]:
    rows = [
        [cls, fmt_pct(m["precision"]), fmt_pct(m["recall"]), fmt_pct(m["f1"]), str(m["support"])]
        for cls, m in report["per_class"].items()
    ]
    return {"columns": _CLASS_COLUMNS, "rows": rows}


def render_report(report: dict, val_accuracy: float | None = None, advisories=()) -> None:
    """Print the accuracy / per-class / confusion-matrix body of a classification report.

    ``val_accuracy`` is the run's validation accuracy, shown beside the test score
    so an overfit gap is visible at a glance; it lives in config.yaml, not the report.
    """
    print_summary(eval_summary(report, val_accuracy))
    print()
    print(f" {_console.bold(present.SECTIONS['per_class'] + ':')}")
    print_per_class(per_class_table(report))
    print()
    confusion_matrix = report.get("confusion_matrix")
    if confusion_matrix:
        print_confusion_matrix(np.array(confusion_matrix["matrix"]), confusion_matrix["classes"])
    print_advisories(advisories)


def render_predictions(formats_run: list[dict], all_formats: bool) -> None:
    if all_formats:
        _print_all_formats(formats_run)
    else:
        _print_single_format(formats_run[0]["results"])


def _print_single_format(results: list[dict]) -> None:
    for r in results:
        if len(r["top_k"]) <= 1:
            print(
                f" {r['filename']:<40} {r['class_name']:<20} "
                f"{r['confidence'] * 100:5.1f}%"
            )
            continue
        print(f" {r['filename']}")
        for e in r["top_k"]:
            print(f"   {e['class_name']:<20} {e['confidence'] * 100:5.1f}%")


def _print_all_formats(formats_run: list[dict]) -> None:
    fmt_names = [f["format"] for f in formats_run]
    col_w = 20

    header = (
        f" {'image':<38}"
        + "".join(f"  {n:<{col_w}}" for n in fmt_names)
        + "  max|diff|"
    )
    print(header)
    print(_console.dim(" " + "─" * (len(header) - 1)))

    ref_results = {
        r["filename"]: r["class_name"] for r in formats_run[0]["results"]
    }
    filenames = [r["filename"] for r in formats_run[0]["results"]]

    for filename in filenames:
        ref_class = ref_results[filename]
        row = f" {filename:<38}"
        for fmt_data in formats_run:
            res = next(
                (r for r in fmt_data["results"] if r["filename"] == filename),
                None,
            )
            if res is None:
                cell = "—"
            else:
                label = f"{res['class_name']} ({res['confidence'] * 100:.1f}%)"
                is_ref = fmt_data is formats_run[0]
                differs = not is_ref and res["class_name"] != ref_class
                if differs:
                    label = f"{label}  ⚠️"
                cell = label
            row += f"  {cell:<{col_w}}"
        ref = next(r for r in formats_run[0]["results"] if r["filename"] == filename)
        diffs = [
            float(np.max(np.abs(np.array(r["probs"]) - np.array(ref["probs"]))))
            for fd in formats_run[1:]
            for r in fd["results"]
            if r["filename"] == filename
        ]
        row += f"  {max(diffs):.4f}" if diffs else "  —"
        print(row)

    ref_rows = formats_run[0]["results"]
    n_classes = len(ref_rows[0]["probs"])
    mean_top = sum(r["confidence"] for r in ref_rows) / len(ref_rows)
    if mean_top <= 1.5 / n_classes:
        print()
        print(
            _console.yellow(
                f" ⚠ near-chance model: mean top-1 confidence {mean_top * 100:.1f}% "
                f"(chance is {100 / n_classes:.1f}%). Matching class names are weak "
                "evidence — compare max|diff| instead."
            )
        )
