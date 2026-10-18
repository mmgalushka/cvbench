"""Terminal rendering of an eval_report.json body, shared across entry points.

Pure NumPy — no TensorFlow/Keras dependency, unlike ``classification/evaluator.py``
and ``detection/evaluator.py`` (which import Keras) or their packages'
``__init__.py`` (which eagerly import TensorFlow via ``data.py``). Importing
either task package to reach its printer would cost ~1.5-2s (see the note atop
``cli/runs.py``), so this module lives outside both, next to ``report.py``
(the shared ``eval_report.json`` envelope) rather than inside a task package.
``classification/evaluator.py`` and ``detection/evaluator.py`` import from
here too, so ``evaluate()``'s live terminal output and the CLI's `runs show`
subcommand print the exact same numbers from the exact same eval_report.json.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from cvbench.core import _console, present
from cvbench.core._confusion import print_confusion_matrix


def print_advisories(advisories) -> None:
    for line in advisories:
        _console.warning(line)
    if advisories:
        print()


def print_classification_body(report: dict, val_accuracy: float | None = None, advisories=()) -> None:
    """Print the accuracy / per-class / confusion-matrix body of a classification report.

    ``val_accuracy`` is the run's validation accuracy, shown beside the test score
    so an overfit gap is visible at a glance; it lives in config.yaml, not the report.
    """
    _print_summary(report, val_accuracy, task="classification")
    print()
    print(f" {_console.bold(present.SECTIONS['per_class'] + ':')}")
    _print_per_class(report, "classification")
    print()
    confusion_matrix = report.get("confusion_matrix")
    if confusion_matrix:
        print_confusion_matrix(np.array(confusion_matrix["matrix"]), confusion_matrix["classes"])
    print_advisories(advisories)


def _print_summary(
    report: dict, val_accuracy: float | None = None, heading: str | None = None, task: str | None = None
) -> None:
    """Split / images lines, then the report's headline scores (indented under HEADING when given)."""
    rows = present.eval_summary(report, val_accuracy, task)
    for label, value in rows[:2]:
        print(_console.dim(f" {label:<22}: {value}"))
    if heading:
        print()
        print(heading)
    for label, value in rows[2:]:
        if heading:
            print(f"   {label:<22} : {value}")
        else:
            print(f" {_console.bold(f'{label:<22}')}: {_console.bold(value)}")


def _print_per_class(report: dict, task: str | None = None) -> None:
    table = present.per_class_table(report, task)
    columns: list[str | tuple[str, Literal["left", "center", "right"]]] = [
        table["columns"][0],
        *((c, "right") for c in table["columns"][1:]),
    ]
    _console.table(columns, table["rows"])


def _print_class_outcomes(breakdown: dict, counts: dict, legend: str) -> None:
    """Per-class outcome table + FP/FN reconciliation + extra-prediction strips.

    Mirrors the WebUI's 'Per-class outcomes' card so the terminal and the
    browser tell the same story.
    """
    classes = breakdown["classes"]
    rows = breakdown["rows"]
    dup, dup_sup = breakdown["duplicate"], breakdown.get("duplicate_suppressible", {})
    spur = breakdown["spurious"]

    def _confused_note(c: str) -> str:
        confused_as = rows[c].get("confused_as") or {}
        return ", ".join(f"→{k} {n}" for k, n in confused_as.items())

    notes = {c: _confused_note(c) for c in classes}
    has_notes = any(notes.values())

    columns: list[str | tuple[str, Literal["left", "center", "right"]]] = [
        "class",
        ("instances", "right"),
        ("matched", "right"),
        ("mis-loc", "right"),
        ("confused", "right"),
        ("missed", "right"),
    ]
    if has_notes:
        columns.append("")

    table_rows = [
        (
            c,
            rows[c]["instances"],
            rows[c]["matched"],
            rows[c]["mislocated"],
            rows[c]["confused"],
            rows[c]["missed"],
            *([notes[c]] if has_notes else []),
        )
        for c in classes
    ]

    note = f"(what happened to every ground-truth box, {legend})"
    print(f" {_console.bold(present.SECTIONS['outcomes'])}  {_console.dim(note)}")
    _console.table(columns, table_rows)
    print()

    misloc = sum(rows[c]["mislocated"] for c in classes)
    missed = sum(rows[c]["missed"] for c in classes)
    nw = max(len(str(counts["fp"])), len(str(counts["fn"])))
    print(_console.dim(
        f"   FP {counts['fp']:>{nw}}  =  {misloc} mis-located  +  {sum(dup.values())} duplicate"
        f"  +  {sum(spur.values())} spurious"
    ))
    print(_console.dim(
        f"   FN {counts['fn']:>{nw}}  =  {missed} missed  +  {misloc} mis-located"
    ))
    print()

    print(f" {_console.bold(present.SECTIONS['extra'])}  {_console.dim('(boxes matched to no ground truth)')}")
    dup_line = " · ".join(f"{c} {dup.get(c, 0)}" for c in classes)
    print(f"   duplicate {sum(dup.values()):<5} {dup_line}")
    n_sup, thr = sum(dup_sup.values()), breakdown.get("dup_suppress_iou", 0.5)
    print(_console.dim(
        f"             {'':<5} {n_sup} of {sum(dup.values())} removable by NMS at IoU {thr}"
        f" — the rest are separate boxes"
    ))
    spur_line = " · ".join(f"{c} {spur.get(c, 0)}" for c in classes)
    print(f"   spurious  {sum(spur.values()):<5} {spur_line}")
    print()


def print_detection_body(report: dict, advisories=()) -> None:
    """Print the localization / outcomes / per-class / confusion-matrix body of a detection report."""
    det = report["detection"]

    _print_summary(
        report,
        heading=(
            f" {_console.bold(present.SECTIONS['localization'])}  "
            f"{_console.dim('(how well-placed are the boxes)')}"
        ),
        task="detection",
    )
    print()

    counts = det["counts"]
    legend = f"conf ≥ {det['conf_threshold']}, IoU ≥ {det['iou_threshold']}"
    breakdown = det.get("class_breakdown")
    if breakdown:
        _print_class_outcomes(breakdown, counts, legend)

    print(
        f" {_console.bold(present.SECTIONS['detection_quality'])}  "
        f"{_console.dim(f'(AP over all scores; P / R / F1 at {legend})')}"
    )
    _print_per_class(report, "detection")
    print()

    cm = np.array(det["confusion_matrix"]["matrix"])
    cm_classes = det["confusion_matrix"]["classes"]
    print(f" {_console.bold(present.SECTIONS['class_confusion'])}  {_console.dim('(rows = true, cols = predicted)')}")
    print_confusion_matrix(cm, cm_classes, title="")
    print(_console.dim(
        f" TP {counts['tp']} · FP {counts['fp']} · FN {counts['fn']}"
        f"  (class-agnostic, {legend})"
    ))
    print()
    print_advisories(advisories)
