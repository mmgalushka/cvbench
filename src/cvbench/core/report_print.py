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

from cvbench.core import _console
from cvbench.core._confusion import print_confusion_matrix


def print_classification_body(report: dict) -> None:
    """Print the accuracy / per-class / confusion-matrix body of a classification report."""
    n_images = report["n_images"]
    overall_acc = f"{report['overall_accuracy'] * 100:.1f}%"
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
    confusion_matrix = report.get("confusion_matrix")
    if confusion_matrix:
        print_confusion_matrix(np.array(confusion_matrix["matrix"]), confusion_matrix["classes"])


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

    print(f" {_console.bold('Per-class outcomes')}  {_console.dim('(what happened to every ground-truth box)')}")
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

    print(f" {_console.bold('Extra predictions')}  {_console.dim('(boxes matched to no ground truth)')}")
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


def print_detection_body(report: dict) -> None:
    """Print the localization / outcomes / per-class / confusion-matrix body of a detection report."""
    n_images = report["n_images"]
    det = report["detection"]
    loc = det.get("localization", {})
    sweep = loc.get("recall_sweep", {})

    print(_console.dim(" Split             : test"))
    print(_console.dim(f" Images evaluated  : {n_images}"))
    print()

    print(f" {_console.bold('Localization')}  {_console.dim('(how well-placed are the boxes)')}")
    print(f"   Mean IoU (matched)     : {_console.bold(_pct(loc.get('mean_iou')))}")
    print(f"   AP@50 / AP@75          : {_pct(loc.get('ap50'))} / {_pct(loc.get('ap75'))}")
    print(
        "   Recall @ IoU .5/.75/.9 : "
        f"{_pct(sweep.get('0.5'))} / {_pct(sweep.get('0.75'))} / {_pct(sweep.get('0.9'))}"
    )
    print(_console.dim(f"   mAP@50                 : {_pct(det['map50'])}"))
    print()

    counts = det["counts"]
    breakdown = det.get("class_breakdown")
    if breakdown:
        _print_class_outcomes(breakdown, counts)

    print(f" {_console.bold('Detection quality')}  {_console.dim('(per-class AP / precision / recall)')}")
    per_class = report["per_class"]
    _console.table(
        [
            "class",
            ("AP@50", "right"),
            ("AP@75", "right"),
            ("P", "right"),
            ("R", "right"),
            ("F1", "right"),
            ("support", "right"),
        ],
        [
            (
                cls,
                f"{m['ap']:.4f}" if m["ap"] is not None else "n/a",
                f"{m['ap75']:.4f}" if m.get("ap75") is not None else "n/a",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['support']} instances",
            )
            for cls, m in per_class.items()
        ],
    )
    print()

    cm = np.array(det["confusion_matrix"]["matrix"])
    cm_classes = det["confusion_matrix"]["classes"]
    print(f" {_console.bold('Class-confusion matrix')}  {_console.dim('(rows = true, cols = predicted)')}")
    print_confusion_matrix(cm, cm_classes, title="")
    print(_console.dim(
        f" TP {counts['tp']} · FP {counts['fp']} · FN {counts['fn']}"
        f"  (class-agnostic, conf ≥ {det['conf_threshold']}, IoU ≥ {det['iou_threshold']})"
    ))
    print()
