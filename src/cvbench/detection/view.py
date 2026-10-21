"""Task-specific composition of detection predictions and eval reports (TensorFlow-free).

Builds on ``core/present.py`` (what to show) and ``core/render.py`` (how to print it).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from cvbench.core import console, present
from cvbench.core.confusion import print_confusion_matrix
from cvbench.core.present import fmt_pct
from cvbench.core.render import print_advisories, print_per_class, print_summary
from cvbench.core.task import TaskView

_CLASS_COLUMNS = ["Class", "AP@50", "AP@75", "Precision", "Recall", "F1", "Support"]

# More than this many false positives per image means the detector is firing on
# background, which almost always points at an undertrained model or a conf that is too low.
_FP_PER_IMAGE_ADVISORY = 2.0


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
    print(f" {console.bold(present.SECTIONS['outcomes'])}  {console.dim(note)}")
    console.table(columns, table_rows)
    print()

    misloc = sum(rows[c]["mislocated"] for c in classes)
    missed = sum(rows[c]["missed"] for c in classes)
    nw = max(len(str(counts["fp"])), len(str(counts["fn"])))
    print(console.dim(
        f"   FP {counts['fp']:>{nw}}  =  {misloc} mis-located  +  {sum(dup.values())} duplicate"
        f"  +  {sum(spur.values())} spurious"
    ))
    print(console.dim(
        f"   FN {counts['fn']:>{nw}}  =  {missed} missed  +  {misloc} mis-located"
    ))
    print()

    print(f" {console.bold(present.SECTIONS['extra'])}  {console.dim('(boxes matched to no ground truth)')}")
    dup_line = " · ".join(f"{c} {dup.get(c, 0)}" for c in classes)
    print(f"   duplicate {sum(dup.values()):<5} {dup_line}")
    n_sup, thr = sum(dup_sup.values()), breakdown.get("dup_suppress_iou", 0.5)
    print(console.dim(
        f"             {'':<5} {n_sup} of {sum(dup.values())} removable by NMS at IoU {thr}"
        f" — the rest are separate boxes"
    ))
    spur_line = " · ".join(f"{c} {spur.get(c, 0)}" for c in classes)
    print(f"   spurious  {sum(spur.values()):<5} {spur_line}")
    print()


def _print_single_format(results: list[dict]) -> None:
    for r in results:
        dets = r["detections"]
        print(f" {r['filename']}  {len(dets)} detection{'s' if len(dets) != 1 else ''}")
        for d in dets:
            print(
                f"   {d['class_name']:<20} {d['confidence'] * 100:5.1f}%"
                f"  x={d['x']:.3f} y={d['y']:.3f} w={d['w']:.3f} h={d['h']:.3f}"
            )


def _print_all_formats(formats_run: list[dict]) -> None:
    """One row per image: detection count and class mix per format, flagging
    formats that disagree with the first (reference) one."""
    col_w = 28
    fmt_names = [f["format"] for f in formats_run]
    header = f" {'image':<38}" + "".join(f"  {n:<{col_w}}" for n in fmt_names)
    print(header)
    print(console.dim(" " + "─" * (len(header) - 1)))

    def summary(res: dict) -> tuple[int, list[str]]:
        names = sorted(d["class_name"] for d in res["detections"])
        return len(names), names

    for ref in formats_run[0]["results"]:
        ref_summary = summary(ref)
        row = f" {ref['filename']:<38}"
        for fmt_data in formats_run:
            res = next(
                (r for r in fmt_data["results"] if r["filename"] == ref["filename"]),
                None,
            )
            if res is None:
                cell = "—"
            else:
                count, names = summary(res)
                top = max(res["detections"], key=lambda d: d["confidence"], default=None)
                cell = f"{count} box{'es' if count != 1 else ''}"
                if top:
                    cell += f" (top {top['confidence'] * 100:.1f}%)"
                if fmt_data is not formats_run[0] and (count, names) != ref_summary:
                    cell += "  ⚠️"
            row += f"  {cell:<{col_w}}"
        print(row)


class DetectionView(TaskView):
    def render_predictions(self, formats_run: list[dict], all_formats: bool) -> None:
        if all_formats:
            _print_all_formats(formats_run)
        else:
            _print_single_format(formats_run[0]["results"])

    def eval_summary(self, report: dict, val_accuracy: float | None = None) -> list[tuple[str, str]]:
        det = report["detection"]
        loc = det.get("localization", {})
        sweep = loc.get("recall_sweep", {})
        return [
            ("Split", str(report.get("split") or "test")),
            ("Images evaluated", str(report["n_images"])),
            ("Mean IoU (matched)", fmt_pct(loc.get("mean_iou"), empty="n/a")),
            ("AP@50 / AP@75", f"{fmt_pct(loc.get('ap50'), empty='n/a')} / {fmt_pct(loc.get('ap75'), empty='n/a')}"),
            (
                "Recall @ IoU .5/.75/.9",
                " / ".join(fmt_pct(sweep.get(k), empty="n/a") for k in ("0.5", "0.75", "0.9")),
            ),
            ("mAP@50", fmt_pct(det.get("map50"), empty="n/a")),
        ]

    def per_class_table(self, report: dict) -> dict[str, list]:
        rows = [
            [
                cls,
                fmt_pct(m.get("ap"), empty="n/a"),
                fmt_pct(m.get("ap75"), empty="n/a"),
                fmt_pct(m["precision"]),
                fmt_pct(m["recall"]),
                fmt_pct(m["f1"]),
                str(m["support"]),
            ]
            for cls, m in report["per_class"].items()
        ]
        return {"columns": _CLASS_COLUMNS, "rows": rows}

    def report_advisories(self, report: dict) -> list[str]:
        det = report["detection"]
        n_images = report["n_images"]
        fp = det["counts"]["fp"]
        if n_images and fp / n_images >= _FP_PER_IMAGE_ADVISORY:
            return [
                f"{fp / n_images:.1f} false positives per image at conf ≥ {det['conf_threshold']} — "
                "the detector is likely undertrained; train longer or raise --conf."
            ]
        return []

    def render_report(self, report: dict, val_accuracy: float | None = None, advisories=()) -> None:
        """Print the localization / outcomes / per-class / confusion-matrix body of a detection report."""
        det = report["detection"]

        print_summary(
            self.eval_summary(report),
            heading=(
                f" {console.bold(present.SECTIONS['localization'])}  "
                f"{console.dim('(how well-placed are the boxes)')}"
            ),
        )
        print()

        counts = det["counts"]
        legend = f"conf ≥ {det['conf_threshold']}, IoU ≥ {det['iou_threshold']}"
        breakdown = det.get("class_breakdown")
        if breakdown:
            _print_class_outcomes(breakdown, counts, legend)

        print(
            f" {console.bold(present.SECTIONS['detection_quality'])}  "
            f"{console.dim(f'(AP over all scores; P / R / F1 at {legend})')}"
        )
        print_per_class(self.per_class_table(report))
        print()

        cm = np.array(det["confusion_matrix"]["matrix"])
        cm_classes = det["confusion_matrix"]["classes"]
        print(f" {console.bold(present.SECTIONS['class_confusion'])}  {console.dim('(rows = true, cols = predicted)')}")
        print_confusion_matrix(cm, cm_classes, title="")
        print(console.dim(
            f" TP {counts['tp']} · FP {counts['fp']} · FN {counts['fn']}"
            f"  (class-agnostic, {legend})"
        ))
        print()
        print_advisories(advisories)
