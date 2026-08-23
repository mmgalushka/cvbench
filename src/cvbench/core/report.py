"""The shared eval_report.json envelope and file writing.

Every task's evaluator builds its own report dict by calling
``report_envelope()``, then ``write_report()`` to serialize it. Keeping both
here — rather than duplicated per task — is what lets a generic reader
(``core/runs.py``, ``web/api/runs.py``) pull a run's primary score without
knowing which task produced it: ``overall``, ``per_class`` and ``samples``
are the only keys a generic consumer may read. Everything task-specific goes
in a block named after the task (``report["classification"]``,
``report["detection"]``, ...).
"""
from __future__ import annotations

import json
from pathlib import Path


def report_envelope(
    *,
    task: str,
    split: str,
    n_images: int,
    overall_metric: str,
    overall_value: float | None,
    overall_label: str,
    per_class: dict,
    samples: list,
    **task_block_and_legacy,
) -> dict:
    """Assemble the shared eval_report.json shape.

    ``task_block_and_legacy`` carries the task-specific block (e.g.
    ``classification={...}``) plus any flat legacy-mirror keys a task wants
    to keep for older readers (e.g. ``overall_accuracy=...``).
    """
    report = {
        "task": task,
        "split": split,
        "n_images": n_images,
        "overall": {
            "metric": overall_metric,
            "value": overall_value,
            "label": overall_label,
        },
        "per_class": per_class,
        "samples": samples,
    }
    report.update(task_block_and_legacy)
    return report


def write_report(report: dict, out_dir: str | Path) -> Path:
    """Write REPORT as ``<out_dir>/eval_report.json``, creating out_dir if needed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report_path
