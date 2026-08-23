"""Shared eval_report.json file writing.

Every task's evaluator builds its own report dict; this is the one place that
serializes it. The task-agnostic envelope (``task``, ``overall``, back-compat
mirrors) is layered on top of this in the orchestration step that dispatches
across tasks — see ``services/evaluation.py``.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_report(report: dict, out_dir: str | Path) -> Path:
    """Write REPORT as ``<out_dir>/eval_report.json``, creating out_dir if needed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report_path
