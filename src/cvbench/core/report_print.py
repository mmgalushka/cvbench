"""Terminal rendering of an eval_report.json body, shared across entry points.

Pure NumPy — no TensorFlow/Keras dependency — and task-agnostic: it holds the
building blocks (headline rows, per-class table, advisories). Each task's
``view.py`` composes them into its own report body, so ``evaluate()``'s live
terminal output and the CLI's `runs show` print the exact same numbers from the
exact same eval_report.json.
"""
from __future__ import annotations

from typing import Literal

from cvbench.core import _console


def print_advisories(advisories) -> None:
    for line in advisories:
        _console.warning(line)
    if advisories:
        print()


def print_summary(rows: list[tuple[str, str]], heading: str | None = None) -> None:
    """Split / images lines, then the report's headline scores (indented under HEADING when given)."""
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


def print_per_class(table: dict) -> None:
    columns: list[str | tuple[str, Literal["left", "center", "right"]]] = [
        table["columns"][0],
        *((c, "right") for c in table["columns"][1:]),
    ]
    _console.table(columns, table["rows"])
