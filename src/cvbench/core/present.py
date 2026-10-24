"""Canonical presentation of runs, eval reports and datasets.

The terminal (Rich tables in ``cli/``) and the WebUI (``web/api`` → ``static/js``)
read the same data. Every label, column, unit and advisory they show lives here so
the two cannot drift: one unit convention (fractions in [0, 1] are shown as
percentages with one decimal; losses are plain decimals), one set of column and
section names, and one place that decides which advisories apply.

Layering: this module decides *what* to show, shared by CLI and WebUI (pure Python,
no Rich, no TensorFlow, so both import it without a startup cost); ``core/render.py``
decides *how* to print it in the terminal; each ``<task>/view.py`` composes the two
into that task's own report and prediction output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cvbench.core.exp_store import loss_still_falling
from cvbench.core.task import MetricSpec

EMPTY = "—"

# ---------------------------------------------------------------------------
# Number formatting — the single unit convention
# ---------------------------------------------------------------------------


def fmt_pct(value: float | None, *, empty: str = EMPTY) -> str:
    """A fraction in [0, 1] as a percentage with one decimal: 0.9345 → '93.5%'."""
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else empty


def fmt_loss(value: float | None) -> str:
    """A loss (not a fraction) as a plain 4-decimal number."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else EMPTY


def fmt_metric(metric: str | None, value: float | None) -> str:
    """A selection/score metric by name, formatted the way its task's schema says."""
    spec = _metric_spec(metric)
    is_loss = spec.fmt == "loss" if spec else bool(metric and metric.endswith("loss"))
    return fmt_loss(value) if is_loss else fmt_pct(value)


def fmt_lr(value: float | None) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) else EMPTY


def fmt_epochs(done: int | None, total: int | None = None) -> str:
    """'12 / 20' when the target is known, '12' when it is not, '—' when unknown."""
    if done is None:
        return EMPTY
    return f"{done} / {total}" if total else str(done)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def _metric_spec(metric: str | None) -> MetricSpec | None:
    from cvbench.tasks import all_metrics

    return all_metrics().get(metric or "")


def _default_primary_metric() -> str:
    """The primary metric of the default task — the one shown without a label."""
    from cvbench.tasks import task_or_default

    return task_or_default(None).primary_metric.name


def metric_labels() -> dict[str, str]:
    """``{metric name: display label}`` for every metric of every task."""
    from cvbench.tasks import all_metrics

    return {name: spec.label for name, spec in all_metrics().items()}

# Section titles shared by `runs show` / `evaluate` and the WebUI Evaluation tab.
SECTIONS = {
    "config": "Config",
    "exports": "Exports",
    "evaluation": "Evaluation",
    "per_class": "Per-class metrics",
    "confusion": "Confusion matrix",
    "localization": "Localization",
    "outcomes": "Per-class outcomes",
    "extra": "Extra predictions",
    "detection_quality": "Detection quality",
    "class_confusion": "Class-confusion matrix",
}

EMPTY_RUNS = "No runs found in '{dir}'."
EMPTY_DATASETS = "No datasets found in '{dir}'."

TRAIN_LONGER = "Validation loss was still falling when training stopped — train longer (more --epochs)."

def task_short(task: str | None, *, is_sweep: bool = False) -> str:
    from cvbench.tasks import task_or_default

    short = task_or_default(task).short_name
    return f"{short}·sweep" if is_sweep else short


def metric_label(metric: str | None) -> str:
    metric = metric or _default_primary_metric()
    spec = _metric_spec(metric)
    return spec.label if spec else metric


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

# (key, label, alignment) — the columns of `runs list` and of the WebUI Runs table.
RUN_COLUMNS: list[tuple[str, str, str]] = [
    ("name", "Run", "left"),
    ("task", "Task", "left"),
    ("backbone", "Backbone", "left"),
    ("date", "Date", "left"),
    ("status", "Status", "left"),
    ("val_loss", "Val Loss", "right"),
    ("val_accuracy", "Val Acc", "right"),
    ("test", "Test", "right"),
    ("epochs", "Epochs", "right"),
]


# Below this terminal width the terminal drops these columns (the WebUI always shows them),
# rather than truncating run names.
NARROW_WIDTH = 110
_NARROW_HIDDEN = ("backbone", "date")


def run_columns(width: int | None = None) -> list[tuple[str, str, str]]:
    """RUN_COLUMNS, minus the ones a terminal ``width`` columns wide has no room for."""
    if width is None or width >= NARROW_WIDTH:
        return RUN_COLUMNS
    return [c for c in RUN_COLUMNS if c[0] not in _NARROW_HIDDEN]


def present_run(entry: dict[str, Any]) -> dict[str, str]:
    """Display strings for one run or sweep entry (``scan_experiments`` / ``sweep_entry`` shape).

    ``test`` is the bare primary value; ``test_label`` names its metric ('' for the default
    task's, which needs no name), so the terminal can append it and the WebUI can badge it.
    """
    metric = entry.get("primary_metric")
    label = metric_label(metric) if metric and metric != _default_primary_metric() else ""
    return {
        "name": str(entry.get("name", "")),
        "task": task_short(entry.get("task"), is_sweep=bool(entry.get("is_sweep"))),
        "backbone": str(entry.get("backbone") or EMPTY),
        "date": str(entry.get("date") or EMPTY),
        "status": str(entry.get("status") or "?"),
        "val_loss": fmt_loss(entry.get("val_loss")),
        "val_accuracy": fmt_pct(entry.get("val_accuracy")),
        "test": fmt_metric(metric, entry.get("primary_value")),
        "test_label": label,
        "epochs": fmt_epochs(entry.get("epochs_run"), entry.get("epochs")),
        "lr": fmt_lr(entry.get("lr")),
    }


def run_detail_display(entry: dict[str, Any]) -> dict[str, Any]:
    """``present_run`` plus the ordered headline ``fields`` of the run page."""
    return {**present_run(entry), "fields": [list(p) for p in run_fields(entry)]}


def _test_field_label(metric: str | None) -> str:
    if not metric or metric == _default_primary_metric():
        return f"Test {metric_label(metric).lower()}"
    return f"Test {metric_label(metric)}"


def run_fields(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """Ordered (label, value) facts shown at the top of a single run — `runs show` and the run page."""
    d = present_run(entry)
    return [
        ("Backbone", d["backbone"]),
        ("Learning rate", d["lr"]),
        ("Epochs", d["epochs"]),
        ("Val loss", d["val_loss"]),
        ("Val accuracy", d["val_accuracy"]),
        (_test_field_label(entry.get("primary_metric")), d["test"]),
        ("Status", d["status"]),
        ("Date", d["date"]),
    ]


def run_cell(display: dict[str, str], key: str) -> str:
    """One cell of a run row as plain text (the terminal appends the test metric label)."""
    if key == "test" and display["test_label"] and display["test"] != EMPTY:
        return f"{display['test']} {display['test_label']}"
    return display[key]


# ---------------------------------------------------------------------------
# Eval reports
# ---------------------------------------------------------------------------

def _report_task(report: dict, task: str | None):
    """The task a report belongs to — an unnamed or unregistered one reads as the default task."""
    from cvbench.tasks import task_or_default

    return task_or_default(task or report.get("task"))


def eval_summary(report: dict, val_accuracy: float | None = None, task: str | None = None) -> list[tuple[str, str]]:
    """Ordered (label, value) headline facts of an eval report, for any task."""
    return _report_task(report, task).view().eval_summary(report, val_accuracy)


def per_class_table(report: dict, task: str | None = None) -> dict[str, list]:
    """Columns and formatted rows of the per-class metrics table, for any task."""
    return _report_task(report, task).view().per_class_table(report)


def run_advisories(run_dir: Path | str, report: dict | None) -> list[str]:
    """Every advisory that applies to a run's evaluation — shown by `evaluate`, `runs show` and the WebUI."""
    advisories: list[str] = []
    if report is not None:
        advisories += _report_task(report, None).view().report_advisories(report)
    if loss_still_falling(run_dir):
        advisories.append(TRAIN_LONGER)
    return advisories


def eval_display(report: dict, val_accuracy: float | None = None) -> dict[str, Any] | None:
    """Preformatted eval-report fields the API sends next to the raw report.

    ``None`` for a legacy report missing the fields the formatters need.
    """
    try:
        return {
            "summary": [list(p) for p in eval_summary(report, val_accuracy)],
            "per_class": per_class_table(report),
            "sections": SECTIONS,
        }
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

# (key, label, alignment) — the columns of `data list` and of the WebUI Datasets table.
DATASET_COLUMNS: list[tuple[str, str, str]] = [
    ("name", "Dataset", "left"),
    ("task", "Task", "left"),
    ("train", "Train", "right"),
    ("val", "Val", "right"),
    ("test", "Test", "right"),
    ("classes", "Classes", "right"),
]


def present_dataset(info: dict[str, Any]) -> dict[str, str]:
    """Display strings for one ``datasets.catalog.dataset_info`` entry."""
    splits = info.get("splits", {})
    return {
        "name": str(info["name"]),
        "task": task_short(info.get("task")),
        "train": str(splits.get("train", {}).get("images", 0)),
        "val": str(splits.get("val", {}).get("images", 0)),
        "test": str(splits.get("test", {}).get("images", 0)),
        "classes": str(info.get("num_classes", 0)),
    }


def labels() -> dict[str, Any]:
    """Everything the WebUI needs to label itself, served once at ``/api/present``."""
    from cvbench.tasks import DEFAULT_TASK, task_meta

    return {
        "run_columns": [{"key": k, "label": label, "align": a} for k, label, a in RUN_COLUMNS],
        "dataset_columns": [{"key": k, "label": label, "align": a} for k, label, a in DATASET_COLUMNS],
        "sections": SECTIONS,
        "empty": {"runs": EMPTY_RUNS, "datasets": EMPTY_DATASETS},
        "test_metrics": metric_labels(),
        "tasks": task_meta(),
        "default_task": DEFAULT_TASK,
    }
