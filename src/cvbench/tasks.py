"""The task registry — the only module that names both task implementations.

Deliberately outside ``cvbench.core`` so core never names a task package.
Import targets are lazy ``(module, attr)`` strings, not eager imports: this
keeps ``import cvbench.tasks`` itself cheap (no TensorFlow), and it means a
config naming an unregistered task fails with a clear error instead of an
import-time crash.
"""
from __future__ import annotations

import importlib
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING

from cvbench.core.errors import ConfigError

if TYPE_CHECKING:
    from cvbench.core.config import CVBenchConfig
    from cvbench.core.task import MetricSpec, Task

# name -> (module to import, attribute holding the Task subclass)
_TASKS: dict[str, tuple[str, str]] = {
    "classification": ("cvbench.classification", "ClassificationTask"),
    "detection": ("cvbench.detection", "DetectionTask"),
}

# Entry-point group an external package registers its Task subclasses under:
#   [project.entry-points."cvbench.tasks"]
#   segmentation = "my_pkg.segmentation:SegmentationTask"
TASK_ENTRY_POINT_GROUP = "cvbench.tasks"


def _registry() -> dict[str, tuple[str, str]]:
    """The built-in tasks plus those installed packages register as entry points.

    Nothing is imported here (an entry point only names ``(module, attr)``); a built-in
    task always wins over a plugin of the same name.
    """
    registry = dict(_TASKS)
    for ep in entry_points(group=TASK_ENTRY_POINT_GROUP):
        if ep.name in _TASKS:
            warnings.warn(
                f"Ignoring plugin task '{ep.name}' ({ep.value}): a built-in task has that name.",
                stacklevel=2,
            )
            continue
        registry.setdefault(ep.name, (ep.module, ep.attr))
    return registry


def task_names() -> tuple[str, ...]:
    """Names of every registered task, built-in ones first."""
    return tuple(_registry())


# Snapshot taken at import; use task_names() to see plugins registered later.
TASK_NAMES: tuple[str, ...] = task_names()

# The task assumed for a config or report that doesn't name one (pre-detection runs).
DEFAULT_TASK = "classification"

# Flag text and verb used when an option is passed to a run whose task ignores it.
_OPTION_FLAGS = {"conf": ("--conf", "applies"), "top_k": ("--top-k and --probs", "apply")}


def get_task(name: str) -> Task:
    """Instantiate the task registered under NAME.

    Raise ConfigError with the list of valid names if NAME isn't registered.
    """
    try:
        module_name, attr = _registry()[name]
    except KeyError:
        raise ConfigError(
            f"Unknown task '{name}'. Valid options: {', '.join(task_names())}"
        ) from None
    module = importlib.import_module(module_name)
    task_cls = getattr(module, attr)
    return task_cls()


def resolve_task(cfg: CVBenchConfig) -> Task:
    """Instantiate the task named by cfg.task, defaulting to classification."""
    return get_task(getattr(cfg, "task", None) or DEFAULT_TASK)


def task_or_default(name: str | None) -> Task:
    """The task named NAME, or the default task for a missing or unregistered name."""
    return get_task(name if name in task_names() else DEFAULT_TASK)


def detect_task_name(data_dir: str | Path) -> str:
    """Name of the task whose dataset layout DATA_DIR matches (highest ``layout_priority`` first)."""
    root = Path(data_dir)
    tasks = sorted((get_task(n) for n in task_names()), key=lambda t: -t.layout_priority)
    for task in tasks:  # stable: ties keep registration order
        if task.matches_layout(root):
            return task.name
    return DEFAULT_TASK


def layout_errors(data_dir: str | Path, task_name: str | None = None) -> list[str]:
    """Layout problems for DATA_DIR under TASK_NAME (sniffed from the layout when omitted)."""
    root = Path(data_dir)
    return get_task(task_name or detect_task_name(root)).layout_errors(root)


def task_meta() -> dict[str, dict]:
    """``{task name: Task.meta()}`` for every registered task — served to the WebUI."""
    return {name: get_task(name).meta() for name in task_names()}


def all_metrics() -> dict[str, MetricSpec]:
    """Every metric any task records, by name."""
    out: dict[str, MetricSpec] = {}
    for name in task_names():
        for m in get_task(name).metrics_schema():
            out.setdefault(m.name, m)
    return out


def selection_metrics() -> dict[str, MetricSpec]:
    """Every metric a sweep can rank on, by name, across all tasks."""
    out: dict[str, MetricSpec] = {}
    for name in task_names():
        task = get_task(name)
        for m in task.metrics_schema():
            if m.name in task.selection_metric_names:
                out.setdefault(m.name, m)
    return out


def check_selection_metric(metric: str, task_name: str) -> None:
    """Raise ConfigError when TASK_NAME cannot rank a sweep on METRIC."""
    task = get_task(task_name)
    if metric not in task.selection_metric_names:
        owners = [n for n in task_names() if metric in get_task(n).selection_metric_names]
        raise ConfigError(
            f"--metric {metric} is only available for {' / '.join(owners)} datasets."
            if owners
            else f"Unknown metric '{metric}'."
        )


def task_for_metric(metric: str) -> str:
    """The task a selection METRIC belongs to: the only one recording it, else the default."""
    owners = [n for n in task_names() if get_task(n).metric(metric) is not None]
    return owners[0] if len(owners) == 1 else DEFAULT_TASK


def tasks_supporting(option: str) -> list[str]:
    """Names of the tasks whose predictions honour OPTION (``conf`` or ``top_k``)."""
    return [n for n in task_names() if option in get_task(n).predict_options()]


def option_scope_message(option: str) -> str:
    """e.g. ``--conf only applies to detection runs``."""
    flag, verb = _OPTION_FLAGS[option]
    return f"{flag} only {verb} to {' / '.join(tasks_supporting(option))} runs"


def check_predict_options(task: Task, conf: float | None, top_k: int | None = 1) -> None:
    """Raise ConfigError when CONF or TOP_K is set for a task that doesn't honour it.

    ``top_k=1`` is the CLI default, so only another value counts as "set".
    """
    supported = task.predict_options()
    if conf is not None and "conf" not in supported:
        raise ConfigError(option_scope_message("conf"))
    if top_k != 1 and "top_k" not in supported:
        raise ConfigError(option_scope_message("top_k"))
