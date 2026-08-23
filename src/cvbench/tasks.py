"""The task registry — the only module that names both task implementations.

Deliberately outside ``cvbench.core`` so core never names a task package.
Import targets are lazy ``(module, attr)`` strings, not eager imports: this
keeps ``import cvbench.tasks`` itself cheap (no TensorFlow), and it means a
config naming an unregistered task fails with a clear error instead of an
import-time crash.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cvbench.core.config import CVBenchConfig
    from cvbench.core.task import Task

# name -> (module to import, attribute holding the Task subclass)
_TASKS: dict[str, tuple[str, str]] = {
    "classification": ("cvbench.classification", "ClassificationTask"),
    "detection": ("cvbench.detection", "DetectionTask"),
}

TASK_NAMES: tuple[str, ...] = tuple(_TASKS)


def get_task(name: str) -> "Task":
    """Instantiate the task registered under NAME.

    Raises ValueError with the list of valid names if NAME isn't registered.
    """
    try:
        module_name, attr = _TASKS[name]
    except KeyError:
        raise ValueError(
            f"Unknown task '{name}'. Valid options: {', '.join(TASK_NAMES)}"
        ) from None
    module = importlib.import_module(module_name)
    task_cls = getattr(module, attr)
    return task_cls()


def resolve_task(cfg: "CVBenchConfig") -> "Task":
    """Instantiate the task named by cfg.task, defaulting to classification."""
    return get_task(getattr(cfg, "task", None) or "classification")
