"""Dataset layout adapters: how a dataset folder is recognised, validated and resolved.

A ``LayoutAdapter`` owns the three layout questions for one on-disk dataset shape:

* ``sniff(root)``: does ROOT look like this layout? (used to pick the task)
* ``errors(root)``: what is wrong with ROOT as this layout? (human-readable)
* ``resolve(cfg, task)``: fill ``cfg.data.*`` and return the ``DatasetSpec``.

A task names its adapter with ``Task.layout``; the order in which adapters are tried when
the task is sniffed is the task's ``layout_priority`` (see ``cvbench.tasks.detect_task_name``).
The registry is TensorFlow-free so the CLI and WebUI can sniff datasets cheaply.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from cvbench.core.errors import ConfigError

if TYPE_CHECKING:
    from cvbench.core.config import CVBenchConfig
    from cvbench.core.task import DatasetSpec, Task


class LayoutAdapter(abc.ABC):
    """One on-disk dataset shape."""

    name: ClassVar[str]

    @abc.abstractmethod
    def sniff(self, root: Path) -> bool:
        """Whether ROOT looks like this layout."""

    @abc.abstractmethod
    def errors(self, root: Path) -> list[str]:
        """Human-readable problems with ROOT as this layout (empty when it is trainable)."""

    @abc.abstractmethod
    def resolve(self, cfg: CVBenchConfig, task: Task) -> DatasetSpec:
        """Set ``cfg.data.{train,val,test}_dir`` (plus classes for labeled layouts) and describe the dataset."""


LAYOUT_ADAPTERS: dict[str, LayoutAdapter] = {}


def register_layout(adapter: LayoutAdapter) -> LayoutAdapter:
    """Register ADAPTER under its ``name``; a second adapter with that name replaces the first."""
    LAYOUT_ADAPTERS[adapter.name] = adapter
    return adapter


def get_layout(name: str) -> LayoutAdapter:
    """The adapter registered under NAME; ConfigError listing the valid names otherwise."""
    try:
        return LAYOUT_ADAPTERS[name]
    except KeyError:
        raise ConfigError(
            f"Unknown dataset layout '{name}'. Valid options: {', '.join(LAYOUT_ADAPTERS) or 'none registered'}"
        ) from None


def layout_names() -> tuple[str, ...]:
    return tuple(LAYOUT_ADAPTERS)
