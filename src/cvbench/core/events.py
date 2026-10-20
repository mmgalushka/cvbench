"""Progress events emitted by training, evaluation and sweeps.

The trainer and the services never print: they hand small frozen event objects
to an optional ``listener`` callable. The CLI turns them into terminal output
(``cvbench.cli._listeners``); the WebUI job runner can queue them for live
progress. A ``None`` listener means "no one is watching".
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """Base class of every progress event."""


Listener = Callable[[Event], None]


def emit(listener: Listener | None, event: Event) -> None:
    """Deliver EVENT to LISTENER, if there is one."""
    if listener is not None:
        listener(event)


# ---- generic ---------------------------------------------------------------

@dataclass(frozen=True)
class Notice(Event):
    """A message for the user; LEVEL is "info", "warning" or "error"."""

    message: str
    level: str = "warning"


@dataclass(frozen=True)
class DeviceDetected(Event):
    """The compute devices found; PURPOSE is a present participle, e.g. "training"."""

    purpose: str
    gpus: tuple[str, ...]


# ---- training --------------------------------------------------------------

@dataclass(frozen=True)
class SeedSet(Event):
    seed: int


@dataclass(frozen=True)
class RunStarted(Event):
    """Training is about to start (before any resumed checkpoint is loaded)."""

    exp_dir: str
    cfg: Any  # CVBenchConfig
    val_images: int | None = None
    resume_checkpoint: str | None = None


@dataclass(frozen=True)
class EpochEnded(Event):
    """One epoch finished; EPOCH is 0-based as in Keras, LOGS holds its metrics."""

    epoch: int
    logs: dict[str, float]


@dataclass(frozen=True)
class InterruptRequested(Event):
    """The user pressed Ctrl-C; training stops after the current batch."""


@dataclass(frozen=True)
class RunFinished(Event):
    """Training ended; STATUS is "done" or "interrupted"."""

    exp_dir: str
    run_name: str
    status: str
    final_metrics: dict[str, float]
    best_events: list[dict] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    interrupt_checkpoint: str | None = None
    data_dir: str = ""


# ---- evaluation ------------------------------------------------------------

@dataclass(frozen=True)
class EvalStarted(Event):
    num_files: int
    num_classes: int


@dataclass(frozen=True)
class EvalProgress(Event):
    """DONE of TOTAL batches predicted (TOTAL is None when unknown); FINISHED on the last event."""

    done: int
    total: int | None
    finished: bool = False


def track_batches(ds: Any, listener: Listener | None) -> Iterator[Any]:
    """Iterate DS, reporting an ``EvalProgress`` before the first batch and after each one."""
    cardinality = int(ds.cardinality().numpy())
    total = cardinality if cardinality > 0 else None
    done = 0
    emit(listener, EvalProgress(done, total))
    for item in ds:
        yield item
        done += 1
        emit(listener, EvalProgress(done, total))
    emit(listener, EvalProgress(done, total, finished=True))


@dataclass(frozen=True)
class SweepEvalTrialStarted(Event):
    """Test-split evaluation of one trial of a sweep begins (INDEX of TOTAL, 1-based)."""

    index: int
    total: int
    trial: str


# ---- sweeps ----------------------------------------------------------------

@dataclass(frozen=True)
class TrialStarted(Event):
    index: int
    total: int
    name: str
    params: dict[str, str]


@dataclass(frozen=True)
class TrialFinished(Event):
    index: int
    name: str


@dataclass(frozen=True)
class TrialFailed(Event):
    """A trial failed in STAGE ("training" or "val evaluation"); the sweep carries on."""

    index: int
    name: str
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class SweepInterrupted(Event):
    """Ctrl-C stopped the sweep during STAGE ("training" or "val evaluation") of trial NAME."""

    name: str
    stage: str


@dataclass(frozen=True)
class SweepFinished(Event):
    sweep_dir: str
