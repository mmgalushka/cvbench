"""Hyper-parameter sweeps: a directory of ordinary experiments plus a manifest.

Layout::

    experiments/<sweep_name>/          (no config.yaml)
        sweep.yaml                     manifest: intent only (axes, metric, ...)
        <sweep_name>_001/              trial = ordinary experiment dir
        <sweep_name>_002/ ...

Membership is location: a trial belongs to the sweep because it lives in the sweep
directory. Per-trial params, status and results are never stored in the manifest;
they are read live from each trial's `config.yaml`. A planned-but-absent trial is
reported as `missing` by comparing the grid implied by `axes` with the trial dirs.

This module is TensorFlow-free (see tests/test_import_boundaries.py).
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from cvbench.core.config import CVBenchConfig, load_config
from cvbench.core.exp_store import SWEEP_MANIFEST, _read_entry, sweep_dirs, validate_run_name

MANIFEST_VERSION = 1

# `train` flags that may be swept (click parameter names of cli/train.py).
# Output/--from/--resume are intentionally excluded: they identify a run, not a hyper-parameter.
SWEEPABLE_FLAGS: tuple[str, ...] = (
    "backbone",
    "weights",
    "epochs",
    "lr",
    "batch_size",
    "input_size",
    "dropout",
    "optimizer",
    "loss",
    "lr_scheduler",
    "class_weight",
    "fine_tune_from_layer",
    "val_split",
    "seed",
    "augmentation",
)

# Flags whose value is a spec with comma-separated params (e.g. `sgd:momentum=0.9,weight_decay=1e-4`).
_SPEC_FLAGS = ("optimizer", "loss", "lr_scheduler")
# Numeric flags where `a:b` would be a range (not supported yet).
_RANGE_FLAGS = ("epochs", "lr", "batch_size", "input_size", "dropout", "fine_tune_from_layer", "val_split", "seed")

_METRIC_DIRECTIONS = {
    "val_loss": "min",
    "val_accuracy": "max",
    "test_accuracy": "max",
    "map50": "max",
}


class SweepError(ValueError):
    """Invalid sweep definition, name, or manifest."""


# ---------------------------------------------------------------------------
# Axis parsing / grid expansion
# ---------------------------------------------------------------------------

def _split_spec_items(flag: str, tokens: list[str]) -> list[str]:
    """Re-join comma-split tokens belonging to one optimizer/loss/lr_scheduler spec."""
    items: list[str] = []
    for tok in tokens:
        continuation = False
        if items and "=" in tok and ":" not in tok:
            if flag == "lr_scheduler":
                # Specs are bare `key=value,...` lists, so a token starts a new item
                # only when its key already appeared in the current item.
                key = tok.split("=", 1)[0].strip()
                seen = {p.split("=", 1)[0].strip() for p in items[-1].split(",")}
                continuation = key not in seen
            else:
                continuation = ":" in items[-1] or "=" in items[-1]
        if continuation:
            items[-1] += "," + tok
        else:
            items.append(tok)
    return items


def _split_braced(raw: str) -> list[str]:
    """Split on commas that are outside `{...}` (class_weight JSON dicts)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in raw:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def split_axis_values(flag: str, raw: str) -> list[str]:
    """Split a comma-separated CLI value into stripped, non-empty items.

    Plain flags split on every comma. Exceptions:

    * `optimizer` / `loss` / `lr_scheduler`: specs carry comma-separated params
      (`adam,sgd:momentum=0.9,weight_decay=1e-4`). A token containing `=` and no `:`
      is re-joined to the previous item when that item already has params
      (contains `:` or `=`), so the example yields `["adam", "sgd:momentum=0.9,weight_decay=1e-4"]`.
      `lr_scheduler` specs have no `:` (`patience=5,factor=0.5`), so a `key=value` token
      continues the previous item unless its key was already used there
      (`patience=5,factor=0.5,patience=10` -> two items).
    * `class_weight`: commas inside `{...}` JSON dicts are preserved.

    A `:` range such as `0.1:0.5` on numeric flags is not supported and raises SweepError.
    Empty items (`a,,b`, trailing comma) raise SweepError.
    """
    if flag not in SWEEPABLE_FLAGS:
        raise SweepError(f"Unknown sweep flag '{flag}'. Sweepable flags: {', '.join(SWEEPABLE_FLAGS)}")
    tokens = _split_braced(raw) if flag == "class_weight" else raw.split(",")
    tokens = [t.strip() for t in tokens]
    if any(not t for t in tokens):
        raise SweepError(f"--{flag.replace('_', '-')}: empty value in list {raw!r}")
    if flag in _RANGE_FLAGS and any(":" in t for t in tokens):
        raise SweepError(
            f"--{flag.replace('_', '-')}: ranges are not supported under grid; "
            f"list explicit comma-separated values instead (got {raw!r})"
        )
    return _split_spec_items(flag, tokens) if flag in _SPEC_FLAGS else tokens


def expand_grid(axes: dict[str, list[str]]) -> list[dict[str, str]]:
    """Cartesian product of the axes in deterministic order (first axis slowest)."""
    if not axes:
        raise SweepError("A sweep needs at least one axis (a flag with values to sweep).")
    for flag, values in axes.items():
        if not values:
            raise SweepError(f"Axis '{flag}' has no values.")
        if len(set(values)) != len(values):
            raise SweepError(f"Axis '{flag}' has duplicate values: {values}")
    flags = list(axes)
    return [dict(zip(flags, combo, strict=True)) for combo in itertools.product(*(axes[f] for f in flags))]


def trial_dir_name(sweep_name: str, index: int) -> str:
    """Directory name of the 1-based trial `index`, e.g. `shapes_lr_003`."""
    return f"{sweep_name}_{index:03d}"


def validate_sweep_name(name: str) -> None:
    """Raise SweepError if name is not a safe directory name (same rules as run names)."""
    try:
        validate_run_name(name)
    except ValueError as e:
        raise SweepError(str(e)) from e


def default_metric(task: str) -> tuple[str, str]:
    """Default selection (metric, direction) for a task."""
    if task == "detection":
        return "map50", "max"
    return "val_loss", "min"


def metric_direction(metric: str) -> str:
    """Return "min" or "max" for a known selection metric."""
    try:
        return _METRIC_DIRECTIONS[metric]
    except KeyError:
        raise SweepError(
            f"Unknown metric '{metric}'. Valid metrics: {', '.join(_METRIC_DIRECTIONS)}"
        ) from None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class SweepManifest:
    version: int
    name: str
    date: str
    data_dir: str
    strategy: str
    metric: str
    direction: str
    axes: dict[str, list[str]]


def write_manifest(sweep_dir: str | Path, manifest: SweepManifest) -> None:
    """Write `sweep.yaml` into sweep_dir."""
    data = asdict(manifest)
    data["axes"] = {k: [str(v) for v in vals] for k, vals in manifest.axes.items()}
    with open(Path(sweep_dir) / SWEEP_MANIFEST, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def read_manifest(sweep_dir: str | Path) -> SweepManifest:
    """Read `sweep.yaml` from sweep_dir; raise SweepError if missing, invalid or unsupported."""
    path = Path(sweep_dir) / SWEEP_MANIFEST
    if not path.is_file():
        raise SweepError(f"Not a sweep directory (no {SWEEP_MANIFEST}): {sweep_dir}")
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SweepError(f"Invalid {path}: {e}") from e
    if not isinstance(raw, dict):
        raise SweepError(f"Invalid {path}: expected a mapping.")
    version = raw.get("version")
    if version != MANIFEST_VERSION:
        raise SweepError(f"Unsupported sweep manifest version {version!r} in {path} (expected {MANIFEST_VERSION}).")
    try:
        axes_raw = raw["axes"]
        if not isinstance(axes_raw, dict):
            raise TypeError("axes must be a mapping")
        return SweepManifest(
            version=int(version),
            name=str(raw["name"]),
            date=str(raw.get("date", "")),
            data_dir=str(raw.get("data_dir", "")),
            strategy=str(raw.get("strategy", "grid")),
            metric=str(raw["metric"]),
            direction=str(raw["direction"]),
            axes={str(k): [str(v) for v in vals] for k, vals in axes_raw.items()},
        )
    except (KeyError, TypeError, ValueError) as e:
        raise SweepError(f"Invalid {path}: {e!r}") from e


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@dataclass
class TrialRow:
    index: int
    dir: str  # directory name (not a path)
    params: dict[str, str]  # planned axis values
    status: str  # config.yaml run.status, "missing" or "unreadable"
    value: float | None  # selection-metric value
    is_best: bool = False


def _metric_value(entry: dict[str, Any], metric: str) -> float | None:
    if metric == "map50":
        value = entry.get("test_accuracy") if entry.get("test_metric") == "map50" else None
    else:
        value = entry.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def best_trial(rows: list[TrialRow]) -> TrialRow | None:
    """Return the row flagged `is_best`, or None."""
    return next((r for r in rows if r.is_best), None)


def summarize(sweep_dir: str | Path) -> tuple[SweepManifest, list[TrialRow]]:
    """Read the manifest and build one row per planned trial, marking the best."""
    sweep = Path(sweep_dir)
    manifest = read_manifest(sweep)
    rows: list[TrialRow] = []
    for i, params in enumerate(expand_grid(manifest.axes), start=1):
        name = trial_dir_name(manifest.name, i)
        tdir = sweep / name
        if not tdir.is_dir():
            rows.append(TrialRow(i, name, params, "missing", None))
            continue
        entry = _read_entry(tdir)
        if entry is None:
            rows.append(TrialRow(i, name, params, "unreadable", None))
            continue
        rows.append(TrialRow(i, name, params, str(entry["status"]), _metric_value(entry, manifest.metric)))

    candidates = [r for r in rows if r.status == "done" and r.value is not None]
    if candidates:
        pick = min if manifest.direction == "min" else max
        best = pick(candidates, key=lambda r: r.value)  # type: ignore[arg-type,return-value]
        best.is_best = True
    return manifest, rows


# ---------------------------------------------------------------------------
# Reading swept values back from a trial's config.yaml (for display)
# ---------------------------------------------------------------------------

_FLAG_READERS: dict[str, Callable[[CVBenchConfig], Any]] = {
    "backbone": lambda c: c.model.backbone,
    "weights": lambda c: c.model.weights,
    "epochs": lambda c: c.training.epochs,
    "lr": lambda c: c.training.learning_rate,
    "batch_size": lambda c: c.data.batch_size,
    "input_size": lambda c: c.model.input_size,
    "dropout": lambda c: c.model.dropout,
    "optimizer": lambda c: c.training.optimizer.type,
    "loss": lambda c: c.training.loss.type,
    "lr_scheduler": lambda c: c.training.lr_scheduler.patience,
    "class_weight": lambda c: c.training.class_weight,
    "fine_tune_from_layer": lambda c: c.model.fine_tune_from_layer,
    "val_split": lambda c: c.data.val_split,
    "seed": lambda c: c.training.seed,
}


def trial_config_values(trial_dir: str | Path, flags: list[str]) -> dict[str, Any]:
    """Actual values of the given sweep flags as recorded in a trial's config.yaml.

    Flags without a simple config field (e.g. `augmentation`) are omitted; returns {}
    if the config cannot be read.
    """
    try:
        cfg = load_config(str(trial_dir))
    except Exception:
        return {}
    return {f: _FLAG_READERS[f](cfg) for f in flags if f in _FLAG_READERS}


# ---------------------------------------------------------------------------
# Listing (one summary entry per sweep, for `runs list`)
# ---------------------------------------------------------------------------

def sweep_entry(sweep_dir: str | Path) -> dict[str, Any] | None:
    """Summarize a sweep as one `runs list` entry (same keys as `scan_experiments` plus `is_sweep`).

    `val_loss` / `epochs_run` come from the best trial (never the ranking metric) and are None
    when no trial has finished. Status is `running` while any trial runs, else `done`.
    Returns None if the manifest is missing or unreadable.
    """
    sweep = Path(sweep_dir)
    try:
        manifest, rows = summarize(sweep)
    except SweepError:
        return None
    entries = [e for r in rows if (e := _read_entry(sweep / r.dir)) is not None]
    best = best_trial(rows)
    best_entry = _read_entry(sweep / best.dir) if best else None
    task = entries[0]["task"] if entries else ("detection" if manifest.metric == "map50" else "classification")
    return {
        "name": manifest.name or sweep.name,
        "dir": str(sweep),
        "task": task,
        "status": "running" if any(e["status"] == "running" for e in entries) else "done",
        "val_loss": best_entry["val_loss"] if best_entry else None,
        "epochs_run": best_entry["epochs_run"] if best_entry else None,
        "date": manifest.date,
        "is_sweep": True,
    }


def scan_sweeps(parent_dir: str) -> list[dict[str, Any]]:
    """One `sweep_entry` per readable sweep directly under parent_dir."""
    return [e for d in sweep_dirs(parent_dir) if (e := sweep_entry(d)) is not None]
