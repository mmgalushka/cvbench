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
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from cvbench.core.config import FLAG_KEYS, KEY_FLAGS, key_type, resolve_key
from cvbench.core.errors import ConfigError, CVBenchError
from cvbench.core.exp_store import SWEEP_MANIFEST, _read_entry, sweep_dirs, validate_run_name
from cvbench.tasks import task_for_metric

MANIFEST_VERSION = 1

# Curated base space for `--strategy random` (see sample_grid / cli/sweep.py), keyed by dotted
# config key. Any axis the user passes explicitly overrides the corresponding entry before sampling.
DEFAULT_RANDOM_SPACE: dict[str, list[str]] = {
    # 1e-2 was dropped: combined with sgd:momentum=0.9 and a fully unfrozen backbone
    # (fine_tune_from_layer=-1) it reliably diverges to NaN on a from-scratch detection
    # head within a handful of batches. 1e-3 (the `train` default) is a safer ceiling.
    "training.learning_rate": ["1e-3", "3e-4", "1e-4", "1e-5"],
    "training.optimizer": ["adam", "sgd:momentum=0.9"],
    "model.dropout": ["0.0", "0.2", "0.3", "0.5"],
    "model.fine_tune_from_layer": ["0", "-1"],
    "training.lr_scheduler": ["patience=2,factor=0.5", "patience=5,factor=0.3"],
    "data.batch_size": ["16", "32", "64"],
    "training.loss": ["crossentropy", "focal:gamma=2.0"],
}

# Keys whose value is a spec with comma-separated params (e.g. `sgd:momentum=0.9,weight_decay=1e-4`).
_SPEC_KEYS = ("training.optimizer", "training.loss", "training.lr_scheduler")

# Names older manifests used for a metric, mapped to the name runs record it under.
_METRIC_ALIASES = {"test_accuracy": "accuracy"}


class SweepError(ConfigError):
    """Invalid sweep definition, name, or manifest."""


# ---------------------------------------------------------------------------
# Axis parsing / grid expansion
# ---------------------------------------------------------------------------

def _split_spec_items(key: str, tokens: list[str]) -> list[str]:
    """Re-join comma-split tokens belonging to one optimizer/loss/lr_scheduler spec."""
    items: list[str] = []
    for tok in tokens:
        continuation = False
        if items and "=" in tok and ":" not in tok:
            if key == "training.lr_scheduler":
                # Specs are bare `key=value,...` lists, so a token starts a new item
                # only when its key already appeared in the current item.
                name = tok.split("=", 1)[0].strip()
                seen = {p.split("=", 1)[0].strip() for p in items[-1].split(",")}
                continuation = name not in seen
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

    ``flag`` is a flag alias (`lr`) or any dotted config key; unknown names raise SweepError.
    Plain keys split on every comma. Exceptions:

    * `optimizer` / `loss` / `lr_scheduler`: specs carry comma-separated params
      (`adam,sgd:momentum=0.9,weight_decay=1e-4`). A token containing `=` and no `:`
      is re-joined to the previous item when that item already has params
      (contains `:` or `=`), so the example yields `["adam", "sgd:momentum=0.9,weight_decay=1e-4"]`.
      `lr_scheduler` specs have no `:` (`patience=5,factor=0.5`), so a `key=value` token
      continues the previous item unless its key was already used there
      (`patience=5,factor=0.5,patience=10` -> two items).
    * `class_weight`: commas inside `{...}` JSON dicts are preserved.

    A `:` range such as `0.1:0.5` on numeric keys is not supported and raises SweepError.
    Empty items (`a,,b`, trailing comma) raise SweepError.
    """
    try:
        key = resolve_key(flag)
    except CVBenchError as e:
        raise SweepError(e.message) from e
    shown = f"--{flag.replace('_', '-')}" if flag in FLAG_KEYS else key
    tokens = _split_braced(raw) if key == "training.class_weight" else raw.split(",")
    tokens = [t.strip() for t in tokens]
    if any(not t for t in tokens):
        raise SweepError(f"{shown}: empty value in list {raw!r}")
    if key_type(key) in (int, float) and any(":" in t for t in tokens):
        raise SweepError(
            f"{shown}: ranges are not supported under grid; "
            f"list explicit comma-separated values instead (got {raw!r})"
        )
    return _split_spec_items(key, tokens) if key in _SPEC_KEYS else tokens


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


def sample_grid(axes: dict[str, list[str]], n: int, seed: int) -> list[dict[str, str]]:
    """Sample `n` distinct combinations without replacement from the cartesian product of axes.

    Raises SweepError if `n` is not less than the number of distinct combinations available
    (sampling the whole space is just a grid; use `--strategy grid` for that).
    """
    combos = expand_grid(axes)
    if n < 1:
        raise SweepError("--trials must be at least 1.")
    if n >= len(combos):
        raise SweepError(
            f"--trials {n} must be less than the {len(combos)} distinct combinations available; "
            f"use --strategy grid, or a lower --trials."
        )
    return random.Random(seed).sample(combos, n)


def trial_dir_name(sweep_name: str, index: int) -> str:
    """Directory name of the 1-based trial `index`, e.g. `shapes_lr_003`."""
    return f"{sweep_name}_{index:03d}"


def validate_sweep_name(name: str) -> None:
    """Raise SweepError if name is not a safe directory name (same rules as run names)."""
    try:
        validate_run_name(name)
    except ConfigError as e:
        raise SweepError(str(e)) from e


def default_metric(task: str) -> tuple[str, str]:
    """Default selection (metric, direction) for a task."""
    from cvbench.tasks import get_task

    metric = get_task(task).sweep_metric
    return metric, metric_direction(metric)


def metric_direction(metric: str) -> str:
    """Return "min" or "max" for a known selection metric."""
    from cvbench.tasks import all_metrics

    known = all_metrics()
    try:
        return known[_METRIC_ALIASES.get(metric, metric)].direction
    except KeyError:
        raise SweepError(
            f"Unknown metric '{metric}'. Valid metrics: {', '.join([*known, *_METRIC_ALIASES])}"
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
    trials: list[dict[str, str]] | None = None  # strategy "random": the actual sampled combos
    sample_seed: int | None = None  # strategy "random": seed used to sample `trials`


def write_manifest(sweep_dir: str | Path, manifest: SweepManifest) -> None:
    """Write `sweep.yaml` into sweep_dir."""
    data = asdict(manifest)
    data["axes"] = {k: [str(v) for v in vals] for k, vals in manifest.axes.items()}
    if manifest.trials is not None:
        data["trials"] = [{str(k): str(v) for k, v in t.items()} for t in manifest.trials]
    with open(Path(sweep_dir) / SWEEP_MANIFEST, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _canonical(name: str) -> str:
    """Dotted key of a manifest axis; older manifests stored the `train` flag name (`lr`)."""
    return FLAG_KEYS.get(str(name), str(name))


def display_key(key: str) -> str:
    """Short name of an axis key for tables: its flag alias when it has one (`lr`), else the dotted key."""
    return KEY_FLAGS.get(key, key)


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
        trials_raw = raw.get("trials")
        trials = None
        if trials_raw is not None:
            if not isinstance(trials_raw, list):
                raise TypeError("trials must be a list")
            trials = [{str(k): str(v) for k, v in t.items()} for t in trials_raw]
        sample_seed_raw = raw.get("sample_seed")
        return SweepManifest(
            version=int(version),
            name=str(raw["name"]),
            date=str(raw.get("date", "")),
            data_dir=str(raw.get("data_dir", "")),
            strategy=str(raw.get("strategy", "grid")),
            metric=str(raw["metric"]),
            direction=str(raw["direction"]),
            axes={_canonical(k): [str(v) for v in vals] for k, vals in axes_raw.items()},
            trials=[{_canonical(k): v for k, v in t.items()} for t in trials] if trials is not None else None,
            sample_seed=int(sample_seed_raw) if sample_seed_raw is not None else None,
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
    is_best: bool = False  # best on the selection (val) metric — this is the sweep's pick
    test_value: float | None = None  # from eval_report.json; None until the trial is evaluated
    test_metric: str | None = None
    is_best_test: bool = False  # indicator only; never used to choose the winner


def _test_result(tdir: Path) -> tuple[str | None, float | None]:
    """(metric, value) of the trial's test evaluation, read from eval_report.json."""
    try:
        overall = json.loads((tdir / "eval_report.json").read_text()).get("overall")
        value = overall["value"]
        return overall.get("metric"), float(value)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None, None


def _metric_value(entry: dict[str, Any], metric: str) -> float | None:
    metric = _METRIC_ALIASES.get(metric, metric)
    if metric == entry.get("primary_metric"):
        value = entry.get("primary_value")
    else:
        value = entry.get("metrics", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def selection_metric_of_trial(trial_dir: str | Path) -> str | None:
    """The metric its sweep ranks on, when `trial_dir` is a sweep trial; else None."""
    parent = Path(trial_dir).resolve().parent
    if not (parent / SWEEP_MANIFEST).is_file():
        return None
    try:
        return read_manifest(parent).metric
    except SweepError:
        return None


def best_trial(rows: list[TrialRow]) -> TrialRow | None:
    """Return the row flagged `is_best`, or None."""
    return next((r for r in rows if r.is_best), None)


def summarize(sweep_dir: str | Path) -> tuple[SweepManifest, list[TrialRow]]:
    """Read the manifest and build one row per planned trial, marking the best."""
    sweep = Path(sweep_dir)
    manifest = read_manifest(sweep)
    combos = manifest.trials if manifest.trials is not None else expand_grid(manifest.axes)
    rows: list[TrialRow] = []
    for i, params in enumerate(combos, start=1):
        name = trial_dir_name(manifest.name, i)
        tdir = sweep / name
        if not tdir.is_dir():
            rows.append(TrialRow(i, name, params, "missing", None))
            continue
        entry = _read_entry(tdir)
        if entry is None:
            rows.append(TrialRow(i, name, params, "unreadable", None))
            continue
        test_metric, test_value = _test_result(tdir)
        rows.append(TrialRow(i, name, params, str(entry["status"]), _metric_value(entry, manifest.metric),
                             test_value=test_value, test_metric=test_metric))

    candidates = [r for r in rows if r.status == "done" and r.value is not None]
    if candidates:
        pick = min if manifest.direction == "min" else max
        best = pick(candidates, key=lambda r: r.value)  # type: ignore[arg-type,return-value]
        best.is_best = True
    tested = [r for r in candidates if r.test_value is not None]
    if tested:
        max(tested, key=lambda r: r.test_value).is_best_test = True  # type: ignore[arg-type,return-value]
    return manifest, rows


# ---------------------------------------------------------------------------
# Listing (one summary entry per sweep, for `runs list`)
# ---------------------------------------------------------------------------

def sweep_entry(sweep_dir: str | Path) -> dict[str, Any] | None:
    """Summarize a sweep as one `runs list` entry (same keys as `scan_experiments` plus `is_sweep` and `n_trials`).

    `val_loss` / `val_accuracy` / `primary_metric` / `primary_value` / `epochs_run` come from the best trial (never
    the ranking metric) and are None when no trial has finished. Status is `running` while any trial runs, else `done`.
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
    task = entries[0]["task"] if entries else task_for_metric(manifest.metric)
    return {
        "name": manifest.name or sweep.name,
        "dir": str(sweep),
        "task": task,
        "backbone": best_entry["backbone"] if best_entry else None,
        "epochs": best_entry["epochs"] if best_entry else None,
        "status": "running" if any(e["status"] == "running" for e in entries) else "done",
        "val_loss": best_entry["val_loss"] if best_entry else None,
        "val_accuracy": best_entry["val_accuracy"] if best_entry else None,
        "primary_metric": best_entry["primary_metric"] if best_entry else None,
        "primary_value": best_entry["primary_value"] if best_entry else None,
        "epochs_run": best_entry["epochs_run"] if best_entry else None,
        "date": manifest.date,
        "is_sweep": True,
        "n_trials": len(rows),
    }


def scan_sweeps(parent_dir: str) -> list[dict[str, Any]]:
    """One `sweep_entry` per readable sweep directly under parent_dir."""
    return [e for d in sweep_dirs(parent_dir) if (e := sweep_entry(d)) is not None]
