"""``sweep`` — grid search over ``train`` flags.

A sweep is a directory of ordinary experiments plus a ``sweep.yaml`` manifest
(see :mod:`cvbench.core.sweep_store`).
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import click

from cvbench.cli import _help
from cvbench.cli.train import _parse_class_weight, _parse_loss, _parse_lr_scheduler, _parse_optimizer
from cvbench.core import _console
from cvbench.core.exp_store import EXPERIMENTS_DIR, assert_name_available, make_unique_dir
from cvbench.core.sweep_store import (
    SWEEPABLE_FLAGS,
    SweepError,
    SweepManifest,
    TrialRow,
    best_trial,
    default_metric,
    expand_grid,
    metric_direction,
    split_axis_values,
    summarize,
    trial_dir_name,
    validate_sweep_name,
    write_manifest,
)

# NOTE: cvbench.services.training / evaluation (and cvbench.tasks) are imported inside
# the functions that need them — they pull in TensorFlow, and importing them at module
# scope would slow down every `--help` and the `commands` overview.

_INT_FLAGS = ("epochs", "batch_size", "input_size", "fine_tune_from_layer", "seed")
_FLOAT_FLAGS = ("lr", "dropout", "val_split")
_METRICS = ("val_loss", "val_accuracy", "map50")


def _convert(flag: str, raw: str) -> Any:
    """Convert one string value of a sweepable flag to the type `run_training` expects."""
    opt = f"--{flag.replace('_', '-')}"
    try:
        if flag in _INT_FLAGS:
            return int(raw)
        if flag in _FLOAT_FLAGS:
            return float(raw)
    except ValueError:
        kind = "an integer" if flag in _INT_FLAGS else "a number"
        raise click.BadParameter(f"{raw!r} is not {kind}", param_hint=opt) from None
    if flag == "weights":
        if raw not in ("imagenet", "none"):
            raise click.BadParameter(f"{raw!r} is not one of imagenet, none", param_hint=opt)
        return raw
    if flag == "augmentation":
        if not os.path.isfile(raw):
            raise click.BadParameter(f"augmentation file not found: {raw!r}", param_hint=opt)
        return raw
    try:
        if flag == "optimizer":
            return _parse_optimizer(raw)
        if flag == "loss":
            return _parse_loss(raw)
        if flag == "lr_scheduler":
            return _parse_lr_scheduler(raw)
        if flag == "class_weight":
            return _parse_class_weight(raw)
    except click.BadParameter as e:
        e.param_hint = opt
        raise
    except ValueError as e:
        raise click.BadParameter(f"invalid value {raw!r}: {e}", param_hint=opt) from None
    return raw  # backbone


# `run_training` keyword for each sweep flag (only the ones that differ from the flag name).
_KWARG = {"augmentation": "aug_file"}


def _kwargs(values: dict[str, Any]) -> dict[str, Any]:
    return {_KWARG.get(f, f): v for f, v in values.items()}


def print_trial_table(
    rows: list[TrialRow], axes: list[str], metric: str | None, *, show_test: bool = False,
) -> None:
    """Trial table with a ★ on the best trial. With `show_test`, add a test-score column, also starred.

    The val star is the sweep's pick; the test star is only an indicator (a trial
    that hasn't been evaluated yet shows a dash). A legend is printed under the table.
    """
    columns: list[Any] = [("#", "right"), "Trial", *axes]
    if metric is not None:
        if show_test:
            test_metric = next((r.test_metric for r in rows if r.test_metric), None)
            columns += [metric, f"test {test_metric}" if test_metric else "test"]
        else:
            columns += [metric]
        columns += ["Status"]
    vals = [("—" if r.value is None else f"{r.value:.4f}") for r in rows]
    tests = [("—" if r.test_value is None else f"{r.test_value:.4f}") for r in rows]
    val_cells = _console.star_cells(vals, [r.is_best for r in rows])
    test_cells = _console.star_cells(tests, [r.is_best_test for r in rows])
    body = []
    for i, r in enumerate(rows):
        cells: list[Any] = [r.index, r.dir, *(r.params[a] for a in axes)]
        if metric is not None:
            if show_test:
                cells += [val_cells[i], test_cells[i], r.status]
            else:
                cells += [val_cells[i], r.status]
        body.append(cells)
    _console.table(columns, body)
    if metric is not None and any(r.is_best for r in rows):
        print(_console.dim(
            " ★ = best on that score. The sweep's pick is the val ★; the test ★ is only an indicator."
            if show_test else " ★ = the sweep's pick (best on the selection metric)."
        ))


@_help.command(
    short_help="Grid-search training flags and report the best trial.",
    examples=[
        ("sweep data/shapes --lr 1e-3,1e-4",
         "Train once per learning rate and rank the trials by val_loss"),
        ("sweep data/shapes --backbone efficientnet_b0,resnet_18 --lr 1e-3,1e-4 --epochs 5",
         "2 x 2 grid (4 trials); --epochs 5 is a fixed setting"),
        ("sweep data/shapes --lr 1e-3,1e-4 --name shapes_lr --metric val_accuracy",
         "Name the sweep and rank by validation accuracy"),
        ("sweep data/shapes --lr 1e-3,1e-4 --show",
         "Print the trials that would run, then stop"),
    ],
    see_also=[
        ("runs show <trial>", "inspect the best trial"),
        ("evaluate <sweep>", "score every trial on the held-out test split"),
        ("evaluate <trial>", "score just one trial (e.g. the best) on the test split"),
        ("train data/shapes", "run a single configuration"),
    ],
)
@click.argument("data_dir")
@click.option("--name", "name", default=None,
              help="Sweep name = directory under experiments/ (default: sweep_<YYYY_MM_DD>).")
@click.option("--strategy", default="grid", type=click.Choice(["grid", "random"]), show_default=True,
              help="Search strategy. Only grid is implemented; random will follow.")
@click.option("--metric", default=None, type=click.Choice(list(_METRICS)),
              help="Metric used to pick the best trial (default: val_loss; map50 for detection).")
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True),
              help="Load config from an existing experiment as the baseline for every trial.")
@click.option("--show", is_flag=True, help="Print the resolved trials and their count, then exit (no training).")
@click.option("--backbone", default=None, help="Backbone name(s), comma-separated (e.g. efficientnet_b0,resnet_18).")
@click.option("--weights", default=None, help="Weight init(s), comma-separated: imagenet,none.")
@click.option("--epochs", default=None, help="Number of epochs, comma-separated (e.g. 5,10).")
@click.option("--lr", default=None, help="Learning rate(s), comma-separated (e.g. 1e-3,1e-4).")
@click.option("--batch-size", default=None, help="Batch size(s), comma-separated (e.g. 16,32).")
@click.option("--input-size", default=None, help="Input size(s) in pixels, comma-separated (e.g. 224,320).")
@click.option("--dropout", default=None, help="Dropout rate(s), comma-separated (e.g. 0.2,0.5).")
@click.option("--augmentation", default=None,
              help="Augmentation YAML file(s), comma-separated (see 'data aug').")
@click.option("--class-weight", default=None,
              help="Class weighting(s), comma-separated: null,auto,'{\"cat\": 2.0}'.")
@click.option("--loss", default=None,
              help="Loss(es), comma-separated (e.g. crossentropy,focal:gamma=2.0).")
@click.option("--optimizer", default=None,
              help="Optimizer(s), comma-separated (e.g. adam,sgd:momentum=0.9).")
@click.option("--lr-scheduler", default=None,
              help="LR scheduler(s), comma-separated (e.g. patience=3,patience=5,factor=0.3).")
@click.option("--fine-tune-from-layer", default=None,
              help="Fine-tune layer index(es), comma-separated (0=frozen, -1=all).")
@click.option("--val-split", default=None,
              help="Validation split fraction(s), comma-separated (when no val/ directory exists).")
@click.option("--seed", default=None, help="Random seed(s), comma-separated.")
def sweep(data_dir, name, strategy, metric, from_dir, show, **flag_values):
    """Train every combination of the given flag values on DATA_DIR.

    Each sweepable flag takes a comma-separated list: one value fixes the
    setting, two or more make a grid axis. At least one flag needs two or more
    values. Trials run one after another into experiments/<name>/; a failing
    trial is reported and the sweep continues. Ends with a table ranking the trials.

    DATA_DIR can be a full path (data/my_dataset) or a bare dataset name
    resolved under data/.
    """
    from cvbench.core.config import load_config
    from cvbench.core.data_store import resolve_data_dir
    from cvbench.datasets.layout import detect_task_name

    try:
        if strategy != "grid":
            raise click.ClickException(
                f"--strategy {strategy} is not implemented yet; only 'grid' is supported in this release."
            )

        # Parse every list up front so a typo fails before anything is created or trained.
        axes: dict[str, list[str]] = {}
        fixed_raw: dict[str, str] = {}
        for flag in SWEEPABLE_FLAGS:
            raw = flag_values.get(flag)
            if raw is None:
                continue
            values = split_axis_values(flag, raw)
            if len(values) >= 2:
                axes[flag] = values
            else:
                fixed_raw[flag] = values[0]
        if not axes:
            raise click.ClickException(
                "Nothing to sweep: give at least one flag two or more comma-separated values "
                "(e.g. --lr 1e-3,1e-4). For a single configuration use 'train'."
            )
        fixed = {f: _convert(f, v) for f, v in fixed_raw.items()}
        trials = expand_grid(axes)
        trial_values = [{f: _convert(f, v) for f, v in t.items()} for t in trials]

        data_dir = resolve_data_dir(data_dir)
        task = load_config(from_dir).task if from_dir is not None else detect_task_name(data_dir)
        if metric is None:
            metric, direction = default_metric(task)
        else:
            direction = metric_direction(metric)
        if metric == "map50" and task != "detection":
            raise click.ClickException("--metric map50 is only available for detection datasets.")

        if name is None:
            name = make_unique_dir(EXPERIMENTS_DIR, f"sweep_{date.today().strftime('%Y_%m_%d')}").name
        validate_sweep_name(name)
        assert_name_available(name)
        for i in range(1, len(trials) + 1):
            assert_name_available(trial_dir_name(name, i))
    except (SweepError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    planned = [TrialRow(i, trial_dir_name(name, i), t, "planned", None) for i, t in enumerate(trials, start=1)]
    axis_names = list(axes)

    print(_console.rule(79, "white"))
    print(f" {_console.bold(f'CVBench — sweep {name}')}")
    print(_console.rule(79, "white"))
    print(f" {len(trials)} trials ({' x '.join(f'{len(v)} {k}' for k, v in axes.items())}), "
          f"strategy {strategy}, metric {metric} ({direction})")
    if fixed_raw:
        print(_console.dim(" fixed: " + ", ".join(f"{k}={v}" for k, v in fixed_raw.items())))
    if show:
        print_trial_table(planned, axis_names, None)
        print(_console.dim(f" --show: nothing was trained. Total: {len(trials)} trials."))
        return

    sweep_dir = Path(EXPERIMENTS_DIR) / name
    sweep_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(sweep_dir, SweepManifest(
        version=1, name=name, date=date.today().strftime("%Y-%m-%d"), data_dir=str(data_dir),
        strategy=strategy, metric=metric, direction=direction, axes=axes,
    ))

    from cvbench.services.training import run_training  # deferred: pulls in TensorFlow

    for i, (values, plan) in enumerate(zip(trial_values, trials, strict=True), start=1):
        trial_name = trial_dir_name(name, i)
        trial_dir = str(sweep_dir / trial_name)
        print()
        print(f" {_console.bold(f'[{i}/{len(trials)}] {trial_name}')}  "
              + ", ".join(f"{k}={v}" for k, v in plan.items()))
        try:
            run_training(data_dir=data_dir, output_dir=trial_dir, from_dir=from_dir,
                         **_kwargs({**fixed, **values}))
            if metric == "map50":
                _evaluate_on_val(trial_dir)
        except Exception as e:  # noqa: BLE001 - one bad trial must not stop the sweep
            _console.error(f"Trial {trial_name} failed: {type(e).__name__}: {e}")
            continue

    _finish(sweep_dir)


def _finish(sweep_dir: Path) -> None:
    manifest, rows = summarize(sweep_dir)
    print()
    print(_console.rule(79, "white"))
    print(f" {_console.bold(f'CVBench — sweep {manifest.name} results')}  "
          f"{_console.dim(f'{manifest.metric} ({manifest.direction})')}")
    print(_console.rule(79, "white"))
    print_trial_table(rows, list(manifest.axes), manifest.metric)
    best = best_trial(rows)
    n_failed = sum(1 for r in rows if r.status != "done")
    if n_failed:
        _console.warning(f"{n_failed} of {len(rows)} trials did not finish.")
    if best is None:
        _console.warning("No trial produced a value for the selection metric.")
    else:
        print(f" Best: {_console.green(best.dir)} ({manifest.metric} = {best.value:.4f})")
        print(_console.dim(f" Inspect it with: runs show {best.dir}"))
        print(_console.dim(f" Evaluate the best trial on the test split:  evaluate {best.dir}"))
        print(_console.dim(f" Evaluate all trials on the test split:      evaluate {manifest.name}"))


if __name__ == "__main__":
    sweep(prog_name="sweep", args=sys.argv[1:])
