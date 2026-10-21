"""``sweep`` — grid search over ``train`` flags.

A sweep is a directory of ordinary experiments plus a ``sweep.yaml`` manifest
(see :mod:`cvbench.core.sweep_store`).
"""
from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

import click

from cvbench.cli import _help
from cvbench.cli._listeners import ConsoleListener
from cvbench.cli.train import parse_override, validate_dataset
from cvbench.core import _console, present
from cvbench.core.config import FLAG_KEYS, build_config, resolve_key
from cvbench.core.events import Event, SweepFinished, SweepInterrupted, TrialFailed, TrialStarted
from cvbench.core.exp_store import EXPERIMENTS_DIR, assert_name_available, make_unique_dir
from cvbench.core.sweep_store import (
    DEFAULT_RANDOM_SPACE,
    SweepManifest,
    TrialRow,
    best_trial,
    default_metric,
    display_key,
    expand_grid,
    metric_direction,
    sample_grid,
    split_axis_values,
    summarize,
    trial_dir_name,
    validate_sweep_name,
)

# NOTE: cvbench.services.training / evaluation (and cvbench.tasks) are imported inside
# the functions that need them — they pull in TensorFlow, and importing them at module
# scope would slow down every `--help` and the `commands` overview.



def _selection_metric_names() -> list[str]:
    from cvbench.tasks import selection_metrics

    return list(selection_metrics())


def _overrides(params: dict[str, str]) -> dict[str, Any]:
    """Text values of axis keys -> the dotted overrides `run_training` applies (spec keys expand to several)."""
    out: dict[str, Any] = {}
    for key, raw in params.items():
        out.update(parse_override(key, raw))
    return out


def print_trial_table(
    rows: list[TrialRow], axes: list[str], metric: str | None, *, show_test: bool = False,
) -> None:
    """Trial table with a ★ on the best trial. With `show_test`, add a test-score column, also starred.

    The val star is the sweep's pick; the test star is only an indicator (a trial
    that hasn't been evaluated yet shows a dash). A legend is printed under the table.
    """
    columns: list[Any] = [("#", "right"), "Trial", *map(display_key, axes)]
    if metric is not None:
        if show_test:
            test_metric = next((r.test_metric for r in rows if r.test_metric), None)
            columns += [metric, f"test {test_metric}" if test_metric else "test"]
        else:
            columns += [metric]
        columns += ["Status"]
    vals = [present.fmt_metric(metric, r.value) for r in rows]
    tests = [present.fmt_pct(r.test_value) for r in rows]
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
        ("sweep data/shapes --strategy random --trials 10",
         "Sample 10 trials from the built-in default search space"),
        ("sweep data/shapes --strategy random --trials 10 --lr 1e-3,1e-4,1e-5",
         "Random search, overriding the default --lr candidates"),
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
              help="Search strategy. 'grid' trains every combination; 'random' samples --trials of them "
                   "from a curated default space (overridable per flag).")
@click.option("--trials", "n_trials", default=None, type=int,
              help="Number of trials to sample. Required for --strategy random; rejected for grid.")
@click.option("--sample-seed", "sample_seed", default=None, type=int,
              help="Seed for --strategy random sampling (auto-picked and recorded in sweep.yaml if omitted).")
@click.option("--metric", default=None, type=click.Choice(_selection_metric_names()),
              help="Metric used to pick the best trial (default: val_loss; map50 for detection).")
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True),
              help="Load config from an existing run as the baseline for every trial.")
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
@click.option("--set", "set_options", multiple=True, metavar="KEY=V1,V2",
              help="Sweep any config key by its dotted name, e.g. --set training.optimizer.weight_decay=0,1e-4 "
                   "(repeatable; the flags above are shortcuts for common keys).")
def sweep(data_dir, name, strategy, n_trials, sample_seed, metric, from_dir, show, set_options, **flag_values):
    """Train every combination (grid) or a sample of combinations (random) on DATA_DIR.

    Each sweep flag, and each --set KEY=V1,V2 for any other config key, takes a
    comma-separated list: one value fixes the setting, two or more make a grid axis. Under --strategy grid, at least one
    flag needs two or more values. Under --strategy random, a curated default
    search space is used for any flag not explicitly passed; --trials picks how
    many combinations to sample from it. Trials run one after another into
    experiments/<name>/; a failing trial is reported and the sweep continues.
    Ends with a table ranking the trials.

    DATA_DIR can be a full path (data/my_dataset) or a bare dataset name
    resolved under data/.
    """
    from cvbench.core.config import load_config
    from cvbench.core.data_store import resolve_data_dir
    from cvbench.tasks import check_selection_metric, detect_task_name

    if strategy == "random":
        if n_trials is None:
            raise click.ClickException("--trials is required for --strategy random.")
    elif n_trials is not None:
        raise click.ClickException(
            "--trials is only valid with --strategy random (grid trains every combination)."
        )

    # Parse every list up front so a typo fails before anything is created or trained.
    # Random search starts from the curated default space; any flag the user passes
    # explicitly overrides the corresponding entry (grid starts from nothing).
    space: dict[str, list[str]] = dict(DEFAULT_RANDOM_SPACE) if strategy == "random" else {}
    given: list[tuple[str, str]] = [(f, flag_values[f]) for f in FLAG_KEYS if flag_values.get(f) is not None]
    for item in set_options:
        key, sep, raw = item.partition("=")
        if not sep:
            raise click.BadParameter(
                f"expected key=v1,v2, got {item!r} (e.g. training.epochs=5,10)", param_hint="--set"
            )
        given.append((key.strip(), raw))
    explicit: set[str] = set()
    for flag, raw in given:
        key = resolve_key(flag)
        if key in explicit:
            raise click.ClickException(f"{key} was given more than once.")
        explicit.add(key)
        space[key] = split_axis_values(flag, raw)
    axes: dict[str, list[str]] = {}
    fixed_raw: dict[str, str] = {}
    for flag, values in space.items():
        if len(values) >= 2:
            axes[flag] = values
        else:
            fixed_raw[flag] = values[0]
    if strategy == "grid" and not axes:
        raise click.ClickException(
            "Nothing to sweep: give at least one flag two or more comma-separated values "
            "(e.g. --lr 1e-3,1e-4). For a single configuration use 'train'."
        )
    fixed = _overrides(fixed_raw)
    if strategy == "random":
        if sample_seed is None:
            sample_seed = random.randrange(2**31)
        trials = sample_grid(axes, n_trials, sample_seed)
    else:
        trials = expand_grid(axes)
    trial_values = [_overrides(t) for t in trials]

    data_dir = resolve_data_dir(data_dir)
    validate_dataset(data_dir, from_dir)
    for values in trial_values:  # type-check every override now, before anything is created or trained
        build_config(data_dir, from_dir, {**fixed, **values})
    task = load_config(from_dir).task if from_dir is not None else detect_task_name(data_dir)
    if metric is None:
        metric, direction = default_metric(task)
    else:
        direction = metric_direction(metric)
    check_selection_metric(metric, task)
    if name is None:
        name = make_unique_dir(EXPERIMENTS_DIR, f"sweep_{date.today().strftime('%Y_%m_%d')}").name
    validate_sweep_name(name)
    assert_name_available(name)
    for i in range(1, len(trials) + 1):
        assert_name_available(trial_dir_name(name, i))

    planned = [TrialRow(i, trial_dir_name(name, i), t, "planned", None) for i, t in enumerate(trials, start=1)]
    axis_names = list(axes)

    print(_console.rule(79, "white"))
    print(f" {_console.bold(f'CVBench — sweep {name}')}")
    print(_console.rule(79, "white"))
    print(f" {len(trials)} trials ({' x '.join(f'{len(v)} {display_key(k)}' for k, v in axes.items())}), "
          f"strategy {strategy}, metric {metric} ({direction})")
    if fixed_raw:
        print(_console.dim(" fixed: " + ", ".join(f"{display_key(k)}={v}" for k, v in fixed_raw.items())))
    if strategy == "random":
        print(_console.dim(f" sample-seed {sample_seed} (recorded in sweep.yaml; reuse it to repeat these trials)"))
    if show:
        print_trial_table(planned, axis_names, None)
        print(_console.dim(f" --show: nothing was trained. Total: {len(trials)} trials."))
        return

    from cvbench.services.sweep import run_sweep  # deferred: pulls in TensorFlow

    manifest = SweepManifest(
        version=1, name=name, date=date.today().strftime("%Y-%m-%d"), data_dir=str(data_dir),
        strategy=strategy, metric=metric, direction=direction, axes=axes,
        trials=trials if strategy == "random" else None,
        sample_seed=sample_seed if strategy == "random" else None,
    )
    run_sweep(manifest, trials, trial_values, task_name=task, fixed=fixed, from_dir=from_dir,
              listener=_SweepPrinter())


class _SweepPrinter(ConsoleListener):
    """Terminal output of a sweep: the trial headers, failures and the final ranking."""

    def __call__(self, event: Event) -> None:
        if isinstance(event, TrialStarted):
            print()
            print(f" {_console.bold(f'[{event.index}/{event.total}] {event.name}')}  "
                  + ", ".join(f"{display_key(k)}={v}" for k, v in event.params.items()))
        elif isinstance(event, TrialFailed):
            what = "failed" if event.stage == "training" else "trained but val evaluation failed"
            _console.error(f"Trial {event.name} {what}: {event.error_type}: {event.message}")
        elif isinstance(event, SweepInterrupted):
            detail = (f"trial {event.name} did not finish" if event.stage == "training"
                      else f"evaluating {event.name}")
            _console.warning(f"\n Interrupted — stopping the sweep ({detail}).")
        elif isinstance(event, SweepFinished):
            _finish(Path(event.sweep_dir))
        else:
            super().__call__(event)


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
        score = present.fmt_metric(manifest.metric, best.value)
        print(f" Best: {_console.green(best.dir)} ({manifest.metric} = {score})")
        print(_console.dim(f" Inspect it with: runs show {best.dir}"))
        print(_console.dim(f" Evaluate the best trial on the test split:  evaluate {best.dir}"))
        print(_console.dim(f" Evaluate all trials on the test split:      evaluate {manifest.name}"))


if __name__ == "__main__":
    sweep(prog_name="sweep", args=sys.argv[1:])
