import json
import os
import shutil

import click
import numpy as np

from cvbench.cli import _help
from cvbench.core import console, present
from cvbench.core.config import load_config, update_run_status
from cvbench.core.confusion import print_confusion_matrix
from cvbench.core.exp_store import (
    EXPERIMENTS_DIR,
    assert_name_available,
    assert_renamable,
    is_sweep_dir,
    resolve_experiments_dir,
    resolve_run_dir,
    scan_experiments,
    validate_run_name,
)
from cvbench.core.sweep_store import scan_sweeps
from cvbench.tasks import TASK_NAMES, get_task

# NOTE: cvbench.services.export is imported inside export() — it pulls in
# TensorFlow, and importing it at module scope would make every `runs …`
# invocation (and `runs --help`, and the `commands` overview) take ~1.5s.


# Settings `runs show` always lists, even at their default — the knobs people compare runs on.
_ALWAYS_SHOWN = (
    "model.fine_tune_from_layer",
    "training.loss.type",
    "training.optimizer.type",
    "training.seed",
)
# Run bookkeeping (shown above) and bulky/derived values that would only add noise.
_HIDDEN_PREFIXES = (
    "run.",
    "augmentation.",
    "data.classes",
    *(p for name in TASK_NAMES for p in get_task(name).hidden_config_prefixes),
)


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}."))
        else:
            flat[key] = v
    return flat


def _config_lines(cfg) -> list[tuple[str, object]]:
    """(key, value) pairs for every non-default config setting, plus the always-shown knobs."""
    import dataclasses

    from cvbench.core.config import CVBenchConfig

    current = _flatten(dataclasses.asdict(cfg))
    default = _flatten(dataclasses.asdict(CVBenchConfig()))
    lines = []
    for key, value in current.items():
        if key.endswith("_explicit"):
            continue
        if key in _ALWAYS_SHOWN or (
            value != default.get(key) and not key.startswith(_HIDDEN_PREFIXES)
        ):
            lines.append((key, value))
    names = [getattr(t, "name", None) or "one_of" for t in cfg.augmentation.transforms]
    lines.append(("augmentation", ", ".join(names) if names else "none"))
    return lines


_DEFAULT_EXPERIMENTS_DIR = EXPERIMENTS_DIR


@_help.group(
    examples=[
        ("runs list", "every run, newest first"),
        ("runs export my_run --format tflite", "package a run for a device"),
    ],
)
def runs():
    """Manage and inspect runs."""


@runs.command(
    "list",
    short_help="List runs (default: experiments/).",
    examples=[
        ("runs list", "All runs, newest first"),
        ("runs list --sort val_loss", "Order by validation loss instead of date"),
    ],
    see_also=[("evaluate <run>", "score a run on the test split")],
)
@click.argument("experiments_dir", metavar="RUNS_DIR", default=_DEFAULT_EXPERIMENTS_DIR)
@click.option(
    "--sort",
    default="date",
    type=click.Choice(["val_accuracy", "val_loss", "test", "date", "backbone"]),
    show_default=True,
)
def list_runs(experiments_dir, sort):
    """List runs in RUNS_DIR (default: experiments/).

    RUNS_DIR may also be a sweep name to list that sweep's trials.
    """
    experiments_dir = resolve_experiments_dir(experiments_dir)
    entries = scan_experiments(experiments_dir, sort_by=sort)
    sweeps = scan_sweeps(experiments_dir)
    if sort == "date":
        # scan_experiments sorts newest first; slot sweeps in by their own date.
        entries = sorted(entries + sweeps, key=lambda r: r.get("date") or "", reverse=True)
    else:
        entries = entries + sweeps
    if not entries:
        print(f" {present.EMPTY_RUNS.format(dir=experiments_dir)}")
        return

    displays = [present.present_run(r) for r in entries]
    columns = present.run_columns(console.width())
    console.table(
        [(label, align) if align == "right" else label for _, label, align in columns],
        [
            [d["name"] if key == "name" else present.run_cell(d, key) for key, _, _ in columns]
            for d in displays
        ],
    )
    if sweeps:
        print(console.dim(" Sweeps: `runs list <sweep>` shows their trials."))


def _show_sweep(sweep_dir):
    """`runs show` for a sweep: manifest intent plus the trial table (no per-run details)."""
    from cvbench.cli.sweep import print_trial_table
    from cvbench.core.sweep_store import best_trial, display_key, summarize

    manifest, rows = summarize(sweep_dir)

    print(console.rule())
    print(f" {console.bold(f'CVBench — sweep {manifest.name}')}")
    print(console.rule())
    for label, value in [
        ("Date", manifest.date),
        ("Data", manifest.data_dir),
        ("Strategy", manifest.strategy),
        ("Metric", f"{manifest.metric} ({manifest.direction})"),
        ("Axes", ", ".join(f"{display_key(k)}={','.join(v)}" for k, v in manifest.axes.items())),
    ]:
        print(f" {console.dim(f'{label:<10}')} {value}")
    print()
    print_trial_table(rows, list(manifest.axes), manifest.metric, show_test=True)
    best = best_trial(rows)
    if best is not None:
        score = present.fmt_metric(manifest.metric, best.value)
        print(f" Best: {console.green(best.dir)} ({manifest.metric} = {score})")
        print(console.dim(f" Inspect a trial with: runs show {best.dir}"))


@runs.command(
    short_help="Show full details for a single run.",
    examples=[
        ("runs show my_run", "Config, metrics, exports, and eval results for one run"),
    ],
    see_also=[("evaluate <run>", "score a run on the test split")],
)
@click.argument("experiment", metavar="RUN")
def show(experiment):
    """Show full config, metrics, exports, and eval results for RUN.

    RUN is a run name or full path to a run directory. For a sweep, shows its
    settings and trial table instead.
    """
    from pathlib import Path

    from cvbench.core.exp_store import _read_entry

    run_dir = Path(resolve_run_dir(experiment, allow_sweep=True))

    if is_sweep_dir(run_dir):
        _show_sweep(run_dir)
        return

    cfg = load_config(str(run_dir))

    entry = _read_entry(run_dir)
    run_name = entry.get("name", run_dir.name)

    print(console.rule())
    print(f" {console.bold(f'CVBench — run {run_name}')}")
    print(console.rule())
    for label, value in present.run_fields(entry):
        print(f" {label:<22}: {value}")
    print(console.rule())

    print(f" {console.bold(present.SECTIONS['config'])} {console.dim('(non-default values)')}")
    for key, value in _config_lines(cfg):
        print(f"   {key:<30}: {value}")
    if cfg.run.cli_command:
        print(f"   {'command':<30}: {cfg.run.cli_command}")
    print(console.rule())

    export_dir = run_dir / "export"
    exports = sorted(d.name for d in export_dir.iterdir() if d.is_dir()) if export_dir.is_dir() else []
    print(f" {console.bold(present.SECTIONS['exports'])}")
    if exports:
        for name in exports:
            print(f"   {name}")
    else:
        print(console.dim("   None"))
    print(console.rule())

    eval_path = run_dir / "eval_report.json"
    print(f" {console.bold(present.SECTIONS['evaluation'])}")
    if eval_path.exists():
        with open(eval_path) as f:
            report = json.load(f)
        task_name = report.get("task")
        advisories = present.run_advisories(run_dir, report)
        if task_name in TASK_NAMES:
            get_task(task_name).view().render_report(report, entry.get("val_accuracy"), advisories)
        else:
            # Unknown/legacy report shape — fall back to the generic
            # overall/per-class/confusion-matrix rendering.
            overall = report.get("overall", {})
            print(f"   {overall.get('label', 'Overall')}: {overall.get('value')}")
            per_class = report.get("per_class", {})
            if per_class:
                rows = [
                    (cls, m.get("precision", "—"), m.get("recall", "—"), m.get("f1", "—"), m.get("support", "—"))
                    for cls, m in per_class.items()
                ]
                console.table(
                    ["Class", ("Precision", "right"), ("Recall", "right"), ("F1", "right"), ("Support", "right")],
                    rows,
                )
            confusion_matrix = report.get("confusion_matrix")
            if confusion_matrix:
                print()
                print_confusion_matrix(
                    np.array(confusion_matrix["matrix"]), confusion_matrix["classes"]
                )
    else:
        print(console.dim("   No eval_report.json found. Run: evaluate " + run_dir.name))
    print(console.rule())


@runs.command(
    short_help="Rename a run directory and update its config.",
    examples=[
        ("runs rename cls_effnet_b0_2026_01_21 baseline",
         "Give a run a memorable name"),
    ],
)
@click.argument("experiment", metavar="RUN")
@click.argument("new_name")
def rename(experiment, new_name):
    """Rename a run directory and update its config.

    RUN is a run name or full path. NEW_NAME must contain only letters,
    digits, underscores, and hyphens.
    """
    from pathlib import Path

    run_dir = Path(resolve_run_dir(experiment, allow_sweep=True))

    assert_renamable(run_dir)

    cfg = load_config(str(run_dir))
    if cfg.run.status == "running":
        raise click.ClickException("Cannot rename a currently running run.")

    validate_run_name(new_name)
    assert_name_available(new_name, current_dir=run_dir)

    new_dir = run_dir.parent / new_name
    os.rename(run_dir, new_dir)
    update_run_status(str(new_dir), name=new_name)
    print(console.green(f" Renamed '{run_dir.name}' → '{new_name}'."))


@runs.command(
    short_help="Export a run to TFLite / ONNX / Hailo, or print Jetson steps.",
    examples=[
        ("runs export my_run --format tflite", "Plain float TFLite model"),
        ("runs export my_run --format tflite --quantize int8", "Dynamic-range int8 TFLite (float32 I/O, smaller file)"),
        ("runs export my_run --format onnx", "ONNX model for onnxruntime"),
        ("runs export my_run --format hailo", "Prepare a Hailo compilation package"),
        ("runs export my_run --format plan", "Print Jetson TensorRT build instructions"),
    ],
    see_also=[("predict my_run images/ --format tflite", "test the exported model")],
)
@click.argument("experiment", metavar="RUN")
@click.option(
    "--format",
    "fmt",
    required=True,
    type=click.Choice(["tflite", "onnx", "plan", "hailo"]),
    help=(
        "Export format (plan prepares model.onnx and prints Jetson TensorRT steps; hailo prepares"
        " a Hailo Docker package). onnx/plan need the 'export' extra: pip install 'cvbench[export]'."
        " Exported models take float32 NHWC input, values 0-255; normalisation is inside the model."
    ),
)
@click.option(
    "--quantize",
    default="none",
    type=click.Choice(["none", "float16", "int8"]),
    show_default=True,
    help=(
        "TFLite quantization mode (ignored for other formats). int8 is dynamic-range:"
        " int8 weights with float32 input/output, not full-integer."
    ),
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    help=(
        "Parent directory for tflite/onnx exports; a <format> subfolder is created inside"
        " (default: <run>/export/). Ignored for plan and hailo."
    ),
)
@click.option(
    "--hw-arch",
    "hw_arch",
    default="hailo8l",
    type=click.Choice(["hailo8", "hailo8l", "hailo8r"]),
    show_default=True,
    help="Hailo target architecture used in the printed compile steps (hailo format only).",
)
@click.option(
    "--calib-total",
    "calib_total",
    default=1024,
    type=int,
    show_default=True,
    help="Target total images in the Hailo calibration set.",
)
@click.option(
    "--calib-strategy",
    "calib_strategy",
    default="stratified",
    type=click.Choice(["stratified", "proportional", "equal", "diverse"]),
    show_default=True,
    help=(
        "How to distribute calibration samples: stratified (equal per class + k-means within"
        " each class, recommended), proportional (by class size), equal (same per class),"
        " diverse (k-means across all images)."
    ),
)
def export(experiment, fmt, quantize, output_dir, hw_arch, calib_total, calib_strategy):
    """Export the best checkpoint of RUN to TFLite, ONNX, or Hailo package,
    or print Jetson deployment instructions (plan).

    RUN is a run name or full path to a run directory.
    """
    from cvbench.services.export import run_export  # deferred: pulls in TensorFlow

    run_export(
        experiment, format=fmt, quantize=quantize, output_dir=output_dir,
        calib_total=calib_total, calib_strategy=calib_strategy, hw_arch=hw_arch,
    )


@runs.command(
    short_help="Delete a run, or just one of its exports.",
    examples=[
        ("runs delete old_run --yes", "Remove a run and everything in it"),
        ("runs delete my_run --export tflite", "Remove only the tflite export, keep the run"),
    ],
)
@click.argument("experiment", metavar="RUN")
@click.option(
    "--export",
    "export_subfolder",
    default=None,
    metavar="SUBFOLDER",
    help="Delete only this export subfolder (e.g. tflite, onnx, hailo). Omit to delete the entire run.",
)
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def delete(experiment, export_subfolder, yes):
    """Delete a run or one of its exports.

    RUN is a run name or full path to a run directory.

    Without --export, the entire run directory is removed.
    With --export SUBFOLDER, only that export subfolder is removed.
    """
    from pathlib import Path

    run_dir = Path(resolve_run_dir(experiment, allow_sweep=True))

    if export_subfolder:
        export_base = run_dir / "export"
        target = (export_base / export_subfolder).resolve()
        try:
            target.relative_to(export_base.resolve())
        except ValueError:
            raise click.ClickException("Invalid export subfolder.") from None
        if not target.is_dir():
            raise click.ClickException(
                f"Export '{export_subfolder}' not found in {run_dir.name}."
            )
        label = f"export '{export_subfolder}' from run '{run_dir.name}'"
    else:
        target = run_dir
        if is_sweep_dir(run_dir):
            n_trials = sum(1 for d in run_dir.iterdir() if d.is_dir())
            label = f"sweep '{run_dir.name}' and its {n_trials} trial(s)"
        else:
            label = f"run '{run_dir.name}' and all its contents"

    if not yes:
        click.confirm(
            f"{console.yellow('Warning:')} This will permanently delete {label}. Continue?",
            abort=True,
        )

    shutil.rmtree(target)
    print(console.green(f" Deleted {label}."))



