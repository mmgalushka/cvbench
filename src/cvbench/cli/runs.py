import json
import os
import shutil

import click
import numpy as np

from cvbench.cli import _help
from cvbench.core import _console
from cvbench.core._confusion import print_confusion_matrix
from cvbench.core.config import load_config, update_run_status
from cvbench.core.exp_store import (
    EXPERIMENTS_DIR,
    assert_name_available,
    best_experiment,
    resolve_run_dir,
    scan_experiments,
    validate_run_name,
)

# NOTE: cvbench.services.export is imported inside export() — it pulls in
# TensorFlow, and importing it at module scope would make every `runs …`
# invocation (and `runs --help`, and the `commands` overview) take ~1.5s.


def _fit(s: str, width: int) -> str:
    """Fit string to width using a middle ellipsis, preserving start and end."""
    if len(s) <= width:
        return s
    keep = width - 1  # 1 char for the ellipsis
    head = keep // 2
    tail = keep - head
    return s[:head] + "…" + s[-tail:]


_DEFAULT_EXPERIMENTS_DIR = EXPERIMENTS_DIR


@_help.group(
    examples=[
        ("runs list", "every run, newest first"),
        ("runs best --metric val_accuracy", "the strongest run so far"),
        ("runs export my_run --format tflite", "package a run for a device"),
    ],
)
def runs():
    """Manage and inspect experiment runs."""


@runs.command(
    "list",
    short_help="List experiment runs (default: experiments/).",
    examples=[
        ("runs list", "All runs, newest first"),
        ("runs list --sort val_loss", "Order by validation loss instead of date"),
    ],
    see_also=[("evaluate <run>", "score a run on the test split"),
              ("runs compare <a> <b>", "put two runs side by side")],
)
@click.argument("experiments_dir", default=_DEFAULT_EXPERIMENTS_DIR)
@click.option(
    "--sort",
    default="date",
    type=click.Choice(["val_accuracy", "val_loss", "date", "backbone"]),
    show_default=True,
)
def list_runs(experiments_dir, sort):
    """List experiments in EXPERIMENTS_DIR (default: experiments/)."""
    entries = scan_experiments(experiments_dir, sort_by=sort)
    if not entries:
        print(f" No experiments found in '{experiments_dir}'.")
        return

    rows = []
    for r in entries:
        loss = r.get("val_loss")
        loss_str = f"{loss:.4f}" if loss is not None else "—"
        task_short = "det" if r.get("task") == "detection" else "cls"
        rows.append((_fit(r["name"], 40), task_short, r.get("status", "?"), loss_str, r.get("epochs_run", "?")))
    _console.table(
        ["Run", "Task", "Status", ("Val Loss", "right"), ("Epochs", "right")],
        rows,
    )


@runs.command(
    short_help="Compare two runs side by side.",
    examples=[
        ("runs compare cls_effnet_b0_2026_01_20 cls_effnet_b3_2026_01_21",
         "Diff hyperparameters and metrics for two runs"),
    ],
)
@click.argument("experiment_a")
@click.argument("experiment_b")
def compare(experiment_a, experiment_b):
    """Compare two experiments side by side.

    EXPERIMENT_A and EXPERIMENT_B are run names (e.g. cls_effnet_b3_lr5e5_2026_01_21)
    or full paths to run directories. Bare names are resolved under experiments/.
    """
    run_a = resolve_run_dir(experiment_a)
    run_b = resolve_run_dir(experiment_b)
    try:
        load_config(run_a)
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_a}") from None
    try:
        load_config(run_b)
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_b}") from None

    from pathlib import Path

    from cvbench.core.exp_store import _read_entry

    a = _read_entry(Path(run_a))
    b = _read_entry(Path(run_b))

    fields = [
        "backbone",
        "lr",
        "epochs",
        "val_loss",
        "val_accuracy",
        "test_accuracy",
        "epochs_run",
        "status",
        "date",
    ]
    name_a = a.get("name", run_a)
    name_b = b.get("name", run_b)

    col_w = 26
    tr = _console.rule(79, "white")
    print(tr)
    print(
        f" {'Field':<22}  {_fit(name_a, col_w):<{col_w}}  {_fit(name_b, col_w):<{col_w}}"
    )
    print(tr)
    for f in fields:
        va = str(a.get(f, "—"))
        vb = str(b.get(f, "—"))
        diff = " ◀" if va != vb else ""
        print(f" {f:<22}  {va:<26}  {vb:<26}{diff}")
    print(tr)


@runs.command(
    short_help="Show full details for a single run.",
    examples=[
        ("runs show my_run", "Config, metrics, exports, and eval results for one run"),
    ],
    see_also=[("runs compare <a> <b>", "put two runs side by side"),
              ("evaluate <run>", "score a run on the test split")],
)
@click.argument("experiment")
def show(experiment):
    """Show full config, metrics, exports, and eval results for EXPERIMENT.

    EXPERIMENT is a run name or full path to a run directory.
    """
    from pathlib import Path

    from cvbench.core.exp_store import _read_entry

    try:
        run_dir = Path(resolve_run_dir(experiment))
    except Exception as e:
        raise click.ClickException(str(e)) from e

    try:
        load_config(str(run_dir))
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_dir}") from None

    entry = _read_entry(run_dir)
    run_name = entry.get("name", run_dir.name)

    fields = [
        "backbone",
        "lr",
        "epochs",
        "val_loss",
        "val_accuracy",
        "test_accuracy",
        "epochs_run",
        "status",
        "date",
    ]

    print(_console.rule())
    print(f" {_console.bold(f'CVBench — run {run_name}')}")
    print(_console.rule())
    for f in fields:
        print(f" {f:<22}: {entry.get(f, '—')}")
    print(_console.rule())

    export_dir = run_dir / "export"
    exports = sorted(d.name for d in export_dir.iterdir() if d.is_dir()) if export_dir.is_dir() else []
    print(f" {_console.bold('Exports')}")
    if exports:
        for name in exports:
            print(f"   {name}")
    else:
        print(_console.dim("   None"))
    print(_console.rule())

    eval_path = run_dir / "eval_report.json"
    print(f" {_console.bold('Evaluation')}")
    if eval_path.exists():
        with open(eval_path) as f:
            report = json.load(f)
        overall = report.get("overall", {})
        print(f"   {overall.get('label', 'Overall')}: {overall.get('value')}")
        per_class = report.get("per_class", {})
        if per_class:
            rows = [
                (cls, m.get("precision", "—"), m.get("recall", "—"), m.get("f1", "—"), m.get("support", "—"))
                for cls, m in per_class.items()
            ]
            _console.table(
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
        print(_console.dim("   No eval_report.json found. Run: evaluate " + run_dir.name))
    print(_console.rule())


@runs.command(
    short_help="Rename a run directory and update its config.",
    examples=[
        ("runs rename cls_effnet_b0_2026_01_21 baseline",
         "Give a run a memorable name"),
    ],
)
@click.argument("experiment")
@click.argument("new_name")
def rename(experiment, new_name):
    """Rename an experiment directory and update its config.

    EXPERIMENT is a run name or full path. NEW_NAME must contain only letters,
    digits, underscores, and hyphens.
    """
    from pathlib import Path

    try:
        run_dir = Path(resolve_run_dir(experiment))
    except Exception as e:
        raise click.ClickException(str(e)) from e

    cfg = load_config(str(run_dir))
    if cfg.run.status == "running":
        raise click.ClickException("Cannot rename a currently running experiment.")

    try:
        validate_run_name(new_name)
        assert_name_available(new_name, current_dir=run_dir)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    new_dir = run_dir.parent / new_name
    os.rename(run_dir, new_dir)
    update_run_status(str(new_dir), name=new_name)
    print(_console.green(f" Renamed '{run_dir.name}' → '{new_name}'."))


@runs.command(
    short_help="Export a run to TFLite / ONNX / Hailo, or print Jetson steps.",
    examples=[
        ("runs export my_run --format tflite", "Plain float TFLite model"),
        ("runs export my_run --format tflite --quantize int8", "Quantized TFLite for microcontrollers"),
        ("runs export my_run --format onnx", "ONNX model for onnxruntime"),
        ("runs export my_run --format hailo", "Prepare a Hailo compilation package"),
        ("runs export my_run --format plan", "Print Jetson TensorRT build instructions"),
    ],
    see_also=[("predict my_run images/ --format tflite", "test the exported model")],
)
@click.argument("experiment")
@click.option(
    "--format",
    "fmt",
    required=True,
    type=click.Choice(["tflite", "onnx", "plan", "hailo"]),
    help="Export format (plan prints Jetson TensorRT instructions; hailo prepares Hailo Docker package).",
)
@click.option(
    "--quantize",
    default="none",
    type=click.Choice(["none", "float16", "int8"]),
    show_default=True,
    help="TFLite quantization mode (ignored for ONNX and plan).",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    help="Output directory (default: <experiment>/export/). Ignored for plan.",
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
def export(experiment, fmt, quantize, output_dir, calib_total, calib_strategy):
    """Export the best checkpoint of EXPERIMENT to TFLite, ONNX, or Hailo package,
    or print Jetson deployment instructions (plan).

    EXPERIMENT is a run name or full path to a run directory.
    """
    from cvbench.services.export import run_export  # deferred: pulls in TensorFlow

    try:
        run_export(
            experiment, format=fmt, quantize=quantize, output_dir=output_dir,
            calib_total=calib_total, calib_strategy=calib_strategy,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e


@runs.command(
    short_help="Delete a run, or just one of its exports.",
    examples=[
        ("runs delete old_run --yes", "Remove a run and everything in it"),
        ("runs delete my_run --export tflite", "Remove only the tflite export, keep the run"),
    ],
)
@click.argument("experiment")
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

    EXPERIMENT is a run name or full path to a run directory.

    Without --export, the entire run directory is removed.
    With --export SUBFOLDER, only that export subfolder is removed.
    """
    from pathlib import Path

    try:
        run_dir = Path(resolve_run_dir(experiment))
    except Exception as e:
        raise click.ClickException(str(e)) from e

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
        label = f"run '{run_dir.name}' and all its contents"

    if not yes:
        click.confirm(
            f"{_console.yellow('Warning:')} This will permanently delete {label}. Continue?",
            abort=True,
        )

    shutil.rmtree(target)
    print(_console.green(f" Deleted {label}."))


@runs.command(
    short_help="Show the single best run by a metric.",
    examples=[
        ("runs best", "Best run by validation loss"),
        ("runs best --metric test_accuracy", "Best run by test accuracy"),
    ],
)
@click.argument("experiments_dir", default=_DEFAULT_EXPERIMENTS_DIR)
@click.option(
    "--metric",
    default="val_loss",
    type=click.Choice(["val_loss", "val_accuracy", "test_accuracy"]),
    show_default=True,
)
def best(experiments_dir, metric):
    """Show the best experiment in EXPERIMENTS_DIR by a given metric."""
    run = best_experiment(experiments_dir, metric)
    if run is None:
        print(
            f" No experiments with metric '{metric}' found in '{experiments_dir}'."
        )
        return

    print(_console.rule())
    print(f" {_console.bold(f'CVBench — best run by {metric}')}")
    print(_console.rule())
    for k, v in run.items():
        print(f" {k:<22}: {v}")
    print(_console.rule())


