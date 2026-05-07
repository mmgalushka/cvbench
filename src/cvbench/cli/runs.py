import os
import shutil

import click

from cvbench.core import _fmt
from cvbench.core.runs import (
    scan_experiments,
    best_experiment,
    resolve_run_dir,
    validate_run_name,
    assert_name_available,
    EXPERIMENTS_DIR,
)
from cvbench.core.config import load_config, update_run_status
from cvbench.services.export import run_export


def _fit(s: str, width: int) -> str:
    """Fit string to width using a middle ellipsis, preserving start and end."""
    if len(s) <= width:
        return s
    keep = width - 1  # 1 char for the ellipsis
    head = keep // 2
    tail = keep - head
    return s[:head] + "…" + s[-tail:]


_DEFAULT_EXPERIMENTS_DIR = EXPERIMENTS_DIR


@click.group()
def runs():
    """Manage and inspect experiment runs."""


@runs.command("list")
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

    tr = _fmt.rule(76, "white")
    print(tr)
    print(f" {'Run':<45} {'Status':<12} {'Val Loss':>9}  {'Epochs':>6}")
    print(tr)
    for r in entries:
        loss = r.get("val_loss")
        loss_str = f"{loss:.4f}" if loss is not None else "   —   "
        name = _fit(r["name"], 45)
        print(
            f" {name:<45} {r.get('status', '?'):<12} {loss_str:>9}  {r.get('epochs_run', '?'):>6}"
        )
    print(tr)


@runs.command()
@click.argument("experiment_a")
@click.argument("experiment_b")
def compare(experiment_a, experiment_b):
    """Compare two experiments side by side.

    EXPERIMENT_A and EXPERIMENT_B are run names (e.g. effnet_b3_lr5e5_trial_2024_01_21)
    or full paths to run directories. Bare names are resolved under experiments/.
    """
    run_a = resolve_run_dir(experiment_a)
    run_b = resolve_run_dir(experiment_b)
    try:
        a_cfg = load_config(run_a)
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_a}")
    try:
        b_cfg = load_config(run_b)
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_b}")

    from cvbench.core.runs import _read_entry
    from pathlib import Path

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
    tr = _fmt.rule(79, "white")
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


@runs.command()
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
        raise click.ClickException(str(e))

    cfg = load_config(str(run_dir))
    if cfg.run.status == "running":
        raise click.ClickException("Cannot rename a currently running experiment.")

    try:
        validate_run_name(new_name)
        assert_name_available(new_name, current_dir=run_dir)
    except ValueError as e:
        raise click.ClickException(str(e))

    new_dir = run_dir.parent / new_name
    os.rename(run_dir, new_dir)
    update_run_status(str(new_dir), name=new_name)
    print(_fmt.green(f" Renamed '{run_dir.name}' → '{new_name}'."))


@runs.command()
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
    "--calib-samples-per-class",
    "calib_samples_per_class",
    default=None,
    type=int,
    help="Images per class in the Hailo calibration set. Overrides the default stratified heuristic.",
)
def export(experiment, fmt, quantize, output_dir, calib_samples_per_class):
    """Export the best checkpoint of EXPERIMENT to TFLite, ONNX, or Hailo package, or print Jetson deployment instructions (plan).

    EXPERIMENT is a run name or full path to a run directory.
    """
    try:
        run_export(
            experiment, format=fmt, quantize=quantize, output_dir=output_dir,
            calib_samples_per_class=calib_samples_per_class,
        )
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))


@runs.command()
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
        raise click.ClickException(str(e))

    if export_subfolder:
        export_base = run_dir / "export"
        target = (export_base / export_subfolder).resolve()
        try:
            target.relative_to(export_base.resolve())
        except ValueError:
            raise click.ClickException("Invalid export subfolder.")
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
            f"{_fmt.yellow('Warning:')} This will permanently delete {label}. Continue?",
            abort=True,
        )

    shutil.rmtree(target)
    print(_fmt.green(f" Deleted {label}."))


@runs.command()
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

    print(_fmt.rule())
    print(f" {_fmt.bold(f'CVBench — best run by {metric}')}")
    print(_fmt.rule())
    for k, v in run.items():
        print(f" {k:<22}: {v}")
    print(_fmt.rule())


