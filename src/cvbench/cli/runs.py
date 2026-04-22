import click

from cvbench.core import _fmt
from cvbench.core.runs import scan_experiments, best_experiment, resolve_run_dir, EXPERIMENTS_DIR
from cvbench.core.config import load_config
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
@click.option("--sort", default="date",
              type=click.Choice(["val_accuracy", "val_loss", "date", "backbone"]),
              show_default=True)
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
        name = _fit(r['name'], 45)
        print(f" {name:<45} {r.get('status', '?'):<12} {loss_str:>9}  {r.get('epochs_run', '?'):>6}")
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
        "backbone", "lr",
        "epochs", "val_loss", "val_accuracy", "test_accuracy", "epochs_run", "status", "date",
    ]
    name_a = a.get("name", run_a)
    name_b = b.get("name", run_b)

    col_w = 26
    tr = _fmt.rule(79, "white")
    print(tr)
    print(f" {'Field':<22}  {_fit(name_a, col_w):<{col_w}}  {_fit(name_b, col_w):<{col_w}}")
    print(tr)
    for f in fields:
        va = str(a.get(f, "—"))
        vb = str(b.get(f, "—"))
        diff = " ◀" if va != vb else ""
        print(f" {f:<22}  {va:<26}  {vb:<26}{diff}")
    print(tr)


@runs.command()
@click.argument("experiment")
@click.option("--format", "fmt", required=True,
              type=click.Choice(["tflite", "onnx", "plan"]),
              help="Export format (plan prints Jetson TensorRT instructions).")
@click.option("--quantize", default="none",
              type=click.Choice(["none", "float16", "int8"]),
              show_default=True,
              help="TFLite quantization mode (ignored for ONNX and plan).")
@click.option("--output", "output_dir", default=None,
              help="Output directory (default: <experiment>/export/). Ignored for plan.")
def export(experiment, fmt, quantize, output_dir):
    """Export the best checkpoint of EXPERIMENT to TFLite or ONNX, or print Jetson deployment instructions (plan).

    EXPERIMENT is a run name or full path to a run directory.
    """
    try:
        run_export(experiment, format=fmt, quantize=quantize, output_dir=output_dir)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))


@runs.command()
@click.argument("experiments_dir", default=_DEFAULT_EXPERIMENTS_DIR)
@click.option("--metric", default="val_loss",
              type=click.Choice(["val_loss", "val_accuracy", "test_accuracy"]),
              show_default=True)
def best(experiments_dir, metric):
    """Show the best experiment in EXPERIMENTS_DIR by a given metric."""
    run = best_experiment(experiments_dir, metric)
    if run is None:
        print(f" No experiments with metric '{metric}' found in '{experiments_dir}'.")
        return

    print(_fmt.rule())
    print(f" {_fmt.bold(f'CVBench — best run by {metric}')}")
    print(_fmt.rule())
    for k, v in run.items():
        print(f" {k:<22}: {v}")
    print(_fmt.rule())
