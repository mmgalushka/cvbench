import click

from core.runs import scan_experiments, best_experiment
from core.config import load_config


_DEFAULT_EXPERIMENTS_DIR = "experiments"


@click.group()
def runs():
    """Manage and inspect experiment runs."""


@runs.command("list")
@click.argument("experiments_dir", default=_DEFAULT_EXPERIMENTS_DIR)
@click.option("--sort", default="date",
              type=click.Choice(["val_accuracy", "date", "backbone"]),
              show_default=True)
def list_runs(experiments_dir, sort):
    """List experiments in EXPERIMENTS_DIR (default: experiments/)."""
    entries = scan_experiments(experiments_dir, sort_by=sort)
    if not entries:
        print(f" No experiments found in '{experiments_dir}'.")
        return

    w = 80
    print("━" * w)
    print(f" {'Run':<45} {'Status':<12} {'Val Acc':>8}  {'Epochs':>6}")
    print("━" * w)
    for r in entries:
        acc = r.get("val_accuracy")
        acc_str = f"{acc:.4f}" if acc is not None else "  —   "
        print(f" {r['name']:<45} {r.get('status', '?'):<12} {acc_str:>8}  {r.get('epochs_run', '?'):>6}")
    print("━" * w)


@runs.command()
@click.argument("run_a", type=click.Path(exists=True))
@click.argument("run_b", type=click.Path(exists=True))
def compare(run_a, run_b):
    """Compare two experiment directories side by side."""
    try:
        a_cfg = load_config(run_a)
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_a}")
    try:
        b_cfg = load_config(run_b)
    except FileNotFoundError:
        raise click.ClickException(f"No config.yaml in: {run_b}")

    from core.runs import _read_entry
    from pathlib import Path
    a = _read_entry(Path(run_a))
    b = _read_entry(Path(run_b))

    fields = [
        "backbone", "lr", "aug_placement",
        "epochs", "val_accuracy", "test_accuracy", "epochs_run", "status", "date",
    ]
    name_a = a.get("name", run_a)
    name_b = b.get("name", run_b)

    w = 75
    print("━" * w)
    print(f" {'Field':<22} {name_a[:24]:<26} {name_b[:24]:<26}")
    print("━" * w)
    for f in fields:
        va = str(a.get(f, "—"))
        vb = str(b.get(f, "—"))
        diff = " ◀" if va != vb else ""
        print(f" {f:<22} {va:<26} {vb:<26}{diff}")
    print("━" * w)


@runs.command()
@click.argument("experiments_dir", default=_DEFAULT_EXPERIMENTS_DIR)
@click.option("--metric", default="val_accuracy",
              type=click.Choice(["val_accuracy", "val_loss", "test_accuracy"]),
              show_default=True)
def best(experiments_dir, metric):
    """Show the best experiment in EXPERIMENTS_DIR by a given metric."""
    run = best_experiment(experiments_dir, metric)
    if run is None:
        print(f" No experiments with metric '{metric}' found in '{experiments_dir}'.")
        return

    w = 55
    print("━" * w)
    print(f" CVBench — best run by {metric}")
    print("━" * w)
    for k, v in run.items():
        print(f" {k:<22}: {v}")
    print("━" * w)
