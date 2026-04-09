import contextlib
import io
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import click
import keras

from core.config import load_config
from core.data import build_dataset, get_class_names
from core import evaluator as _evaluator

_EXPERIMENTS_DIR = "experiments"


def _resolve_run_dir(name: str) -> str:
    """Resolve a run name or path to an existing directory.

    Accepts either a full path (experiments/my_run) or just the run name (my_run).
    If the given value does not exist as-is, looks under 'experiments/'.
    """
    p = Path(name)
    if p.exists():
        return str(p)
    candidate = Path(_EXPERIMENTS_DIR) / name
    if candidate.exists():
        return str(candidate)
    raise click.BadParameter(
        f"Run directory not found: '{name}' (also tried '{candidate}')",
        param_hint="RUN_DIR",
    )


@click.command()
@click.argument("run_dir")
@click.option("--output-dir", default=None, help="Where to write eval outputs (default: run_dir).")
def evaluate(run_dir, output_dir):
    """Evaluate a trained model in RUN_DIR on the held-out test split.

    RUN_DIR can be a full path (experiments/my_run) or just the run name (my_run).
    """
    run_dir = _resolve_run_dir(run_dir)

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        names = ", ".join(g.name for g in gpus)
        print(f"\033[92m🟢 GPU      : {len(gpus)} device(s) — {names}\033[0m")
    else:
        print(f"\033[93m⚠️  GPU      : not available — evaluating on CPU\033[0m")

    cfg = load_config(run_dir)

    class_names = get_class_names(cfg.data.train_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        test_ds = build_dataset(cfg.data.test_dir, class_names, cfg, training=False)

    n_test = sum(1 for _ in Path(cfg.data.test_dir).glob("*/*"))
    print(f" Found {n_test} files for evaluation ({len(class_names)} classes).")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model = keras.saving.load_model(f"{run_dir}/best.keras")

    _evaluator.evaluate(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        run_dir=run_dir,
        output_dir=output_dir,
    )
