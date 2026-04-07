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


@click.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--output-dir", default=None, help="Where to write eval outputs (default: run_dir).")
def evaluate(run_dir, output_dir):
    """Evaluate a trained model in RUN_DIR on the held-out test split."""
    cfg = load_config(run_dir)

    class_names = get_class_names(cfg.data.train_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        test_ds = build_dataset(cfg.data.test_dir, class_names, cfg, training=False)

    n_test = sum(1 for _ in Path(cfg.data.test_dir).glob("*/*"))
    print(f" Found {n_test} files for evaluation ({len(class_names)} classes).")

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

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
