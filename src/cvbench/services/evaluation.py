from __future__ import annotations

import contextlib
import io
import warnings
from pathlib import Path

import keras

from cvbench.core.config import load_config, save_config
from cvbench.core.data import build_dataset, get_class_names
from cvbench.core.runs import resolve_run_dir
from cvbench.core import evaluator as _evaluator
from cvbench.core import _fmt


def run_evaluation(
    experiment: str,
    output_dir: str | None = None,
    # NOTE: on_batch_end is reserved for the WebUI progress streaming.
    # When the WebUI calls run_evaluation() it should pass a callable:
    #
    #   on_batch_end(batch: int, total: int) -> None
    #
    # The evaluator will fire it after every batch so the browser can show
    # a live progress bar via SSE. The CLI leaves this as None — tqdm
    # handles the terminal progress bar.
    #
    # TODO: add on_batch_end parameter to run_evaluation and thread it through
    #       to core/evaluator.py when implementing the WebUI evaluation endpoint
    #       (tracked in a follow-up GitHub issue).
) -> dict:
    """Load a trained model and evaluate it on the test split.

    Returns the evaluation report dict (same structure written to eval_report.json).
    """
    import platform
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    run_dir = resolve_run_dir(experiment)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print(_fmt.green(f"🟢 Apple Silicon GPU (Metal) detected — evaluating on {len(gpus)} device(s)"))
        else:
            names = ", ".join(g.name for g in gpus)
            print(_fmt.green(f"🟢 GPU detected: {len(gpus)} device(s) — {names}"))
    else:
        print(_fmt.yellow("⚠️  GPU not available, evaluating on CPU"))

    cfg = load_config(run_dir)

    class_names = get_class_names(cfg.data.train_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        test_ds = build_dataset(cfg.data.test_dir, class_names, cfg, training=False)

    n_test = sum(1 for _ in Path(cfg.data.test_dir).glob("*/*"))
    print(_fmt.dim(f" Found {n_test} files for evaluation ({len(class_names)} classes)."))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model = keras.saving.load_model(f"{run_dir}/best.keras")

    report = _evaluator.evaluate(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        run_dir=run_dir,
        test_dir=cfg.data.test_dir,
        output_dir=output_dir,
    )

    cfg.run.test_accuracy = report["overall_accuracy"]
    save_config(cfg, run_dir)

    return report
