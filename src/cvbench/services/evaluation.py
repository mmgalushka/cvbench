from __future__ import annotations

from cvbench.core.config import load_config, save_config
from cvbench.core.runs import resolve_run_dir
from cvbench.core import _console
from cvbench.services._runtime import print_device_banner
from cvbench.tasks import resolve_task


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
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    run_dir = resolve_run_dir(experiment)

    print_device_banner("evaluating")

    cfg = load_config(run_dir)
    task = resolve_task(cfg)

    spec = task.resolve_layout(cfg)
    test_ds = task.build_eval_dataset(cfg, spec)

    n_test = task.count_images(cfg.data.test_dir)
    print(_console.dim(f" Found {n_test} files for evaluation ({len(spec.class_names)} classes)."))

    model = task.load_model(f"{run_dir}/best.keras")

    report = task.evaluate(
        model=model,
        eval_ds=test_ds,
        cfg=cfg,
        spec=spec,
        run_dir=run_dir,
        output_dir=output_dir,
    )

    metric, value = task.test_score(report)
    cfg.run.test_accuracy = value
    cfg.run.test_metric = metric
    save_config(cfg, run_dir)

    return report
