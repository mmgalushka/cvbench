from __future__ import annotations

from pathlib import Path

from cvbench.core import _console
from cvbench.core.config import load_config, save_config
from cvbench.core.exp_store import resolve_run_dir
from cvbench.core.sweep_store import selection_metric_of_trial, summarize
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
    # A map50 sweep ranks trials on the val mAP@50 kept in these run fields, so a test
    # score must not overwrite it (the test score stays in eval_report.json).
    if selection_metric_of_trial(run_dir) != "map50":
        cfg.run.test_accuracy = value
        cfg.run.test_metric = metric
        save_config(cfg, run_dir)

    return report


def run_sweep_evaluation(sweep: str) -> list[tuple[str, str | None, float | None]]:
    """Evaluate every finished trial of a sweep on the test split.

    Purely informational: the sweep's best trial is chosen on validation and is not
    affected. Returns one (trial_name, metric, value) row per trial; a trial that did
    not finish or whose evaluation fails has metric/value None.
    """
    sweep_dir = Path(resolve_run_dir(sweep, allow_sweep=True))
    _, rows = summarize(sweep_dir)
    results: list[tuple[str, str | None, float | None]] = []
    todo = [r for r in rows if r.status == "done"]
    for r in rows:
        if r.status != "done":
            _console.warning(f"Skipping {r.dir}: trial status is '{r.status}'.")
            results.append((r.dir, None, None))
    for i, r in enumerate(todo, start=1):
        print()
        print(f" {_console.bold(f'[{i}/{len(todo)}] {r.dir}')}")
        try:
            report = run_evaluation(str(sweep_dir / r.dir))
            cfg = load_config(str(sweep_dir / r.dir))
            metric, value = resolve_task(cfg).test_score(report)
        except Exception as e:  # noqa: BLE001 - one bad trial must not stop the rest
            _console.error(f"Evaluation of {r.dir} failed: {type(e).__name__}: {e}")
            results.append((r.dir, None, None))
            continue
        results.append((r.dir, metric, value))
    return sorted(results, key=lambda t: t[0])
