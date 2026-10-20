from __future__ import annotations

from pathlib import Path

from cvbench.core.config import load_config, save_config
from cvbench.core.events import EvalStarted, Listener, Notice, SweepEvalTrialStarted, emit
from cvbench.core.exp_store import resolve_run_dir
from cvbench.core.sweep_store import selection_metric_of_trial, summarize
from cvbench.services._runtime import report_devices
from cvbench.tasks import option_scope_message, resolve_task


def run_evaluation(
    experiment: str,
    output_dir: str | None = None,
    conf: float | None = None,
    listener: Listener | None = None,
) -> dict:
    """Load a trained model and evaluate it on the test split.

    Returns the evaluation report dict (same structure written to eval_report.json).
    Progress goes to `listener` (device, file counts, per-batch `EvalProgress`).
    """
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    run_dir = resolve_run_dir(experiment)

    report_devices("evaluating", listener)

    cfg = load_config(run_dir)
    task = resolve_task(cfg)
    if conf is not None and "conf" not in task.predict_options():
        emit(listener, Notice(f"{option_scope_message('conf')}; ignoring it."))

    spec = task.resolve_layout(cfg)
    test_ds = task.build_eval_dataset(cfg, spec)

    n_test = task.count_images(cfg.data.test_dir)
    emit(listener, EvalStarted(n_test, len(spec.class_names)))

    model = task.load_model(f"{run_dir}/best.keras")

    report = task.evaluate(
        model=model,
        eval_ds=test_ds,
        cfg=cfg,
        spec=spec,
        run_dir=run_dir,
        output_dir=output_dir,
        conf=conf,
        listener=listener,
    )

    metric, value = task.test_score(report)
    # A sweep that ranks on this metric keeps the trial's val score under its name in the
    # run metrics, so a test score must not overwrite it (the test score stays in eval_report.json).
    if selection_metric_of_trial(run_dir) != metric:
        cfg.run.metrics[metric] = value
        cfg.run.primary_metric = metric
        save_config(cfg, run_dir)

    return report


def run_sweep_evaluation(
    sweep: str, listener: Listener | None = None,
) -> list[tuple[str, str | None, float | None]]:
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
            emit(listener, Notice(f"Skipping {r.dir}: trial status is '{r.status}'."))
            results.append((r.dir, None, None))
    for i, r in enumerate(todo, start=1):
        emit(listener, SweepEvalTrialStarted(i, len(todo), r.dir))
        try:
            report = run_evaluation(str(sweep_dir / r.dir), listener=listener)
            cfg = load_config(str(sweep_dir / r.dir))
            metric, value = resolve_task(cfg).test_score(report)
        except Exception as e:  # noqa: BLE001 - one bad trial must not stop the rest
            emit(listener, Notice(f"Evaluation of {r.dir} failed: {type(e).__name__}: {e}", level="error"))
            results.append((r.dir, None, None))
            continue
        results.append((r.dir, metric, value))
    return sorted(results, key=lambda t: t[0])
