"""Sweep orchestration: train every planned trial in turn and rank the results.

The caller (the ``sweep`` command, later the WebUI) parses and validates the plan;
this module runs it. It prints nothing: trial-level progress goes to ``listener`` as
``cvbench.core.events`` events, and each trial's own training events are forwarded.
"""
from __future__ import annotations

import dataclasses
import gc
import tempfile
from pathlib import Path
from typing import Any

from cvbench.core.config import load_config, update_run_status
from cvbench.core.errors import NotFoundError
from cvbench.core.events import (
    Listener,
    Notice,
    SweepFinished,
    SweepInterrupted,
    TrialFailed,
    TrialFinished,
    TrialStarted,
    emit,
)
from cvbench.core.exp_store import EXPERIMENTS_DIR
from cvbench.core.sweep_store import SweepManifest, trial_dir_name, write_manifest
from cvbench.services import training as _training
from cvbench.tasks import get_task, resolve_task


def evaluate_on_val(trial_dir: str, listener: Listener | None = None) -> None:
    """Score a trained trial on its VAL split and record the headline score in config.yaml.

    `run_evaluation` only ever scores the test split, and picking a hyper-parameter
    by test score would leak the test set into model selection. So this reuses the
    same task API on a spec whose `test_dir` points at the val split. The report is
    written to a throw-away directory (not the trial's eval_report.json, which is
    reserved for the real test evaluation) and only the headline score is stored,
    in `run.metrics` under the task's primary metric name (e.g. `map50`) — where
    `sweep_store.summarize` reads it for that selection metric. Running `evaluate <trial>`
    later overwrites them with the real test score.
    """
    cfg = load_config(trial_dir)
    task = resolve_task(cfg)
    spec = task.resolve_layout(cfg)
    if not spec.val_dir:
        emit(listener, Notice("No val split for this dataset; mAP@50 unavailable."))
        return
    best_path = Path(trial_dir) / "best.keras"
    if not best_path.is_file():
        raise NotFoundError(
            f"no checkpoint was saved ({best_path} is missing) — the monitored metric "
            "likely never improved (check training_log.csv for a NaN/diverging loss)"
        )
    val_spec = dataclasses.replace(spec, test_dir=spec.val_dir)
    val_ds = task.build_eval_dataset(cfg, val_spec)
    model = task.load_model(str(best_path))
    with tempfile.TemporaryDirectory() as tmp:
        report = task.evaluate(
            model=model, eval_ds=val_ds, cfg=cfg, spec=val_spec, run_dir=trial_dir, output_dir=tmp,
            listener=listener,
        )
    metric, value = task.test_score(report)
    update_run_status(trial_dir, metrics={metric: value}, primary_metric=metric)


def _release_memory() -> None:
    """Drop the finished trial's graph and weights so memory does not grow across a long sweep."""
    import keras

    keras.backend.clear_session()
    gc.collect()


def run_sweep(
    manifest: SweepManifest,
    trials: list[dict[str, str]],
    trial_values: list[dict[str, Any]],
    *,
    task_name: str,
    fixed: dict[str, Any] | None = None,
    from_dir: str | None = None,
    listener: Listener | None = None,
) -> Path:
    """Create the sweep directory and train one trial per entry of TRIALS.

    TRIALS holds each trial's axis values as text (for display); TRIAL_VALUES the same
    trials as the dotted overrides `run_training` applies, FIXED being applied to all.
    A failing trial is reported and the sweep carries on; Ctrl-C stops it. Always ends
    with `SweepFinished` carrying the sweep directory. Returns that directory.
    """
    sweep_dir = Path(EXPERIMENTS_DIR) / manifest.name
    sweep_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(sweep_dir, manifest)
    fixed = fixed or {}
    # The task's primary metric only exists after an evaluation, so a sweep ranking on
    # it scores each trial on its val split once training finishes.
    post_hoc_metric = get_task(task_name).primary_metric.name

    for i, (values, plan) in enumerate(zip(trial_values, trials, strict=True), start=1):
        trial_name = trial_dir_name(manifest.name, i)
        trial_dir = str(sweep_dir / trial_name)
        emit(listener, TrialStarted(i, len(trials), trial_name, plan))
        try:
            # training.interrupt.enabled=False: a standalone `train` catches Ctrl-C to finish the
            # current batch and checkpoint gracefully, but under a sweep that would only
            # stop the current trial and silently move on to the next one. Disabling it
            # lets Ctrl-C raise KeyboardInterrupt immediately, caught below to stop the
            # whole sweep right away.
            _training.run_training(
                data_dir=manifest.data_dir, output_dir=trial_dir, from_dir=from_dir,
                overrides={**fixed, **values, "training.interrupt.enabled": False},
                listener=listener,
            )
        except KeyboardInterrupt:
            emit(listener, SweepInterrupted(trial_name, "training"))
            break
        except Exception as e:  # noqa: BLE001 - one bad trial must not stop the sweep
            emit(listener, TrialFailed(i, trial_name, "training", type(e).__name__, str(e)))
            _release_memory()
            continue
        # Evaluation is a separate step from training: a trial that trained fine but has
        # no best.keras (e.g. the monitored metric never improved) must be reported as a
        # val-evaluation failure, not misattributed as "training failed".
        if manifest.metric == post_hoc_metric:
            try:
                evaluate_on_val(trial_dir, listener)
            except KeyboardInterrupt:
                emit(listener, SweepInterrupted(trial_name, "val evaluation"))
                break
            except Exception as e:  # noqa: BLE001 - one bad trial must not stop the sweep
                emit(listener, TrialFailed(i, trial_name, "val evaluation", type(e).__name__, str(e)))
                _release_memory()
                continue
        emit(listener, TrialFinished(i, trial_name))
        _release_memory()

    emit(listener, SweepFinished(str(sweep_dir)))
    return sweep_dir

