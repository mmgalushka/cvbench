from __future__ import annotations

import signal
from pathlib import Path
from typing import TYPE_CHECKING

import keras

from cvbench.core.checkpoint import ArtifactCheckpointer, load_best_history, prune_checkpoints, save_models
from cvbench.core.config import CVBenchConfig, update_run_status
from cvbench.core.events import (
    InterruptRequested,
    Listener,
    RunFinished,
    RunStarted,
    emit,
)
from cvbench.core.fit import FitContext
from cvbench.core.fit import apply_fine_tune_flags as _apply_fine_tune_flags  # noqa: F401  (kept importable here)

if TYPE_CHECKING:
    from cvbench.core.task import Task


def train(
    cfg: CVBenchConfig,
    exp_dir: str,
    train_ds,
    val_ds,
    class_names: list[str],
    model: keras.Model | None,
    task: Task,
    resume_checkpoint: str | None = None,
    num_train_samples: int | None = None,
    class_weight: dict | None = None,
    listener: Listener | None = None,
) -> str:
    """Run the training loop. Returns the experiment directory path.

    MODEL is the primary artifact's model; ``Task.build_models`` adds the other artifacts of
    a multi-artifact task (``Task.artifacts()``). The loop itself is ``Task.fit``.

    Prints nothing: progress goes to LISTENER as ``RunStarted``, ``EpochEnded`` and
    ``RunFinished`` events (see ``cvbench.core.events``).
    """

    run_dir = Path(exp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    val_images = task.count_images(cfg.data.val_dir) if cfg.data.val_dir else None
    emit(listener, RunStarted(exp_dir, cfg, val_images, resume_checkpoint))

    models = task.build_models(cfg, primary=model)

    # Resume from checkpoint if requested
    initial_epoch = 0
    if resume_checkpoint:
        # Load the full models (weights + Adam moment accumulators) so the head's
        # optimizer state carries over smoothly. model.load_weights() would restore
        # weight values but reset all moments to zero, causing an amplified effective
        # LR at the start of the resumed run (identical to the zero-moment problem
        # we're trying to avoid in the backbone during two-phase training).
        models = task.load_models(resume_checkpoint)
        task.prepare_resume(models, cfg, listener)

        if cfg.run.epochs_run > 0:
            # Use the epoch count stored in config (reliable for all checkpoint names,
            # including best.keras which carries no epoch number in its filename).
            initial_epoch = cfg.run.epochs_run
        else:
            import re
            m = re.search(r"epoch[_]?(\d+)", Path(resume_checkpoint).stem)
            if m:
                initial_epoch = int(m.group(1))

    # Graceful interrupt
    _stop_flag = {"value": False}

    def _sigint_handler(signum, frame):
        emit(listener, InterruptRequested())
        _stop_flag["value"] = True

    if cfg.training.interrupt.enabled:
        signal.signal(signal.SIGINT, _sigint_handler)

    # Steps per epoch (needed because train_ds uses repeat())
    if num_train_samples is None:
        raise ValueError("train() requires num_train_samples (train_ds uses repeat())")
    import math
    steps_per_epoch = math.ceil(num_train_samples / cfg.data.batch_size)

    ctx = FitContext(
        cfg=cfg,
        task=task,
        run_dir=run_dir,
        models=models,
        train_ds=train_ds,
        val_ds=val_ds,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        class_weight=class_weight,
        listener=listener,
        checkpointer=ArtifactCheckpointer(cfg, run_dir, task, models),
        stop_requested=lambda: _stop_flag["value"],
    )
    result = task.fit(ctx)

    interrupted = _stop_flag["value"]
    last_epoch = initial_epoch + result.epochs

    # Save interrupt checkpoint
    interrupt_ckpt = None
    if interrupted and cfg.training.interrupt.save_checkpoint:
        ckpt_name = f"interrupt_epoch{last_epoch:03d}.keras"
        interrupt_ckpt = str(run_dir / ckpt_name)
        save_models(task, models, interrupt_ckpt)

    prune_checkpoints(str(run_dir), cfg)

    # Collect final metrics from history
    final_metrics = {}
    for k, v in result.history.items():
        if v:
            final_metrics[k] = round(float(v[-1]), 4)

    update_run_status(
        exp_dir,
        status="interrupted" if interrupted else "done",
        epochs_run=last_epoch,
        resumable=interrupted,
        resume_checkpoint=interrupt_ckpt,
        metrics={k: v for k, v in task.headline_metrics(final_metrics).items() if v is not None},
    )

    status = "interrupted" if interrupted else "done"
    emit(listener, RunFinished(
        exp_dir=str(exp_dir),
        run_name=cfg.run.name,
        status=status,
        final_metrics=final_metrics,
        best_events=load_best_history(str(run_dir)),
        advisories=[] if interrupted else task.training_advisories(cfg, final_metrics),
        interrupt_checkpoint=interrupt_ckpt,
        data_dir=cfg.data.data_dir,
    ))

    return str(run_dir)
