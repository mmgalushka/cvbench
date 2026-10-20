from __future__ import annotations

import signal
from pathlib import Path
from typing import TYPE_CHECKING

import keras

from cvbench.core import _console
from cvbench.core.checkpoint import build_checkpoint_callback, load_best_history, prune_checkpoints
from cvbench.core.config import CVBenchConfig, update_run_status
from cvbench.core.events import (
    EpochEnded,
    InterruptRequested,
    Listener,
    Notice,
    RunFinished,
    RunStarted,
    emit,
)

if TYPE_CHECKING:
    from cvbench.core.task import Task


def _apply_fine_tune_flags(model: keras.Model, fine_tune_from_layer: int, listener: Listener | None = None) -> None:
    """Re-apply backbone trainable flags to a loaded checkpoint model.

    Needed when resuming with a different fine_tune_from_layer than Phase 1
    (e.g. Phase 2 unfreezes the backbone while the checkpoint still has it frozen).
    """
    try:
        backbone = model.get_layer("backbone")
    except ValueError:
        # Checkpoint saved before backbone layers were named "backbone" —
        # fall back to the old isinstance scan.
        backbone = next(
            (lyr for lyr in model.layers if isinstance(lyr, keras.Model) and lyr is not model),
            None,
        )
    if backbone is None:
        emit(listener, Notice(
            "Could not locate the backbone layer to apply fine_tune_from_layer;"
            " leaving the reloaded checkpoint's trainable flags unchanged."
        ))
        return
    if fine_tune_from_layer == 0:
        backbone.trainable = False
    elif fine_tune_from_layer == -1:
        backbone.trainable = True
    else:
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_from_layer]:
            layer.trainable = False


def train(
    cfg: CVBenchConfig,
    exp_dir: str,
    train_ds,
    val_ds,
    class_names: list[str],
    model: keras.Model,
    task: Task,
    resume_checkpoint: str | None = None,
    num_train_samples: int | None = None,
    class_weight: dict | None = None,
    listener: Listener | None = None,
) -> str:
    """Run the training loop. Returns the experiment directory path.

    Prints nothing: progress goes to LISTENER as ``RunStarted``, ``EpochEnded`` and
    ``RunFinished`` events (see ``cvbench.core.events``).
    """

    run_dir = Path(exp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    val_images = task.count_images(cfg.data.val_dir) if cfg.data.val_dir else None
    emit(listener, RunStarted(exp_dir, cfg, val_images, resume_checkpoint))

    # Resume from checkpoint if requested
    initial_epoch = 0
    if resume_checkpoint:
        # Load the full model (weights + Adam moment accumulators) so the head's
        # optimizer state carries over smoothly. model.load_weights() would restore
        # weight values but reset all moments to zero, causing an amplified effective
        # LR at the start of the resumed run (identical to the zero-moment problem
        # we're trying to avoid in the backbone during two-phase training).
        model = task.load_model(resume_checkpoint)
        # Re-apply trainable flags — the checkpoint reflects Phase 1's config
        # (backbone frozen); Phase 2 may need a different setting.
        _apply_fine_tune_flags(model, cfg.model.fine_tune_from_layer, listener)
        # Sync the optimizer LR to the current config. Phase 2 typically uses
        # a much lower LR; without this the checkpoint's Phase 1 LR would persist.
        model.optimizer.learning_rate.assign(cfg.training.learning_rate)

        if cfg.run.epochs_run > 0:
            # Use the epoch count stored in config (reliable for all checkpoint names,
            # including best.keras which carries no epoch number in its filename).
            initial_epoch = cfg.run.epochs_run
        else:
            import re
            m = re.search(r"epoch[_]?(\d+)", Path(resume_checkpoint).stem)
            if m:
                initial_epoch = int(m.group(1))

    # Callbacks
    callbacks = [
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv"), append=bool(initial_epoch)),
    ]
    ckpt_cb = build_checkpoint_callback(cfg, str(run_dir))
    if ckpt_cb:
        callbacks.append(ckpt_cb)

    lrs = cfg.training.lr_scheduler
    if lrs.patience > 0:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=lrs.monitor,
            factor=lrs.factor,
            patience=lrs.patience,
            min_lr=lrs.min_lr,
            verbose=1 if _console.is_tty() else 2,
        ))

    class _EpochEmitter(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            emit(listener, EpochEnded(epoch, {k: float(v) for k, v in (logs or {}).items()}))

    callbacks.append(_EpochEmitter())

    # Graceful interrupt
    _stop_flag = {"value": False}

    def _sigint_handler(signum, frame):
        emit(listener, InterruptRequested())
        _stop_flag["value"] = True

    if cfg.training.interrupt.enabled:
        signal.signal(signal.SIGINT, _sigint_handler)

    class _GracefulStop(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if _stop_flag["value"]:
                self.model.stop_training = True

    callbacks.append(_GracefulStop())

    # Steps per epoch (needed because train_ds uses repeat())
    if num_train_samples is None:
        raise ValueError("train() requires num_train_samples (train_ds uses repeat())")
    import math
    steps_per_epoch = math.ceil(num_train_samples / cfg.data.batch_size)

    history = model.fit(
        train_ds,
        epochs=cfg.training.epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1 if _console.is_tty() else 2,
    )

    interrupted = _stop_flag["value"]
    last_epoch = initial_epoch + len(history.epoch)

    # Save interrupt checkpoint
    interrupt_ckpt = None
    if interrupted and cfg.training.interrupt.save_checkpoint:
        ckpt_name = f"interrupt_epoch{last_epoch:03d}.keras"
        interrupt_ckpt = str(run_dir / ckpt_name)
        model.save(interrupt_ckpt)

    prune_checkpoints(str(run_dir), cfg)

    # Collect final metrics from history
    final_metrics = {}
    for k, v in history.history.items():
        if v:
            final_metrics[k] = round(float(v[-1]), 4)

    update_run_status(
        exp_dir,
        status="interrupted" if interrupted else "done",
        epochs_run=last_epoch,
        resumable=interrupted,
        resume_checkpoint=interrupt_ckpt,
        **task.headline_metrics(final_metrics),
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
