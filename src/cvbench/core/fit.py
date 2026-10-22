"""The training-loop hook: what ``Task.fit`` receives and returns, and the default Keras loop.

``core/trainer.py`` prepares everything a loop needs (models, datasets, resume state, stop
flag) into a ``FitContext`` and hands it to ``Task.fit``. The default ``KerasFitLoop`` is
exactly the former inline ``model.fit`` call with its callbacks. A task with another shape
of training (a GAN's alternating updates, a VAE with a KL term) overrides ``Task.fit`` and
reports each epoch through ``ctx.end_epoch``, which keeps ``training_log.csv``, the
``EpochEnded`` events and the per-artifact checkpoints working.

This module is TensorFlow-free at import; Keras is imported inside the loop.
"""
from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cvbench.core.events import EpochEnded, Listener, Notice, emit

if TYPE_CHECKING:
    import keras

    from cvbench.core.checkpoint import ArtifactCheckpointer
    from cvbench.core.config import CVBenchConfig
    from cvbench.core.task import Task


@dataclass
class FitResult:
    """What a loop reports back: per-metric history and the number of epochs it ran this call."""

    history: dict[str, list[float]]
    epochs: int


@dataclass
class FitContext:
    """Everything a training loop needs; built by ``core.trainer.train``."""

    cfg: CVBenchConfig
    task: Task
    run_dir: Path
    models: dict[str, keras.Model]  # by artifact name, primary first
    train_ds: Any
    val_ds: Any
    steps_per_epoch: int
    initial_epoch: int
    class_weight: dict | None
    listener: Listener | None
    checkpointer: ArtifactCheckpointer
    stop_requested: Callable[[], bool]  # True once Ctrl-C asked for a graceful stop
    csv_fields: list[str] = field(default_factory=list)

    @property
    def model(self) -> keras.Model:
        """The primary artifact's model."""
        return self.models[self.task.primary_artifact]

    def end_epoch(self, epoch: int, logs: dict[str, float]) -> None:
        """Record epoch EPOCH (zero-based): CSV row, ``EpochEnded`` event, checkpoint decision."""
        logs = {k: float(v) for k, v in logs.items()}
        self._append_csv(epoch, logs)
        emit(self.listener, EpochEnded(epoch, logs))
        self.checkpointer.on_epoch_end(epoch, logs)

    def _append_csv(self, epoch: int, logs: dict[str, float]) -> None:
        path = self.run_dir / "training_log.csv"
        if not self.csv_fields:  # first row of this call: a resumed run appends, a fresh one starts over
            self.csv_fields = ["epoch", *logs]
            if not (self.initial_epoch and path.exists()):
                path.unlink(missing_ok=True)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields, extrasaction="ignore")
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({"epoch": epoch, **logs})


class KerasFitLoop:
    """The default loop: ``model.fit`` on the primary model with the standard callbacks."""

    @staticmethod
    def prepare_resume(model: keras.Model, cfg: CVBenchConfig, listener: Listener | None = None) -> None:
        """Re-apply trainable flags and sync the learning rate on a reloaded checkpoint model."""
        # Re-apply trainable flags — the checkpoint reflects Phase 1's config
        # (backbone frozen); Phase 2 may need a different setting.
        apply_fine_tune_flags(model, cfg.model.fine_tune_from_layer, listener)
        # Sync the optimizer LR to the current config. Phase 2 typically uses
        # a much lower LR; without this the checkpoint's Phase 1 LR would persist.
        model.optimizer.learning_rate.assign(cfg.training.learning_rate)

    def fit(self, ctx: FitContext) -> FitResult:
        import keras

        from cvbench.core import console

        cfg, run_dir = ctx.cfg, ctx.run_dir
        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.CSVLogger(str(run_dir / "training_log.csv"), append=bool(ctx.initial_epoch)),
        ]
        ckpt_cb = ctx.checkpointer.keras_callback()
        if ckpt_cb:
            callbacks.append(ckpt_cb)

        lrs = cfg.training.lr_scheduler
        if lrs.patience > 0:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor=lrs.monitor,
                factor=lrs.factor,
                patience=lrs.patience,
                min_lr=lrs.min_lr,
                verbose=1 if console.is_tty() else 2,
            ))

        class _EpochEmitter(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                emit(ctx.listener, EpochEnded(epoch, {k: float(v) for k, v in (logs or {}).items()}))

        callbacks.append(_EpochEmitter())

        class _GracefulStop(keras.callbacks.Callback):
            def on_batch_end(self_, batch, logs=None):
                if ctx.stop_requested():
                    self_.model.stop_training = True

        callbacks.append(_GracefulStop())

        history = ctx.model.fit(
            ctx.train_ds,
            epochs=cfg.training.epochs,
            initial_epoch=ctx.initial_epoch,
            steps_per_epoch=ctx.steps_per_epoch,
            validation_data=ctx.val_ds,
            callbacks=callbacks,
            class_weight=ctx.class_weight,
            verbose=1 if console.is_tty() else 2,
        )
        return FitResult(history=history.history, epochs=len(history.epoch))


def apply_fine_tune_flags(model: keras.Model, fine_tune_from_layer: int, listener: Listener | None = None) -> None:
    """Re-apply backbone trainable flags to a loaded checkpoint model.

    Needed when resuming with a different fine_tune_from_layer than Phase 1
    (e.g. Phase 2 unfreezes the backbone while the checkpoint still has it frozen).
    """
    import keras

    from cvbench.core.backbone import apply_freeze

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
    apply_freeze(backbone, fine_tune_from_layer)
