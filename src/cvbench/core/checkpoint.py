from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import keras

from cvbench.core import console
from cvbench.core.config import CVBenchConfig
from cvbench.core.errors import ConfigError

if TYPE_CHECKING:
    from cvbench.core.task import Task


def _append_best_event(run_dir: str, epoch: int, monitor: str, value: float | None) -> None:
    record = {"epoch": epoch + 1, "metric": monitor, "value": round(float(value), 6) if value is not None else None}
    with open(Path(run_dir) / "best_history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def load_best_history(run_dir: str) -> list[dict]:
    path = Path(run_dir) / "best_history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_checkpoint_callback(cfg: CVBenchConfig, run_dir: str) -> keras.callbacks.ModelCheckpoint | None:
    """Return a ModelCheckpoint callback configured per strategy, or None."""
    ckpt_cfg = cfg.training.checkpoints
    strategy = ckpt_cfg.strategy

    if strategy == "best_only":
        monitor = ckpt_cfg.monitor

        class _BestOnly(keras.callbacks.ModelCheckpoint):
            def on_epoch_end(self, epoch, logs=None):
                prev_best = self.best
                super().on_epoch_end(epoch, logs)
                if self.best != prev_best:
                    val = logs.get(monitor) if logs else None
                    val_str = f"{val:.4f}" if val is not None else "—"
                    _append_best_event(run_dir, epoch, monitor, val)
                    print()
                    console.success(f"New best saved  {console.dim(monitor)} = {console.bold(val_str)}")

        return _BestOnly(
            filepath=str(Path(run_dir) / "best.keras"),
            monitor=monitor,
            mode=ckpt_cfg.mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        )

    if strategy in ("every_epoch", "every_n_epochs"):
        # Keras 3 removed the `period` argument. For every_n_epochs we subclass
        # and skip saves on off-epochs; pruning then keeps only the last N.
        every_n = ckpt_cfg.every_n_epochs if strategy == "every_n_epochs" else 1

        class _EveryNEpochs(keras.callbacks.ModelCheckpoint):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % every_n == 0:
                    super().on_epoch_end(epoch, logs)
                    print()
                    console.success(f"Checkpoint saved  epoch {console.bold(str(epoch + 1))}")

        return _EveryNEpochs(
            filepath=str(Path(run_dir) / "epoch_{epoch:03d}.keras"),
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
            verbose=0,
        )

    raise ConfigError(f"Unknown checkpoint strategy: '{strategy}'")


def save_models(task: Task, models: dict[str, keras.Model], checkpoint: str | Path) -> None:
    """Save every artifact in MODELS next to CHECKPOINT (the primary artifact's path)."""
    for name, model in models.items():
        model.save(str(task.artifact_path(checkpoint, name)))


class ArtifactCheckpointer:
    """Checkpointing for a run's artifacts, driven by ``training.checkpoints``.

    ``keras_callback()`` is the Keras-fit path (primary model only, the former behaviour);
    ``on_epoch_end`` applies the same strategy for a custom fit loop and saves *every*
    artifact, so ``best.keras`` plus ``best.<name>.keras`` stay in step.
    """

    def __init__(self, cfg: CVBenchConfig, run_dir: str | Path, task: Task, models: dict[str, keras.Model]):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.task = task
        self.models = models
        self.best: float | None = None

    def keras_callback(self) -> keras.callbacks.ModelCheckpoint | None:
        return build_checkpoint_callback(self.cfg, str(self.run_dir))

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        ckpt = self.cfg.training.checkpoints
        if ckpt.strategy == "best_only":
            value = logs.get(ckpt.monitor)
            if value is None:
                return
            improved = (
                self.best is None
                or (value < self.best if ckpt.mode == "min" else value > self.best)
            )
            if improved:
                self.best = value
                save_models(self.task, self.models, self.run_dir / "best.keras")
                _append_best_event(str(self.run_dir), epoch, ckpt.monitor, value)
                print()
                console.success(f"New best saved  {console.dim(ckpt.monitor)} = {console.bold(f'{value:.4f}')}")
        elif ckpt.strategy in ("every_epoch", "every_n_epochs"):
            every_n = ckpt.every_n_epochs if ckpt.strategy == "every_n_epochs" else 1
            if (epoch + 1) % every_n == 0:
                save_models(self.task, self.models, self.run_dir / f"epoch_{epoch + 1:03d}.keras")
                print()
                console.success(f"Checkpoint saved  epoch {console.bold(str(epoch + 1))}")
        else:
            raise ConfigError(f"Unknown checkpoint strategy: '{ckpt.strategy}'")


def prune_checkpoints(run_dir: str, cfg: CVBenchConfig):
    """Remove old rolling checkpoints, keeping the most recent keep_last_n epochs.

    'best.keras' is never touched. Only epoch_NNN.keras files (and the sibling
    epoch_NNN.<artifact>.keras files of a multi-artifact run) are pruned.
    """
    keep = cfg.training.checkpoints.keep_last_n
    pattern = re.compile(r"epoch_(\d+)(?:\.\w+)?\.keras$")

    by_epoch: dict[int, list[Path]] = {}
    for p in Path(run_dir).iterdir():
        m = pattern.match(p.name)
        if m:
            by_epoch.setdefault(int(m.group(1)), []).append(p)

    epochs = sorted(by_epoch)
    for epoch in epochs[: max(0, len(epochs) - keep)]:
        for path in by_epoch[epoch]:
            path.unlink(missing_ok=True)
