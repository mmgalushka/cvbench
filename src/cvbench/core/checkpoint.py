from __future__ import annotations

import re
from pathlib import Path

import keras

from cvbench.core import _fmt
from cvbench.core.config import CVBenchConfig


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
                    print(f"\n  {_fmt.green('✓ New best saved')}  {_fmt.dim(monitor)} = {_fmt.bold(val_str)}")

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
                    print(f"\n  {_fmt.green('✓ Checkpoint saved')}  epoch {_fmt.bold(str(epoch + 1))}")

        return _EveryNEpochs(
            filepath=str(Path(run_dir) / "epoch_{epoch:03d}.keras"),
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
            verbose=0,
        )

    raise ValueError(f"Unknown checkpoint strategy: '{strategy}'")


def prune_checkpoints(run_dir: str, cfg: CVBenchConfig):
    """Remove old rolling checkpoints, keeping the most recent keep_last_n.

    'best.keras' is never touched. Only epoch_NNN.keras files are pruned.
    """
    keep = cfg.training.checkpoints.keep_last_n
    pattern = re.compile(r"epoch_(\d+)\.keras$")

    epoch_files = []
    for p in Path(run_dir).iterdir():
        m = pattern.match(p.name)
        if m:
            epoch_files.append((int(m.group(1)), p))

    epoch_files.sort(key=lambda t: t[0])

    to_delete = epoch_files[: max(0, len(epoch_files) - keep)]
    for _, path in to_delete:
        path.unlink(missing_ok=True)
