from __future__ import annotations

import signal
from pathlib import Path

import keras

from cvbench.core.checkpoint import build_checkpoint_callback, prune_checkpoints
from cvbench.core.config import CVBenchConfig, update_run_status
from cvbench.core import _fmt


def _print_header(exp_dir: str, cfg: CVBenchConfig):
    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — train')}")
    print(_fmt.rule())
    print(f" Data      : {_fmt.dim(cfg.data.data_dir)}")
    print(f" Backbone  : {cfg.model.backbone}")
    print(f" Epochs    : {cfg.training.epochs}")
    opt = cfg.training.optimizer
    opt_label = opt.type.upper()
    if opt.weight_decay > 0:
        opt_label += f"  weight_decay={opt.weight_decay}"
    if opt.type == "sgd":
        opt_label += f"  momentum={opt.momentum}"
    print(f" Optimizer : {opt_label}")
    lrs = cfg.training.lr_scheduler
    if lrs.patience > 0:
        print(f" LR        : {cfg.training.learning_rate}  →  reduce by {lrs.factor}x after {lrs.patience} flat epoch(s) (min {lrs.min_lr})")
    else:
        print(f" LR        : {cfg.training.learning_rate}")
    n_transforms = len(cfg.augmentation.transforms)
    print(f" Aug       : {n_transforms} transform(s)")
    print(f" Output    : {_fmt.dim(exp_dir)}")
    print(_fmt.rule())
    print()  # breathing room before epoch output


def train(
    cfg: CVBenchConfig,
    exp_dir: str,
    train_ds,
    val_ds,
    class_names: list[str],
    model: keras.Model,
    resume_checkpoint: str | None = None,
    num_train_samples: int | None = None,
    class_weight: dict | None = None,
) -> str:
    """Run the training loop. Returns the experiment directory path."""

    run_dir = Path(exp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _print_header(exp_dir, cfg)

    # Resume from checkpoint if requested
    initial_epoch = 0
    if resume_checkpoint:
        print(f" Resuming from: {resume_checkpoint}")
        model.load_weights(resume_checkpoint)
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
        keras.callbacks.TensorBoard(log_dir=str(run_dir / "logs"), update_freq="epoch"),
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
            verbose=1,
        ))

    # Graceful interrupt
    _stop_flag = {"value": False}

    def _sigint_handler(signum, frame):
        print(f"\r\033[K\n {_fmt.yellow('⚠️  Interrupt received')} — finishing current batch then saving...\n")
        _stop_flag["value"] = True

    if cfg.training.interrupt.enabled:
        signal.signal(signal.SIGINT, _sigint_handler)

    class _GracefulStop(keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if _stop_flag["value"]:
                self.model.stop_training = True

    callbacks.append(_GracefulStop())

    # Steps per epoch (needed because train_ds uses repeat())
    import math
    if num_train_samples is None:
        import tensorflow as tf
        num_train_samples = sum(1 for _ in tf.data.Dataset.list_files(
            str(Path(cfg.data.train_dir) / "*" / "*"), shuffle=False
        ))
    steps_per_epoch = math.ceil(num_train_samples / cfg.data.batch_size)

    history = model.fit(
        train_ds,
        epochs=cfg.training.epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    interrupted = _stop_flag["value"]
    last_epoch = initial_epoch + len(history.epoch)

    # Save interrupt checkpoint
    interrupt_ckpt = None
    if interrupted and cfg.training.interrupt.save_checkpoint:
        ckpt_name = f"interrupt_epoch{last_epoch:03d}.keras"
        interrupt_ckpt = str(run_dir / ckpt_name)
        model.save(interrupt_ckpt)
        print(f"\n {_fmt.green('Checkpoint saved')} → {_fmt.dim(interrupt_ckpt)}")
        print(
            f" To resume: train {cfg.data.data_dir}"
            f" --from {exp_dir}"
            f" --resume {interrupt_ckpt}\n"
        )

    prune_checkpoints(str(run_dir), cfg)

    # Collect final metrics from history
    final_metrics = {}
    for k, v in history.history.items():
        if v:
            final_metrics[k] = round(float(v[-1]), 4)

    stopped_by = "interrupted" if interrupted else f"completed ({last_epoch} epochs)"
    if not interrupted and history.history:
        if model.stop_training and last_epoch < cfg.training.epochs:
            stopped_by = "early stopping (patience)"

    update_run_status(
        exp_dir,
        status="interrupted" if interrupted else "done",
        epochs_run=last_epoch,
        val_accuracy=final_metrics.get("val_accuracy"),
        val_loss=final_metrics.get("val_loss"),
        resumable=interrupted,
        resume_checkpoint=interrupt_ckpt,
    )

    status_label = "interrupted" if interrupted else "complete"
    status_color = _fmt.yellow if interrupted else _fmt.green
    print()  # breathing room after last epoch line
    print(_fmt.rule())
    print(f" {_fmt.bold(f'Training {status_color(status_label)}')}")
    if final_metrics:
        max_k = max(len(k) for k in final_metrics)
        for k, v in final_metrics.items():
            print(f"   {_fmt.bold(f'{k:<{max_k}}')} : {v:.4f}")
    print(f" Run directory → {_fmt.dim(str(run_dir))}")
    print(_fmt.rule())

    return str(run_dir)
