from __future__ import annotations

import sys
from datetime import date

from cvbench.core.augment import apply_augmentation
from cvbench.core.config import build_config, save_config, LossConfig, OptimizerConfig, LRSchedulerConfig
from cvbench.core.runs import make_run_name, make_unique_dir, EXPERIMENTS_DIR
from cvbench.core import trainer as _trainer
from cvbench.datasets.layout import detect_task_name
from cvbench.services._runtime import print_device_banner
from cvbench.tasks import get_task


def run_training(
    data_dir: str,
    output_dir: str | None = None,
    # NOTE: on_epoch_end is reserved for the WebUI progress streaming.
    # When the WebUI calls run_training() it should pass a callable:
    #
    #   on_epoch_end(epoch: int, logs: dict) -> None
    #
    # The trainer will fire it after every epoch. The WebUI implementation
    # should push the logs dict to an SSE event queue so the browser receives
    # live metrics (loss, val_accuracy, etc.) without polling.
    #
    # The CLI leaves this as None — Keras verbose=1 handles stdout output.
    #
    # To wire this up, core/trainer.py needs a small _ProgressEmitter callback:
    #
    #   class _ProgressEmitter(keras.callbacks.Callback):
    #       def __init__(self, fn): self._fn = fn
    #       def on_epoch_end(self, epoch, logs=None):
    #           if self._fn: self._fn(epoch, logs or {})
    #
    # TODO: add on_epoch_end parameter and _ProgressEmitter when implementing
    #       the WebUI training endpoint (tracked in a follow-up GitHub issue).
    from_dir: str | None = None,
    backbone: str | None = None,
    weights: str | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    input_size: int | None = None,
    dropout: float | None = None,
    aug_file: str | None = None,
    resume: str | None = None,
    class_weight=None,
    loss: LossConfig | None = None,
    optimizer: OptimizerConfig | None = None,
    lr_scheduler: LRSchedulerConfig | None = None,
    fine_tune_from_layer: int | None = None,
    val_split: float | None = None,
    seed: int | None = None,
) -> str:
    """Orchestrate a full training run.

    Builds config, datasets, model, and delegates to core trainer.
    Returns the experiment directory path.
    """
    from cvbench.core import _fmt

    print_device_banner("training")

    if seed is not None:
        import keras
        keras.utils.set_random_seed(seed)
        print(_fmt.dim(f" Seed: {seed} (reproducible run)"))

    cfg = build_config(
        data_dir=data_dir,
        from_dir=from_dir,
        backbone=backbone,
        weights=weights,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_size=input_size,
        dropout=dropout,
        class_weight=class_weight,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fine_tune_from_layer=fine_tune_from_layer,
        val_split=val_split,
        seed=seed,
    )

    if aug_file:
        from cvbench.core.config import load_aug_file
        cfg.augmentation = load_aug_file(aug_file)

    # The dataset layout selects the task. A resumed/two-phase run (--from)
    # keeps the task recorded in its baseline config instead of re-sniffing —
    # see core/task.py and the module docstring in cvbench.tasks.
    if from_dir is None:
        cfg.task = detect_task_name(data_dir)
    task = get_task(cfg.task)

    errors = task.validate_config(cfg)
    if errors:
        raise ValueError("Invalid config for task " + repr(cfg.task) + ": " + "; ".join(errors))

    # resolve_layout must run before naming: it's what finalizes
    # cfg.model.backbone (e.g. detection's resnet_18 default when --backbone
    # wasn't passed explicitly), and make_run_name reads that field.
    spec = task.resolve_layout(cfg)
    resolved_weights = task.fit_class_weight(cfg, spec)

    if output_dir is not None:
        exp_dir = output_dir
    else:
        run_name = make_run_name(cfg)
        exp_dir = str(make_unique_dir(EXPERIMENTS_DIR, run_name))

    cfg.run.name = exp_dir.rstrip("/").split("/")[-1]
    cfg.run.date = date.today().strftime("%Y-%m-%d")
    cfg.run.status = "running"
    cfg.run.cli_command = " ".join([sys.argv[0].split("/")[-1]] + sys.argv[1:])

    save_config(cfg, exp_dir)

    train_ds, val_ds, num_train = task.build_datasets(cfg, spec)
    train_ds = apply_augmentation(train_ds, task.filter_transforms(cfg.augmentation.transforms))

    model = task.build_model(cfg)

    _trainer.train(
        cfg=cfg,
        exp_dir=exp_dir,
        train_ds=train_ds,
        val_ds=val_ds,
        class_names=spec.class_names,
        model=model,
        task=task,
        resume_checkpoint=resume,
        num_train_samples=num_train,
        class_weight=resolved_weights,
    )

    return exp_dir
