from __future__ import annotations

import os
import shutil
import sys
from datetime import date
from typing import Any

from cvbench.core import trainer as _trainer
from cvbench.core.config import build_config, save_config, update_run_status
from cvbench.core.errors import ConfigError, NotFoundError
from cvbench.core.events import Listener, SeedSet, emit
from cvbench.core.exp_store import EXPERIMENTS_DIR, check_output_dir, make_run_name, make_unique_dir
from cvbench.services import report_devices
from cvbench.tasks import get_task


def run_training(
    data_dir: str,
    output_dir: str | None = None,
    from_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
    resume: str | None = None,
    overwrite: bool = False,
    listener: Listener | None = None,
) -> str:
    """Orchestrate a full training run.

    Builds config, datasets, model, and delegates to core trainer.
    Returns the experiment directory path.

    `overrides` maps dotted config keys to values (see `core.config.apply_overrides`)
    and is applied on top of the `from_dir` baseline or the defaults. Callers that
    chain several runs in one process (e.g. `sweep`) pass `training.interrupt.enabled:
    False` (graceful Ctrl-C: finish the current batch, checkpoint, and stop) so Ctrl-C
    raises KeyboardInterrupt immediately instead of only stopping the current run.

    Prints nothing: progress is reported to `listener` as `cvbench.core.events` events.
    """
    if output_dir is not None:
        check_output_dir(output_dir, resume=bool(resume), overwrite=overwrite)

    report_devices("training", listener)

    overrides = overrides or {}
    # The dataset layout selects the task; a resumed/two-phase run (--from) keeps the
    # task recorded in its baseline config (see core/task.py and cvbench.tasks).
    cfg = build_config(data_dir, from_dir=from_dir, overrides=overrides)

    if "training.seed" in overrides and cfg.training.seed is not None:
        import keras
        keras.utils.set_random_seed(cfg.training.seed)
        emit(listener, SeedSet(cfg.training.seed))

    task = get_task(cfg.task)

    errors = task.validate_config(cfg)
    if errors:
        raise ConfigError("Invalid config for task " + repr(cfg.task) + ": " + "; ".join(errors))

    # The backbone default must be set before naming: make_run_name reads it.
    if not cfg.model.backbone:
        cfg.model.backbone = task.default_backbone

    spec = task.resolve_layout(cfg)
    resolved_weights = task.fit_class_weight(cfg, spec)

    if output_dir is not None:
        exp_dir = output_dir
    else:
        run_name = make_run_name(cfg)
        exp_dir = str(make_unique_dir(EXPERIMENTS_DIR, run_name))

    # Build datasets and model before touching the filesystem, so a bad dataset,
    # model or resume checkpoint leaves experiments/ unchanged.
    if resume and not os.path.isfile(resume):
        raise NotFoundError(f"Resume checkpoint not found: {resume}")
    train_ds, val_ds, num_train = task.build_datasets(cfg, spec)
    model = task.build_model(cfg)

    if overwrite and not resume and os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)  # fresh start: no stale checkpoints or history

    cfg.run.name = exp_dir.rstrip("/").split("/")[-1]
    cfg.run.date = date.today().strftime("%Y-%m-%d")
    cfg.run.status = "running"
    cfg.run.pid = os.getpid()
    cfg.run.primary_metric = task.primary_metric.name
    cfg.run.cli_command = " ".join([sys.argv[0].split("/")[-1]] + sys.argv[1:])

    save_config(cfg, exp_dir)

    # If anything below raises (dataset/model build, or the training loop itself),
    # record it as "failed" instead of leaving status stuck at "running" forever.
    try:
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
            listener=listener,
        )
    except KeyboardInterrupt:
        update_run_status(exp_dir, status="interrupted")
        raise
    except BaseException:
        update_run_status(exp_dir, status="failed")
        raise

    return exp_dir
