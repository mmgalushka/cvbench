from __future__ import annotations

from datetime import date

from cvbench.core.config import build_config, save_config, LossConfig
from cvbench.core.data import (
    build_datasets,
    get_class_distribution,
    get_class_names,
    print_imbalance_warning,
    resolve_class_weights,
)
from cvbench.core.model import build_model
from cvbench.core.runs import make_run_name, make_unique_dir, EXPERIMENTS_DIR
from cvbench.core import trainer as _trainer


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
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    input_size: int | None = None,
    dropout: float | None = None,
    aug_file: str | None = None,
    resume: str | None = None,
    class_weight=None,
    loss: LossConfig | None = None,
    lr_patience: int | None = None,
    lr_factor: float | None = None,
    lr_min: float | None = None,
    fine_tune_from_layer: int | None = None,
    use_lcn: bool | None = None,
    lcn_kernel_size: int | None = None,
    lcn_epsilon: float | None = None,
    val_split: float | None = None,
    mixup_alpha: float | None = None,
    mixup_background_class: str | None = None,
) -> str:
    """Orchestrate a full training run.

    Builds config, datasets, model, and delegates to core trainer.
    Returns the experiment directory path.
    """
    import platform
    import tensorflow as tf

    from cvbench.core import _fmt

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print(_fmt.green(f"🟢 Apple Silicon GPU (Metal) detected — training on {len(gpus)} device(s)"))
        else:
            names = ", ".join(g.name for g in gpus)
            print(_fmt.green(f"🟢 GPU detected: {len(gpus)} device(s) — {names}"))
    else:
        print(_fmt.yellow("⚠️  GPU not available, training on CPU"))

    cfg = build_config(
        data_dir=data_dir,
        from_dir=from_dir,
        backbone=backbone,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_size=input_size,
        dropout=dropout,
        class_weight=class_weight,
        loss=loss,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        lr_min=lr_min,
        fine_tune_from_layer=fine_tune_from_layer,
        use_lcn=use_lcn,
        lcn_kernel_size=lcn_kernel_size,
        lcn_epsilon=lcn_epsilon,
        val_split=val_split,
    )

    if aug_file:
        from cvbench.core.config import load_aug_file
        cfg.augmentation = load_aug_file(aug_file)

    if output_dir is not None:
        exp_dir = output_dir
    else:
        run_name = make_run_name(cfg)
        exp_dir = str(make_unique_dir(EXPERIMENTS_DIR, run_name))

    class_names = get_class_names(cfg.data.train_dir)
    cfg.data.classes = class_names
    cfg.model.num_classes = len(class_names)

    class_dist = get_class_distribution(cfg.data.train_dir)
    print_imbalance_warning(class_dist, cfg.training.class_weight)
    resolved_weights = resolve_class_weights(
        cfg.training.class_weight, class_dist, class_names
    )
    cfg.run.name = exp_dir.rstrip("/").split("/")[-1]
    cfg.run.date = date.today().strftime("%Y-%m-%d")
    cfg.run.status = "running"

    save_config(cfg, exp_dir)

    train_ds, val_ds, _, num_train = build_datasets(cfg)

    # Apply augmentation pipeline outside the model via tf.data.
    # Keras preprocessing layers must run in a native tf.data.map — calling them
    # inside tf.numpy_function strips graph context and causes internal shape errors
    # (e.g. RandomTranslation rank mismatch).  Custom aug_* functions are numpy-based
    # and still use numpy_function.
    if cfg.augmentation.transforms:
        from cvbench.augmentations.pipeline import build_keras_aug_fn, build_custom_aug_fn

        keras_aug = build_keras_aug_fn(cfg.augmentation.transforms)
        custom_aug = build_custom_aug_fn(cfg.augmentation.transforms)

        if keras_aug is not None:
            train_ds = train_ds.map(
                lambda x, y: (keras_aug(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if custom_aug is not None:
            def _custom_aug_map(x, y):
                x_aug = tf.numpy_function(lambda img: custom_aug(img), [x], tf.float32)
                x_aug.set_shape(x.shape)
                return x_aug, y
            train_ds = train_ds.map(_custom_aug_map, num_parallel_calls=tf.data.AUTOTUNE)

    if mixup_alpha and mixup_alpha > 0:
        from cvbench.augmentations.mixup import build_mixup_fn
        if mixup_background_class not in class_names:
            raise ValueError(
                f"--mixup-background-class '{mixup_background_class}' not found. "
                f"Available classes: {class_names}"
            )
        bg_idx = class_names.index(mixup_background_class)
        train_ds = train_ds.map(
            build_mixup_fn(background_class_idx=bg_idx, alpha=mixup_alpha),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        print(_fmt.dim(f" Mixup enabled: alpha={mixup_alpha}, background='{mixup_background_class}' (class {bg_idx})"))

    model = build_model(cfg)

    _trainer.train(
        cfg=cfg,
        exp_dir=exp_dir,
        train_ds=train_ds,
        val_ds=val_ds,
        class_names=class_names,
        model=model,
        resume_checkpoint=resume,
        num_train_samples=num_train,
        class_weight=resolved_weights,
    )

    return exp_dir
