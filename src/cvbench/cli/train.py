import json

import click

from cvbench.core import _fmt
from cvbench.core.config import build_config, save_config
from cvbench.core.data import (
    build_datasets,
    get_class_distribution,
    get_class_names,
    print_class_balance,
    resolve_class_weights,
)
from cvbench.core.model import build_model
from cvbench.core.runs import make_run_name, make_unique_dir, EXPERIMENTS_DIR
from cvbench.core import trainer as _trainer


def _parse_class_weight(value: str | None):
    """Parse --class-weight CLI value: null → None, auto → 'auto', JSON → dict."""
    if value is None or value.lower() in ("null", "none"):
        return None
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    raise click.BadParameter(
        f"Expected null, auto, or a JSON dict like '{{\"cat\": 1.0, \"dog\": 2.5}}', got: {value!r}"
    )


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option("--output", "output_dir", default=None,
              help="Experiment output directory (default: experiments/<auto-name>/).")
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True),
              help="Load config from an existing experiment as baseline.")
@click.option("--backbone", default=None, help="Backbone name (e.g. efficientnet_b0).")
@click.option("--epochs", default=None, type=int, help="Number of training epochs.")
@click.option("--lr", default=None, type=float, help="Learning rate.")
@click.option("--batch-size", default=None, type=int, help="Batch size.")
@click.option("--input-size", default=None, type=int, help="Image input size in pixels.")
@click.option("--dropout", default=None, type=float, help="Dropout rate.")
@click.option("--augmentation", "aug_file", default=None, type=click.Path(exists=True),
              help="Path to an augmentation YAML file.")
@click.option("--resume", default=None,
              help="Path to a checkpoint file to resume training from.")
@click.option("--class-weight", "class_weight_raw", default=None,
              help="Class weighting: null | auto | '{\"cat\": 1.0, \"dog\": 2.5}'")
@click.option("--lr-patience", default=None, type=int,
              help="Enable ReduceLROnPlateau: reduce LR after N epochs with no improvement.")
@click.option("--lr-factor", default=None, type=float,
              help="LR reduction factor (default 0.5). Requires --lr-patience.")
@click.option("--lr-min", default=None, type=float,
              help="Minimum LR floor (default 1e-7). Requires --lr-patience.")
def train(
    data_dir, output_dir, from_dir, backbone, epochs, lr,
    batch_size, input_size, dropout, aug_file, resume, class_weight_raw,
    lr_patience, lr_factor, lr_min,
):
    """Train a model on DATA_DIR.

    DATA_DIR must contain train/, val/, and test/ subdirectories.
    All parameters have sensible defaults and can be overridden individually.
    """
    from datetime import date

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        names = ", ".join(g.name for g in gpus)
        print(_fmt.green(f"🟢 GPU detected: {len(gpus)} device(s) — {names}"))
    else:
        print(_fmt.yellow("⚠️  GPU not available, training on CPU"))

    class_weight_cfg = _parse_class_weight(class_weight_raw)

    cfg = build_config(
        data_dir=data_dir,
        from_dir=from_dir,
        backbone=backbone,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_size=input_size,
        dropout=dropout,
        class_weight=class_weight_cfg,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        lr_min=lr_min,
    )

    if aug_file:
        from cvbench.core.config import load_aug_file
        cfg.augmentation = load_aug_file(aug_file)

    # Determine output directory
    if output_dir is not None:
        exp_dir = output_dir
    else:
        run_name = make_run_name(cfg)
        exp_dir = str(make_unique_dir(EXPERIMENTS_DIR, run_name))

    # Detect classes and fill derived fields before saving config
    class_names = get_class_names(cfg.data.train_dir)
    cfg.data.classes = class_names
    cfg.model.num_classes = len(class_names)

    # Class balance report + resolve weights
    class_dist = get_class_distribution(cfg.data.train_dir)
    print_class_balance(class_dist, cfg.training.class_weight)
    resolved_weights = resolve_class_weights(
        cfg.training.class_weight, class_dist, class_names
    )
    cfg.run.name = exp_dir.rstrip("/").split("/")[-1]
    cfg.run.date = date.today().strftime("%Y-%m-%d")
    cfg.run.status = "running"

    # Write config.yaml before training starts — single source of truth
    save_config(cfg, exp_dir)

    # Build datasets
    train_ds, val_ds, _, num_train = build_datasets(cfg)

    # Apply augmentation pipeline outside the model via tf.data.
    # Keras preprocessing layers must run in a native tf.data.map — calling them
    # inside tf.numpy_function strips graph context and causes internal shape errors
    # (e.g. RandomTranslation rank mismatch).  Custom aug_* functions are numpy-based
    # and still use numpy_function.
    if cfg.augmentation.transforms:
        import tensorflow as tf
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
