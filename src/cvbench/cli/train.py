import json

import click

from cvbench.core.config import LossConfig, OptimizerConfig, LRSchedulerConfig
from cvbench.services.training import run_training


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


def _parse_optimizer(value: str | None) -> OptimizerConfig | None:
    if value is None:
        return None
    if ":" not in value:
        opt_type = value.strip()
        params = {}
    else:
        opt_type, params_str = value.split(":", 1)
        params = {}
        for kv in params_str.split(","):
            k, v = kv.split("=", 1)
            params[k.strip()] = v.strip()
    known = {"adam", "sgd"}
    if opt_type.strip() not in known:
        raise click.BadParameter(
            f"Unknown optimizer type '{opt_type}'. Valid options: {', '.join(sorted(known))}"
        )
    return OptimizerConfig(
        type=opt_type.strip(),
        weight_decay=float(params.get("weight_decay", 0.0)),
        momentum=float(params.get("momentum", 0.9)),
    )


def _parse_lr_scheduler(value: str | None) -> LRSchedulerConfig | None:
    if value is None:
        return None
    params = {}
    for kv in value.split(","):
        k, v = kv.split("=", 1)
        params[k.strip()] = v.strip()
    return LRSchedulerConfig(
        patience=int(params.get("patience", 5)),
        factor=float(params.get("factor", 0.5)),
        min_lr=float(params.get("min", 1e-7)),
    )


def _parse_loss(value: str | None) -> LossConfig | None:
    if value is None:
        return None
    if ":" not in value:
        return LossConfig(type=value)
    loss_type, params_str = value.split(":", 1)
    params = {}
    for kv in params_str.split(","):
        k, v = kv.split("=", 1)
        params[k.strip()] = v.strip()
    known = {"crossentropy", "focal"}
    if loss_type.strip() not in known:
        raise click.BadParameter(
            f"Unknown loss type '{loss_type}'. Valid options: {', '.join(sorted(known))}"
        )
    return LossConfig(
        type=loss_type.strip(),
        label_smoothing=float(params.get("label_smoothing", 0.0)),
        focal_gamma=float(params.get("gamma", 2.0)),
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
@click.option("--loss", "loss_raw", default=None,
              help="Loss function: crossentropy | focal | focal:gamma=2.0 | focal:gamma=2.0,label_smoothing=0.1")
@click.option("--optimizer", "optimizer_raw", default=None,
              help="Optimizer: adam | sgd | adam:weight_decay=1e-4 | sgd:weight_decay=1e-4,momentum=0.9")
@click.option("--lr-scheduler", "lr_scheduler_raw", default=None,
              help="LR scheduler: patience=5 | patience=5,factor=0.5,min=1e-7")
@click.option("--fine-tune-from-layer", default=None, type=int,
              help="Unfreeze backbone from this layer index onward (0=frozen, -1=all layers).")
@click.option("--val-split", default=None, type=float,
              help="Fraction of train set to use for validation when no val/ directory exists (default: 0.2).")
@click.option("--seed", default=None, type=int,
              help="Random seed for reproducibility (sets Python, NumPy, and TensorFlow seeds).")
def train(
    data_dir, output_dir, from_dir, backbone, epochs, lr,
    batch_size, input_size, dropout, aug_file, resume, class_weight_raw,
    loss_raw, optimizer_raw, lr_scheduler_raw, fine_tune_from_layer, val_split, seed,
):
    """Train a model on DATA_DIR.

    DATA_DIR must contain train/, val/, and test/ subdirectories.
    All parameters have sensible defaults and can be overridden individually.
    """
    class_weight = _parse_class_weight(class_weight_raw)
    loss = _parse_loss(loss_raw)
    optimizer = _parse_optimizer(optimizer_raw)
    lr_scheduler = _parse_lr_scheduler(lr_scheduler_raw)
    run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        from_dir=from_dir,
        backbone=backbone,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_size=input_size,
        dropout=dropout,
        aug_file=aug_file,
        resume=resume,
        class_weight=class_weight,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        fine_tune_from_layer=fine_tune_from_layer,
        val_split=val_split,
        seed=seed,
    )
