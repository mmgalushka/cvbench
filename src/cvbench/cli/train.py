import json

import click

from cvbench.cli import _help
from cvbench.core.config import (
    DataConfig,
    LossConfig,
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)

_DATA = DataConfig()
_MODEL = ModelConfig()
_TRAIN = TrainingConfig()

# NOTE: cvbench.services.training is imported inside train() — it pulls in
# TensorFlow, and importing it at module scope would make `train --help` (and
# the `commands` overview, which enumerates every command) take ~1.5s to start.


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
        params: dict[str, str] = {}
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


@_help.command(
    examples=[
        ("train data/synthetic --epochs 5",
         "Smoke-test the whole pipeline on the synthetic dataset"),
        ("train data --epochs 20 --backbone efficientnet_b0",
         "Train on your own dataset (needs train/, val/, test/ subfolders)"),
        ("train data --augmentation standard --loss focal:gamma=2.0",
         "Add a saved augmentation config and swap the loss function"),
    ],
    see_also=[
        ("runs list", "find the name of the run you just created"),
        ("evaluate <run>", "score it on the held-out test split"),
        ("runs export <run> --format tflite", "package it for a device"),
    ],
)
@click.argument("data_dir")
@click.option("--output", "output_dir", default=None,
              help="Experiment output directory (default: experiments/<auto-name>/).")
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True),
              help="Load config from an existing experiment as baseline (default: none).")
@click.option("--backbone", default=None,
              help="Backbone name (efficientnet_b0..b5, resnet_18, resnet_50). "
                   f"(default: {_MODEL.backbone}; detection defaults to resnet_18 "
                   "unless this is passed explicitly).")
@click.option("--weights", default=None, type=click.Choice(["imagenet", "none"]),
              help=f"Backbone weight init: imagenet (pretrained) or none (random/scratch) "
                   f"(default: {_MODEL.weights}).")
@click.option("--epochs", default=None, type=int, help=f"Number of training epochs (default: {_TRAIN.epochs}).")
@click.option("--lr", default=None, type=float, help=f"Learning rate (default: {_TRAIN.learning_rate:g}).")
@click.option("--batch-size", default=None, type=int, help=f"Batch size (default: {_DATA.batch_size}).")
@click.option("--input-size", default=None, type=int,
              help=f"Image input size in pixels (default: {_MODEL.input_size}).")
@click.option("--dropout", default=None, type=float, help=f"Dropout rate (default: {_MODEL.dropout}).")
@click.option("--augmentation", "aug_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to an augmentation YAML file, see 'data aug' (default: none).")
@click.option("--resume", default=None,
              help="Path to a checkpoint file to resume training from (default: none).")
@click.option("--class-weight", "class_weight_raw", default=None,
              help="Class weighting: null | auto | '{\"cat\": 1.0, \"dog\": 2.5}' (default: null).")
@click.option("--loss", "loss_raw", default=None,
              help="Loss function: crossentropy | focal | focal:gamma=2.0 | "
                   "focal:gamma=2.0,label_smoothing=0.1 (default: crossentropy)")
@click.option("--optimizer", "optimizer_raw", default=None,
              help="Optimizer: adam | sgd | adam:weight_decay=1e-4 | "
                   "sgd:weight_decay=1e-4,momentum=0.9 (default: adam)")
@click.option("--lr-scheduler", "lr_scheduler_raw", default=None,
              help="LR scheduler: patience=5 | patience=5,factor=0.5,min=1e-7 (default: disabled).")
@click.option("--fine-tune-from-layer", default=None, type=int,
              help="Unfreeze the backbone from this layer index onward: 0 keeps it FROZEN "
                   "(only the new head trains), -1 unfreezes all layers "
                   f"(default: {_MODEL.fine_tune_from_layer}, frozen). "
                   "A frozen backbone often ends near chance accuracy on small or "
                   "unusual datasets; try -1 with a lower --lr.")
@click.option("--val-split", default=None, type=float,
              help="Fraction of train set to use for validation when no val/ directory "
                   f"exists (default: {_DATA.val_split}).")
@click.option("--seed", default=None, type=int,
              help="Random seed for reproducibility, sets Python, NumPy and TensorFlow seeds "
                   "(default: none, not seeded).")
def train(
    data_dir, output_dir, from_dir, backbone, weights, epochs, lr,
    batch_size, input_size, dropout, aug_file, resume, class_weight_raw,
    loss_raw, optimizer_raw, lr_scheduler_raw, fine_tune_from_layer, val_split, seed,
):
    """Train a model on DATA_DIR.

    DATA_DIR must contain train/, val/, and test/ subdirectories. It can be a
    full path (data/my_dataset) or a bare dataset name (my_dataset), resolved
    under data/.

    The task is auto-detected: a dataset with images/ and labels/ folders is
    trained as detection, a dataset of per-class folders as classification.

    Every option has a default (shown in its help). The default backbone is
    frozen, so only the new head trains; if accuracy ends near chance, unfreeze
    it with --fine-tune-from-layer -1 and a lower --lr, or train longer with
    --epochs, a stronger --augmentation, or --class-weight auto on imbalanced
    data.
    """
    from cvbench.core.data_store import resolve_data_dir
    from cvbench.services.training import run_training  # deferred: pulls in TensorFlow

    data_dir = resolve_data_dir(data_dir)

    class_weight = _parse_class_weight(class_weight_raw)
    loss = _parse_loss(loss_raw)
    optimizer = _parse_optimizer(optimizer_raw)
    lr_scheduler = _parse_lr_scheduler(lr_scheduler_raw)
    run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        from_dir=from_dir,
        backbone=backbone,
        weights=weights,
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
