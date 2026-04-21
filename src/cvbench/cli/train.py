import json

import click

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
@click.option("--fine-tune-from-layer", default=None, type=int,
              help="Unfreeze backbone from this layer index onward (0=frozen, -1=all layers).")
@click.option("--use-lcn", is_flag=True, default=False,
              help="Insert a Local Contrast Normalization layer before the backbone (brightness-invariant training).")
@click.option("--lcn-kernel-size", default=None, type=int,
              help="LCN neighbourhood size in pixels (default: 32).")
@click.option("--lcn-epsilon", default=None, type=float,
              help="LCN stability constant for division (default: 1e-3).")
@click.option("--val-split", default=None, type=float,
              help="Fraction of train set to use for validation when no val/ directory exists (default: 0.2).")
def train(
    data_dir, output_dir, from_dir, backbone, epochs, lr,
    batch_size, input_size, dropout, aug_file, resume, class_weight_raw,
    lr_patience, lr_factor, lr_min, fine_tune_from_layer,
    use_lcn, lcn_kernel_size, lcn_epsilon, val_split,
):
    """Train a model on DATA_DIR.

    DATA_DIR must contain train/, val/, and test/ subdirectories.
    All parameters have sensible defaults and can be overridden individually.
    """
    class_weight = _parse_class_weight(class_weight_raw)
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
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        lr_min=lr_min,
        fine_tune_from_layer=fine_tune_from_layer,
        use_lcn=use_lcn or None,
        lcn_kernel_size=lcn_kernel_size,
        lcn_epsilon=lcn_epsilon,
        val_split=val_split,
    )
