import click

from core.config import build_config, save_config
from core.data import build_datasets, get_class_names
from core.model import build_model
from core.registry import resolve_aug
from core.runs import make_run_name, make_unique_dir
from core import trainer as _trainer

# Import builtin augmentations to trigger registration
import augmentations  # noqa: F401


_DEFAULT_EXPERIMENTS_DIR = "experiments"


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
@click.option("--aug-preset", default=None, help="Augmentation preset name.")
@click.option("--aug-placement", default=None,
              type=click.Choice(["inside_model", "outside_model"]),
              help="Augmentation placement.")
@click.option("--resume", default=None,
              help="Path to a checkpoint file to resume training from.")
def train(
    data_dir, output_dir, from_dir, backbone, epochs, lr,
    batch_size, input_size, dropout, aug_preset, aug_placement, resume,
):
    """Train a model on DATA_DIR.

    DATA_DIR must contain train/, val/, and test/ subdirectories.
    All parameters have sensible defaults and can be overridden individually.
    """
    from datetime import date

    cfg = build_config(
        data_dir=data_dir,
        from_dir=from_dir,
        backbone=backbone,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_size=input_size,
        dropout=dropout,
        aug_preset=aug_preset,
        aug_placement=aug_placement,
    )

    # Determine output directory
    if output_dir is not None:
        exp_dir = output_dir
    else:
        run_name = make_run_name(cfg)
        exp_dir = str(make_unique_dir(_DEFAULT_EXPERIMENTS_DIR, run_name))

    # Detect classes and fill derived fields before saving config
    class_names = get_class_names(cfg.data.train_dir)
    cfg.data.classes = class_names
    cfg.model.num_classes = len(class_names)
    cfg.run.name = exp_dir.rstrip("/").split("/")[-1]
    cfg.run.date = date.today().strftime("%Y-%m-%d")
    cfg.run.status = "running"

    # Write config.yaml before training starts — single source of truth
    save_config(cfg, exp_dir)

    # Build augmentation layer
    aug_layer = resolve_aug(cfg.augmentation.preset, cfg.augmentation.params)

    # Build datasets
    train_ds, val_ds, _ = build_datasets(cfg)

    # For outside_model, pass None so the model graph is aug-free
    inside_aug = aug_layer if cfg.augmentation.placement == "inside_model" else None
    model = build_model(cfg, aug_layer=inside_aug)

    if cfg.augmentation.placement == "outside_model":
        import tensorflow as tf
        train_ds = train_ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    _trainer.train(
        cfg=cfg,
        exp_dir=exp_dir,
        train_ds=train_ds,
        val_ds=val_ds,
        class_names=class_names,
        model=model,
        resume_checkpoint=resume,
    )
