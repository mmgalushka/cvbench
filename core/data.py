from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from core.config import CVBenchConfig


def get_class_names(train_dir: str) -> list[str]:
    """Derive class labels from sorted subdirectory names of train_dir."""
    return sorted(p.name for p in Path(train_dir).iterdir() if p.is_dir())


def build_dataset(
    directory: str,
    class_names: list[str],
    cfg: CVBenchConfig,
    training: bool = False,
) -> tf.data.Dataset:
    """Build a tf.data pipeline from an image directory.

    Args:
        directory: Path containing one subdirectory per class.
        class_names: Ordered list of class names (derived from train dir).
        cfg: Resolved experiment config.
        training: If True, apply shuffle and repeat; if False, no shuffle.

    Returns:
        Batched, prefetched tf.data.Dataset yielding (image, label) pairs.
        Images are float32 in [0, 255] — normalisation is the model's job.
    """
    size = cfg.model.input_size
    batch = cfg.data.batch_size

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(size, size),
        batch_size=batch,
        shuffle=training,
        seed=42 if training else None,
    )

    if training:
        ds = ds.repeat()

    return ds.prefetch(tf.data.AUTOTUNE)


def build_datasets(cfg: CVBenchConfig) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build train and val datasets and return class names.

    Returns:
        (train_ds, val_ds, class_names)
    """
    class_names = get_class_names(cfg.data.train_dir)
    train_ds = build_dataset(cfg.data.train_dir, class_names, cfg, training=True)
    val_ds = build_dataset(cfg.data.val_dir, class_names, cfg, training=False)
    return train_ds, val_ds, class_names
