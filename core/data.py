from __future__ import annotations

import contextlib
import io
import math
import os
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


def build_datasets(
    cfg: CVBenchConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], int]:
    """Build train and val datasets and return class names and training sample count.

    When val/ directory is absent, splits training data using cfg.data.val_split.

    Returns:
        (train_ds, val_ds, class_names, num_train_samples)
    """
    class_names = get_class_names(cfg.data.train_dir)
    size = cfg.model.input_size
    batch = cfg.data.batch_size

    total_train = sum(1 for _ in Path(cfg.data.train_dir).glob("*/*"))

    if os.path.isdir(cfg.data.val_dir):
        n_val = sum(1 for _ in Path(cfg.data.val_dir).glob("*/*"))
        with contextlib.redirect_stdout(io.StringIO()):
            train_ds = build_dataset(cfg.data.train_dir, class_names, cfg, training=True)
            val_ds = build_dataset(cfg.data.val_dir, class_names, cfg, training=False)
        print(f" Found {total_train} files for training ({len(class_names)} classes).")
        print(f" Found {n_val} files for validation ({len(class_names)} classes).")
        num_train_samples = total_train
    else:
        split = cfg.data.val_split
        pct_train = int((1 - split) * 100)
        pct_val = int(split * 100)

        common_kwargs = dict(
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=(size, size),
            batch_size=batch,
            seed=42,
            validation_split=split,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            train_ds = (
                tf.keras.utils.image_dataset_from_directory(
                    cfg.data.train_dir, subset="training", shuffle=True, **common_kwargs
                )
                .repeat()
                .prefetch(tf.data.AUTOTUNE)
            )
            val_ds = (
                tf.keras.utils.image_dataset_from_directory(
                    cfg.data.train_dir, subset="validation", shuffle=False, **common_kwargs
                )
                .prefetch(tf.data.AUTOTUNE)
            )
        num_train_samples = math.floor(total_train * (1 - split))
        n_val_samples = total_train - num_train_samples
        print(
            f" Found {total_train} files belonging to {len(class_names)} classes"
            f" — auto-splitting ({pct_train}/{pct_val})"
        )
        print(f"   ├─ {num_train_samples} for training")
        print(f"   └─ {n_val_samples} for validation")

    return train_ds, val_ds, class_names, num_train_samples
