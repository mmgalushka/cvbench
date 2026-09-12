from __future__ import annotations

import contextlib
import io
import math
import os
import random
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from cvbench.core.config import CVBenchConfig
from cvbench.core import _console


def get_class_names(train_dir: str) -> list[str]:
    """Derive class labels from sorted subdirectory names of train_dir."""
    return sorted(p.name for p in Path(train_dir).iterdir() if p.is_dir())


def stratified_image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    image_size=(256, 256),
    batch_size=32,
    seed=None,
    validation_split=None,
    subset=None,
    shuffle=True,
) -> tf.data.Dataset:
    """Like tf.keras.utils.image_dataset_from_directory but with stratified random split.

    Guarantees each class is proportionally represented in both train and val subsets,
    regardless of class imbalance or filename ordering.
    """
    directory = Path(directory)
    if class_names is None:
        class_names = sorted(p.name for p in directory.iterdir() if p.is_dir())

    all_paths, all_labels = [], []
    for class_idx, class_name in enumerate(class_names):
        for f in sorted((directory / class_name).iterdir()):
            if f.is_file():
                all_paths.append(str(f))
                all_labels.append(class_idx)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=validation_split,
        stratify=all_labels,
        random_state=seed,
    )

    paths, label_indices = (
        (train_paths, train_labels) if subset == "training" else (val_paths, val_labels)
    )

    one_hot_labels = tf.one_hot(label_indices, len(class_names))
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(paths), one_hot_labels))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32)
        return img, label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache().batch(batch_size)


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
        Images are RGB float32 in [0, 255] — the model's Rescaling layer normalizes.
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
        seed=cfg.training.seed if training else None,
    )

    ds = ds.cache()

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
        if cfg.data.val_split_explicit:
            _console.warning(
                f"--val-split ignored: a val/ directory was found at {cfg.data.val_dir!r}."
                " Remove val/ or omit --val-split to silence this warning."
            )
        n_val = sum(1 for _ in Path(cfg.data.val_dir).glob("*/*"))
        with contextlib.redirect_stdout(io.StringIO()):
            train_ds = build_dataset(cfg.data.train_dir, class_names, cfg, training=True)
            val_ds = build_dataset(cfg.data.val_dir, class_names, cfg, training=False)
        print(_console.dim(f" Found {total_train} files for training ({len(class_names)} classes)."))
        print(_console.dim(f" Found {n_val} files for validation ({len(class_names)} classes)."))
        num_train_samples = total_train
    else:
        split = cfg.data.val_split
        pct_train = int((1 - split) * 100)
        pct_val = int(split * 100)
        seed = cfg.training.seed if cfg.training.seed is not None else random.randint(0, 2**31 - 1)

        common_kwargs = dict(
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=(size, size),
            batch_size=batch,
            seed=seed,
            validation_split=split,
        )
        train_ds = (
            stratified_image_dataset_from_directory(
                cfg.data.train_dir, subset="training", shuffle=True, **common_kwargs
            )
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = stratified_image_dataset_from_directory(
            cfg.data.train_dir, subset="validation", shuffle=False, **common_kwargs
        ).prefetch(tf.data.AUTOTUNE)
        num_train_samples = math.floor(total_train * (1 - split))
        n_val_samples = total_train - num_train_samples
        print(_console.dim(
            f" Found {total_train} files belonging to {len(class_names)} classes"
            f" — auto-splitting ({pct_train}/{pct_val})"
        ))
        print(_console.dim(f"   ├─ {num_train_samples} for training"))
        print(_console.dim(f"   └─ {n_val_samples} for validation"))

    return train_ds, val_ds, class_names, num_train_samples
