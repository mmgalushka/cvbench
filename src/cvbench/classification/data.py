from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from cvbench.core import _console
from cvbench.core.config import CVBenchConfig
from cvbench.core.input_pipeline import build_image_dataset

# The formats tf.image.decode_image reads (and image_dataset_from_directory lists).
_IMAGE_EXTS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def get_class_names(train_dir: str) -> list[str]:
    """Derive class labels from sorted subdirectory names of train_dir."""
    return sorted(p.name for p in Path(train_dir).iterdir() if p.is_dir())


def list_labeled_images(directory: str, class_names: list[str]) -> tuple[list[str], list[int]]:
    """List every image under DIRECTORY/<class>/ as (paths, class indices).

    Ordered by class, then by path — a stable order for unshuffled datasets.
    """
    paths, labels = [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(directory) / class_name
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                paths.append(str(f))
                labels.append(class_idx)
    return paths, labels


def _dataset(
    paths: list[str],
    labels: list[int],
    class_names: list[str],
    cfg: CVBenchConfig,
    training: bool,
    transforms: list | None = None,
) -> tf.data.Dataset:
    one_hot = np.eye(len(class_names), dtype=np.float32)[np.asarray(labels, dtype=np.int64)]
    return build_image_dataset(
        paths,
        one_hot,
        image_size=cfg.model.input_size,
        batch_size=cfg.data.batch_size,
        training=training,
        seed=cfg.training.seed,
        transforms=transforms,
    )


def build_dataset(
    directory: str,
    class_names: list[str],
    cfg: CVBenchConfig,
    training: bool = False,
    transforms: list | None = None,
) -> tf.data.Dataset:
    """Build a tf.data pipeline from an image directory.

    Args:
        directory: Path containing one subdirectory per class.
        class_names: Ordered list of class names (derived from train dir).
        cfg: Resolved experiment config.
        training: If True, shuffle every epoch, augment and repeat.
        transforms: Augmentation transforms applied when training.

    Returns:
        Batched, prefetched tf.data.Dataset yielding (image, one-hot label) pairs.
        Images are RGB float32 in [0, 255] — the model's Rescaling layer normalizes.
    """
    paths, labels = list_labeled_images(directory, class_names)
    return _dataset(paths, labels, class_names, cfg, training, transforms)


def build_datasets(
    cfg: CVBenchConfig,
    transforms: list | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], int]:
    """Build train and val datasets and return class names and training sample count.

    When val/ directory is absent, splits training data using cfg.data.val_split
    (stratified, so every class is proportionally represented in both subsets).
    TRANSFORMS augment the training dataset only.

    Returns:
        (train_ds, val_ds, class_names, num_train_samples)
    """
    class_names = get_class_names(cfg.data.train_dir)
    all_paths, all_labels = list_labeled_images(cfg.data.train_dir, class_names)
    total_train = len(all_paths)

    if os.path.isdir(cfg.data.val_dir):
        if cfg.data.val_split_explicit:
            _console.warning(
                f"--val-split ignored: a val/ directory was found at {cfg.data.val_dir!r}."
                " Remove val/ or omit --val-split to silence this warning."
            )
        val_paths, val_labels = list_labeled_images(cfg.data.val_dir, class_names)
        train_paths, train_labels = all_paths, all_labels
        print(_console.dim(f" Found {total_train} files for training ({len(class_names)} classes)."))
        print(_console.dim(f" Found {len(val_paths)} files for validation ({len(class_names)} classes)."))
    else:
        split = cfg.data.val_split
        pct_train = int((1 - split) * 100)
        pct_val = int(split * 100)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels,
            test_size=split,
            stratify=all_labels,
            random_state=cfg.training.seed,
        )
        print(_console.dim(
            f" Found {total_train} files belonging to {len(class_names)} classes"
            f" — auto-splitting ({pct_train}/{pct_val})"
        ))
        print(_console.dim(f"   ├─ {len(train_paths)} for training"))
        print(_console.dim(f"   └─ {len(val_paths)} for validation"))

    train_ds = _dataset(train_paths, train_labels, class_names, cfg, True, transforms)
    val_ds = _dataset(val_paths, val_labels, class_names, cfg, False)
    return train_ds, val_ds, class_names, len(train_paths)
