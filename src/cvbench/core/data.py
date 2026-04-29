from __future__ import annotations

import contextlib
import io
import math
import os
from pathlib import Path

import tensorflow as tf

from cvbench.core.config import CVBenchConfig
from cvbench.core import _fmt


def get_class_names(train_dir: str) -> list[str]:
    """Derive class labels from sorted subdirectory names of train_dir."""
    return sorted(p.name for p in Path(train_dir).iterdir() if p.is_dir())


def get_class_distribution(train_dir: str) -> dict[str, int]:
    """Count image files per class. Returns {class_name: count} sorted by count descending."""
    dist = {
        p.name: sum(1 for f in p.iterdir() if f.is_file())
        for p in Path(train_dir).iterdir()
        if p.is_dir()
    }
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def compute_auto_weights(
    class_dist: dict[str, int], class_names: list[str]
) -> dict[int, float]:
    """Inverse-frequency class weights keyed by class index for Keras model.fit()."""
    total = sum(class_dist.values())
    n = len(class_dist)
    return {
        class_names.index(cls): round(total / (n * count), 4)
        for cls, count in class_dist.items()
    }


def resolve_class_weights(
    class_weight_cfg,
    class_dist: dict[str, int],
    class_names: list[str],
) -> dict[int, float] | None:
    """Resolve class_weight config value to a {class_index: weight} dict for Keras, or None."""
    if class_weight_cfg is None:
        return None
    if class_weight_cfg == "auto":
        return compute_auto_weights(class_dist, class_names)
    if isinstance(class_weight_cfg, dict):
        return {class_names.index(cls): float(w) for cls, w in class_weight_cfg.items()}
    return None


def print_class_distribution(class_dist: dict[str, int]) -> None:
    """Print per-class sample counts with a bar chart."""
    counts = list(class_dist.values())
    max_count = max(counts)
    min_count = min(counts)
    total = sum(counts)
    ratio = max_count / min_count if min_count > 0 else float("inf")
    uniform = all(c == counts[0] for c in counts)
    max_cls = max(len(cls) for cls in class_dist)

    bar_width = 20
    print(f" {_fmt.bold('Class distribution:')}")
    print(_fmt.dim(f"   {'Class':<{max_cls}}  {'Images':>6}  {'':^{bar_width}}  {'%':>5}"))
    for cls, count in class_dist.items():
        pct = count / total * 100
        if uniform:
            print(f"   {cls:<{max_cls}}  {count:>6}  {'':^{bar_width}}  {pct:.1f}%")
        else:
            bar = "█" * int(count / max_count * bar_width)
            print(f"   {cls:<{max_cls}}  {count:>6}  {bar:<{bar_width}}  {pct:.1f}%")

    std_counts = (sum((c - total / len(counts)) ** 2 for c in counts) / len(counts)) ** 0.5
    imbalanced = std_counts > 0 and any(abs(c - total / len(counts)) > std_counts for c in counts)
    if imbalanced:
        print()
        print(_fmt.yellow(f" ⚠️  Imbalance ratio {ratio:.0f}:1 detected"))


def print_imbalance_warning(class_dist: dict[str, int], class_weight_cfg) -> None:
    """Print an imbalance warning with class-weight tip when ratio >= 3:1."""
    counts = list(class_dist.values())
    max_count = max(counts)
    min_count = min(counts) if min(counts) > 0 else 1
    ratio = max_count / min_count

    if ratio >= 3.0:
        print(_fmt.yellow(f" ⚠️  Imbalance ratio {ratio:.0f}:1 detected"))
        if class_weight_cfg is None:
            print(f"   {_fmt.dim('Tip: rerun with --class-weight auto')}")
        elif class_weight_cfg == "auto":
            print(f"   {_fmt.green('✓ class_weight=auto applied')}")
        else:
            print(f"   {_fmt.green('✓ custom class weights applied')}")


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
        if cfg.data.val_split_explicit:
            print(_fmt.yellow(
                f"⚠️  --val-split ignored: a val/ directory was found at {cfg.data.val_dir!r}."
                " Remove val/ or omit --val-split to silence this warning."
            ))
        n_val = sum(1 for _ in Path(cfg.data.val_dir).glob("*/*"))
        with contextlib.redirect_stdout(io.StringIO()):
            train_ds = build_dataset(cfg.data.train_dir, class_names, cfg, training=True)
            val_ds = build_dataset(cfg.data.val_dir, class_names, cfg, training=False)
        print(_fmt.dim(f" Found {total_train} files for training ({len(class_names)} classes)."))
        print(_fmt.dim(f" Found {n_val} files for validation ({len(class_names)} classes)."))
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
            _val_raw = tf.keras.utils.image_dataset_from_directory(
                cfg.data.train_dir, subset="validation", shuffle=False, **common_kwargs
            )
            val_ds = _val_raw.prefetch(tf.data.AUTOTUNE)
        num_train_samples = math.floor(total_train * (1 - split))
        n_val_samples = total_train - num_train_samples
        print(_fmt.dim(
            f" Found {total_train} files belonging to {len(class_names)} classes"
            f" — auto-splitting ({pct_train}/{pct_val})"
        ))
        print(_fmt.dim(f"   ├─ {num_train_samples} for training"))
        print(_fmt.dim(f"   └─ {n_val_samples} for validation"))

    return train_ds, val_ds, class_names, num_train_samples
