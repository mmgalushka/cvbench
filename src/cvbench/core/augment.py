"""Apply an augmentation pipeline to a training tf.data.Dataset.

Split out of services/training.py so both the classification and detection
orchestration paths can share it.
"""
from __future__ import annotations

import tensorflow as tf


def apply_augmentation(train_ds: tf.data.Dataset, transforms: list) -> tf.data.Dataset:
    """Map the configured augmentation TRANSFORMS over TRAIN_DS's images.

    Keras preprocessing layers must run in a native tf.data.map — calling them
    inside tf.numpy_function strips graph context and causes internal shape
    errors (e.g. RandomTranslation rank mismatch). Custom aug_* functions are
    numpy-based and still use numpy_function.
    """
    if not transforms:
        return train_ds

    from cvbench.augmentations.pipeline import build_custom_aug_fn, build_keras_aug_fn

    keras_aug = build_keras_aug_fn(transforms)
    custom_aug = build_custom_aug_fn(transforms)

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

    return train_ds
