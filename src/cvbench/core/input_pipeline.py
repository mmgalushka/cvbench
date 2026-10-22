"""The one tf.data input pipeline every task trains, validates and evaluates on.

Tasks supply the image paths and their per-image targets; this module owns the
order of operations, which is where silent training bugs hide:

    list files -> read bytes -> cache -> shuffle -> decode, resize to float32
    -> [augment, resize back] -> batch -> repeat -> prefetch

Augmentation runs on the resized image, so pixel-sized parameters (line
widths, crop sizes) mean the same thing whatever the source resolution.

* The cache holds the *encoded* file bytes, so its memory tracks the dataset's
  size on disk, not its decoded float32 size (~600 KB per 224px image).
* Shuffling comes after the cache and reshuffles every iteration, so each
  epoch sees a fresh order (a shuffle *before* the cache would replay the
  first epoch's order forever).
* Augmentation runs per image, before ``batch()``, so a transform's ``prob``
  is a per-image probability. Each image carries its own seed drawn from
  ``Dataset.random``, so a seeded run augments identically every time,
  however many parallel calls tf.data uses.
"""
from __future__ import annotations

import secrets

import numpy as np
import tensorflow as tf


def build_image_dataset(
    image_paths: list[str],
    targets,
    image_size: int,
    batch_size: int,
    training: bool = False,
    seed: int | None = None,
    transforms: list | None = None,
    joint_targets: bool = False,
) -> tf.data.Dataset:
    """Build a batched, prefetched dataset of ``(image, target)`` pairs.

    Args:
        image_paths: one image file per sample.
        targets: per-sample targets, sliced along axis 0 — an array, or a
            tuple of arrays (e.g. detection's per-scale targets). With JOINT_TARGETS,
            one target image file path per sample instead.
        image_size: images are stretch-resized to (image_size, image_size).
        batch_size: samples per batch.
        training: shuffle every epoch, apply TRANSFORMS and repeat forever.
        seed: makes shuffle order and augmentation reproducible; a fresh
            seed is drawn when None.
        transforms: augmentation transforms (``cfg.augmentation.transforms``,
            already filtered by the task); applied only when TRAINING.
        joint_targets: the target is an image (a segmentation mask): it is decoded to one
            channel, stretch-resized with nearest-neighbour and moved together with the
            image by geometric augmentation (``Task.label_spec.joint_targets``).

    Returns:
        Dataset yielding (image (size, size, 3) float32 in [0, 255], target)
        batches. A training dataset repeats; others run one pass.
    """
    deterministic = seed is not None
    if seed is None:
        seed = secrets.randbits(31)

    ds = cached_bytes(image_paths, targets, read_targets=joint_targets)

    if training:
        ds = ds.shuffle(max(len(image_paths), 1), seed=seed, reshuffle_each_iteration=True)

    augment = _build_augment(transforms or [], seed, deterministic, joint_targets) if training else None

    def _resize(img):
        img = tf.image.resize(img, (image_size, image_size))  # stretch; float32
        img.set_shape((image_size, image_size, 3))
        return img

    def _resize_target(target):
        if not joint_targets:
            return target
        target = tf.image.resize(target, (image_size, image_size), method="nearest")
        target.set_shape((image_size, image_size, 1))
        return target

    def _decode(img_bytes):
        return _resize(tf.image.decode_image(img_bytes, channels=3, expand_animations=False))

    def _decode_target(target):
        if not joint_targets:
            return target
        img = tf.image.decode_image(target, channels=1, expand_animations=False)
        img = tf.image.resize(tf.cast(img, tf.float32), (image_size, image_size), method="nearest")
        img.set_shape((image_size, image_size, 1))
        return img

    if augment is None:
        ds = ds.map(lambda b, y: (_decode(b), _decode_target(y)), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.zip(
            (ds, tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True))
        )
        ds = ds.map(
            lambda by, s: (_decode(by[0]), _decode_target(by[1]), s), num_parallel_calls=tf.data.AUTOTUNE,
        )
        # Augmentation (e.g. keras_crop) may change the size; resize back.
        ds = augment(ds).map(lambda x, y: (_resize(x), _resize_target(y)), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    if training:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def cached_bytes(image_paths: list[str], targets, read_targets: bool = False) -> tf.data.Dataset:
    """In-memory cached dataset of ``(encoded file bytes, target)`` pairs.

    With READ_TARGETS the targets are file paths and their encoded bytes are cached too.
    """
    ds = tf.data.Dataset.from_tensor_slices((tf.constant(image_paths, dtype=tf.string), targets))
    if read_targets:
        ds = ds.map(lambda p, t: (tf.io.read_file(p), tf.io.read_file(t)), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda p, y: (tf.io.read_file(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache()


def _build_augment(transforms: list, seed: int, deterministic: bool, joint_targets: bool = False):
    """Return a dataset transform that augments ``(img, y, element_seed)``
    samples into ``(img, y)``, or None when TRANSFORMS has nothing to apply.

    Keras preprocessing layers run in a native tf.data.map — calling them
    inside tf.numpy_function strips graph context and causes internal shape
    errors (e.g. RandomTranslation rank mismatch). Their random transformation
    comes from a stateful generator, so a DETERMINISTIC run maps them
    sequentially: parallel calls would consume that state in a racy order.
    Custom aug_* functions are numpy-based and run in tf.numpy_function with a
    generator seeded from the element's own seed, so they stay parallel.
    """
    from cvbench.augmentations.pipeline import build_custom_aug_fn, build_joint_keras_aug_fn, build_keras_aug_fn

    if joint_targets:
        joint_aug = build_joint_keras_aug_fn(transforms, seed=seed if deterministic else None)
        keras_aug = None
    else:
        joint_aug = None
        keras_aug = build_keras_aug_fn(transforms, seed=seed if deterministic else None)
    custom_aug = build_custom_aug_fn(transforms)
    if keras_aug is None and joint_aug is None and custom_aug is None:
        return None

    def _custom(img, element_seed):
        # Dataset.random yields any int64; reinterpret as a non-negative seed.
        rng = np.random.default_rng(int(element_seed) & 0xFFFFFFFFFFFFFFFF)
        return custom_aug(img, rng)

    def _custom_map(x, y, s):
        x_aug = tf.numpy_function(_custom, [x, s], tf.float32)
        x_aug.set_shape(x.shape)
        return x_aug, y

    def augment(ds: tf.data.Dataset) -> tf.data.Dataset:
        if joint_aug is not None:
            ds = ds.map(
                lambda x, y, s: (*joint_aug(x, y, s), s),
                num_parallel_calls=None if deterministic else tf.data.AUTOTUNE,
            )
        if keras_aug is not None:
            ds = ds.map(
                lambda x, y, s: (keras_aug(x, s), y, s),
                num_parallel_calls=None if deterministic else tf.data.AUTOTUNE,
            )
        if custom_aug is not None:
            return ds.map(_custom_map, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.map(lambda x, y, s: (x, y))

    return augment
