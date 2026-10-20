"""Shared tf.data input pipeline: per-epoch shuffle, per-image augmentation, seeding, cache."""
import numpy as np
import pytest

pytestmark = pytest.mark.tf

from cvbench.core.config import TransformConfig  # noqa: E402

N = 32
SIZE = 16


@pytest.fixture
def images(tmp_path):
    """N flat-gray PNGs; image i's pixel value encodes its index."""
    from PIL import Image

    paths = []
    for i in range(N):
        p = tmp_path / f"{i:03d}.png"
        Image.fromarray(np.full((SIZE, SIZE, 3), 100 + i, dtype=np.uint8)).save(p)
        paths.append(str(p))
    return paths


def _t(name, prob=1.0, **params):
    return TransformConfig(name=name, prob=prob, params=params)


def _build(paths, **kwargs):
    from cvbench.core.input_pipeline import build_image_dataset

    kwargs.setdefault("batch_size", N)
    return build_image_dataset(paths, np.arange(len(paths), dtype=np.int64), image_size=SIZE, **kwargs)


def _epoch_orders(ds, epochs=2):
    return [labels.numpy().tolist() for _, labels in ds.take(epochs)]


def test_each_epoch_reshuffles_and_the_same_seed_replays_the_orders(images):
    first = _epoch_orders(_build(images, training=True, seed=7))

    assert first[0] != first[1]
    assert sorted(first[0]) == sorted(first[1]) == list(range(N))
    assert _epoch_orders(_build(images, training=True, seed=7)) == first
    assert _epoch_orders(_build(images, training=True, seed=8)) != first


def test_eval_dataset_keeps_file_order_and_runs_one_pass(images):
    batches = list(_build(images, batch_size=10))

    assert [len(labels) for _, labels in batches] == [10, 10, 10, 2]
    assert np.concatenate([labels for _, labels in batches]).tolist() == list(range(N))


def test_eval_dataset_ignores_transforms(images, monkeypatch):
    import cvbench.augmentations as aug_mod

    monkeypatch.setattr(aug_mod, "aug_zero", lambda img: np.zeros_like(img), raising=False)
    imgs, _ = next(iter(_build(images, transforms=[_t("aug_zero")])))

    assert np.all(imgs.numpy() > 0)


@pytest.fixture
def same_images(tmp_path):
    from PIL import Image

    p = tmp_path / "img.png"
    Image.fromarray(np.full((SIZE, SIZE, 3), 100, dtype=np.uint8)).save(p)
    return [str(p)] * 200


def test_custom_aug_prob_applies_per_image_within_one_batch(same_images, monkeypatch):
    import cvbench.augmentations as aug_mod

    monkeypatch.setattr(aug_mod, "aug_zero", lambda img: np.zeros_like(img), raising=False)
    ds = _build(same_images, batch_size=200, training=True, seed=1, transforms=[_t("aug_zero", prob=0.5)])

    imgs, _ = next(iter(ds))
    augmented = int(np.sum(np.all(imgs.numpy() == 0, axis=(1, 2, 3))))

    assert 60 < augmented < 140


def test_keras_aug_prob_applies_per_image_within_one_batch(same_images):
    ds = _build(
        same_images, batch_size=200, training=True, seed=1,
        transforms=[_t("keras_noise", prob=0.5, stddev=1.0)],
    )

    imgs, _ = next(iter(ds))
    augmented = int(np.sum(np.any(imgs.numpy() != 100.0, axis=(1, 2, 3))))

    assert 60 < augmented < 140


def test_same_seed_with_augmentation_gives_identical_first_batches(images):
    transforms = [
        _t("keras_rotation", prob=0.5, factor=0.2),
        _t("aug_salt_pepper", prob=0.7, density=[0.05, 0.2]),
    ]

    def first_batch(seed):
        imgs, labels = next(iter(_build(images, batch_size=8, training=True, seed=seed, transforms=transforms)))
        return imgs.numpy(), labels.numpy()

    a_imgs, a_labels = first_batch(3)
    b_imgs, b_labels = first_batch(3)
    c_imgs, _ = first_batch(4)

    assert np.array_equal(a_labels, b_labels)
    assert np.array_equal(a_imgs, b_imgs)
    assert not np.array_equal(a_imgs, c_imgs)


def test_cache_holds_encoded_bytes_not_decoded_images(images):
    import tensorflow as tf

    from cvbench.core.input_pipeline import cached_bytes

    ds = cached_bytes(images, np.arange(N))
    img_bytes, _ = next(iter(ds))

    assert ds.element_spec[0].dtype == tf.string
    assert len(img_bytes.numpy()) < SIZE * SIZE * 3 * 4  # smaller than the float32 image


def test_cached_dataset_serves_later_passes_from_memory(images):
    import os

    ds = _build(images, batch_size=8)
    first = [imgs.numpy() for imgs, _ in ds]
    for p in images:
        os.remove(p)

    second = [imgs.numpy() for imgs, _ in ds]  # would fail to read the files without the cache

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
