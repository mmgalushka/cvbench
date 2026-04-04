import numpy as np
import pytest

from augmentations.pipeline import build_aug_pipeline
from core.config import TransformConfig


def _transform(name, prob=1.0, **params):
    return TransformConfig(name=name, prob=prob, params=params)


def test_unknown_name_raises():
    t = _transform("bad_name")
    with pytest.raises(ValueError, match="Unknown transform"):
        build_aug_pipeline([t])


def test_unknown_aug_raises_at_build_time():
    t = _transform("aug_does_not_exist")
    with pytest.raises(ValueError, match="Unknown augmentation"):
        build_aug_pipeline([t])


def test_custom_aug_applied(monkeypatch):
    import augmentations as aug_mod

    def fake_blur(img, radius=1.0):
        return (img * 0).astype(np.uint8)

    monkeypatch.setattr(aug_mod, "aug_blur", fake_blur)

    t = _transform("aug_blur", prob=1.0, radius=2.0)
    pipeline = build_aug_pipeline([t])

    img = np.full((4, 4), 128, dtype=np.float32)
    result = pipeline(img)
    assert result.max() == 0.0


def test_prob_zero_skips_transform(monkeypatch):
    import augmentations as aug_mod
    called = []

    def fake_blur(img, radius=1.0):
        called.append(True)
        return img

    monkeypatch.setattr(aug_mod, "aug_blur", fake_blur)

    t = _transform("aug_blur", prob=0.0, radius=1.0)
    pipeline = build_aug_pipeline([t])
    img = np.full((4, 4), 100, dtype=np.float32)
    pipeline(img)
    assert len(called) == 0


def test_prob_one_always_applies(monkeypatch):
    import augmentations as aug_mod
    called = []

    def fake_blur(img, radius=1.0):
        called.append(True)
        return img

    monkeypatch.setattr(aug_mod, "aug_blur", fake_blur)

    t = _transform("aug_blur", prob=1.0, radius=1.0)
    pipeline = build_aug_pipeline([t])
    img = np.full((4, 4), 100, dtype=np.float32)
    for _ in range(20):
        pipeline(img)
    assert len(called) == 20


def test_empty_transforms_is_identity():
    pipeline = build_aug_pipeline([])
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = pipeline(img)
    np.testing.assert_array_equal(result, img)
