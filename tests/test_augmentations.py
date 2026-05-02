"""Tests that every aug_* function handles grayscale (H,W) and colour (H,W,3) images."""
import numpy as np
import pytest

from cvbench.augmentations.blur import aug_blur
from cvbench.augmentations.transmission import aug_rf_transmission
from cvbench.augmentations.edges import (
    aug_brighten_edges_h,
    aug_brighten_edges_v,
    aug_fade_horizontal,
    aug_fade_vertical,
)
from cvbench.augmentations.lines import aug_lines_h, aug_lines_v
from cvbench.augmentations.noise import aug_salt_pepper
from cvbench.augmentations.pipeline import build_aug_pipeline
from cvbench.augmentations.profiles import aug_random_profile_h, aug_random_profile_v
from cvbench.augmentations.tone import aug_fog, aug_gamma
from cvbench.core.config import TransformConfig


# TF's image_dataset_from_directory always produces explicit channel dim:
# grayscale -> (H, W, 1), colour -> (H, W, 3); batches add B as the first axis.
GRAY = np.full((16, 16, 1), 128, dtype=np.uint8)
COLOR = np.full((16, 16, 3), 128, dtype=np.uint8)
BATCH_GRAY = np.full((4, 16, 16, 1), 128, dtype=np.float32)
BATCH_COLOR = np.full((4, 16, 16, 3), 128, dtype=np.float32)


@pytest.mark.parametrize("img", [GRAY, COLOR], ids=["gray", "color"])
class TestEachAugFunction:
    def test_blur(self, img):
        out = aug_blur(img, radius=1.0)
        assert out.shape == img.shape

    def test_fade_horizontal(self, img):
        for side in ("left", "right", "both"):
            out = aug_fade_horizontal(img, side=side)
            assert out.shape == img.shape

    def test_fade_vertical(self, img):
        for side in ("top", "bottom", "both"):
            out = aug_fade_vertical(img, side=side)
            assert out.shape == img.shape

    def test_brighten_edges_h(self, img):
        out = aug_brighten_edges_h(img)
        assert out.shape == img.shape

    def test_brighten_edges_v(self, img):
        out = aug_brighten_edges_v(img)
        assert out.shape == img.shape

    def test_lines_h(self, img):
        out = aug_lines_h(img, n_lines=2, seed=0)
        assert out.shape == img.shape

    def test_lines_v(self, img):
        out = aug_lines_v(img, n_lines=2, seed=0)
        assert out.shape == img.shape

    def test_salt_pepper(self, img):
        out = aug_salt_pepper(img, density=0.05, seed=0)
        assert out.shape == img.shape

    def test_random_profile_h(self, img):
        out = aug_random_profile_h(img, n_changes=3, seed=0)
        assert out.shape == img.shape

    def test_random_profile_v(self, img):
        out = aug_random_profile_v(img, n_changes=3, seed=0)
        assert out.shape == img.shape

    def test_gamma(self, img):
        out = aug_gamma(img, gamma=1.2)
        assert out.shape == img.shape

    def test_fog(self, img):
        out = aug_fog(img, strength=0.3)
        assert out.shape == img.shape

    def test_rf_transmission(self, img):
        out = aug_rf_transmission(img, seed=0)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_rf_transmission_modifies_image(self, img):
        # augmentation must change at least some pixels
        out = aug_rf_transmission(img, seed=1)
        assert not np.array_equal(out, img)

    def test_rf_transmission_no_op_on_tiny(self, img):
        # very narrow bandwidth on a small image — must not crash
        out = aug_rf_transmission(img, bandwidth=0.01, seed=2)
        assert out.shape == img.shape


@pytest.mark.parametrize("batch", [BATCH_GRAY, BATCH_COLOR], ids=["gray", "color"])
class TestPipelineBatch:
    """Pipeline receives batches (B, H, W) or (B, H, W, C) from tf.data."""

    def _t(self, name, **params):
        return TransformConfig(name=name, prob=1.0, params=params)

    def test_blur_batch(self, batch):
        fn = build_aug_pipeline([self._t("aug_blur", radius=1.0)])
        out = fn(batch)
        assert out.shape == batch.shape

    def test_salt_pepper_batch(self, batch):
        fn = build_aug_pipeline([self._t("aug_salt_pepper", density=0.05)])
        out = fn(batch)
        assert out.shape == batch.shape

    def test_random_profile_h_batch(self, batch):
        fn = build_aug_pipeline([self._t("aug_random_profile_h", n_changes=3, max_delta=20.0)])
        out = fn(batch)
        assert out.shape == batch.shape

    def test_random_profile_v_batch(self, batch):
        fn = build_aug_pipeline([self._t("aug_random_profile_v", n_changes=3, max_delta=20.0)])
        out = fn(batch)
        assert out.shape == batch.shape

    def test_fade_horizontal_batch(self, batch):
        fn = build_aug_pipeline([self._t("aug_fade_horizontal", side="both", strength=0.5)])
        out = fn(batch)
        assert out.shape == batch.shape

    def test_gamma_batch(self, batch):
        fn = build_aug_pipeline([self._t("aug_gamma", gamma=1.2)])
        out = fn(batch)
        assert out.shape == batch.shape

    def test_rf_transmission_batch(self, batch):
        fn = build_aug_pipeline([self._t("aug_rf_transmission", bandwidth=0.06)])
        out = fn(batch)
        assert out.shape == batch.shape
