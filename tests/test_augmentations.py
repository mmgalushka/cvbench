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
from cvbench.augmentations.pipeline import (
    build_aug_pipeline,
    build_custom_aug_fn,
    _compute_snr_factor,
    _ABSOLUTE_SNR_FLOOR,
    _SNR_PRESERVATION_RATIO,
)
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


def _weak_signal_image():
    """128×128 grayscale image with a faint 8-pixel stripe — SNR ≈ 0.07."""
    img = np.zeros((128, 128, 1), dtype=np.float32)
    img[60:68, :] = 20  # 20/255 brightness
    return img


class TestSignalPreservation:
    """Verify the three-layer signal-preservation system."""

    def _cfg(self, name, prob=1.0, **params):
        return TransformConfig(name=name, prob=prob, params=params)

    def test_magnitude_scaling_caps_fog_strength(self):
        # With a weak-signal image (SNR ~0.07), aug_fog must not wash out the signal stripe.
        # Run 20 times — magnitude scaling should keep every application gentle enough
        # that the stripe region stays above the absolute noise floor.
        img = _weak_signal_image()
        pipeline = build_custom_aug_fn([self._cfg("aug_fog", strength=[0.0, 0.8])])
        for _ in range(20):
            out = pipeline(img.copy())
            # stripe should not be completely whited out; its mean should stay below 255*0.9
            assert out[60:68].mean() < 255 * 0.9, "fog whited out the weak signal"

    def test_blend_back_preserves_snr_floor(self):
        # Force extreme fog (prob=1, full strength range) on a weak-signal image.
        # The blend-back mechanism should ensure post-augmentation SNR doesn't fall
        # below snr_before * _SNR_PRESERVATION_RATIO.
        img = _weak_signal_image()
        snr_before = _compute_snr_factor(img)
        pipeline = build_custom_aug_fn([self._cfg("aug_fog", strength=[0.9, 0.99])])
        out = pipeline(img.copy())
        snr_after = _compute_snr_factor(out)
        min_allowed = snr_before * _SNR_PRESERVATION_RATIO
        assert snr_after >= min_allowed * 0.95, (  # 5% tolerance for float rounding
            f"SNR dropped from {snr_before:.3f} to {snr_after:.3f}, "
            f"below floor {min_allowed:.3f}"
        )

    def test_hard_floor_skips_destructive_on_near_invisible(self):
        # Image with SNR < _ABSOLUTE_SNR_FLOOR — destructive transforms must be skipped.
        img = np.zeros((64, 64, 1), dtype=np.float32)  # pure black, SNR = 0
        assert _compute_snr_factor(img) < _ABSOLUTE_SNR_FLOOR

        original = img.copy()
        pipeline = build_custom_aug_fn([self._cfg("aug_fog", strength=[0.5, 0.9])])
        out = pipeline(img.copy())
        # fog on pure black produces pure white; hard floor must prevent this
        assert out.mean() < 10, "destructive transform fired below SNR hard floor"

    def test_strong_signal_unaffected_by_preservation(self):
        # A high-SNR image (dark background, bright stripe) should receive full-strength
        # augmentation and NOT be blended back toward the original.
        img = np.zeros((64, 64, 1), dtype=np.float32)
        img[20:44, :] = 220  # bright stripe — strong contrast, SNR well above floor
        snr = _compute_snr_factor(img)
        assert snr > 0.5

        pipeline = build_custom_aug_fn([self._cfg("aug_fog", strength=[0.3, 0.3])])
        out = pipeline(img.copy())
        # fog at strength=0.3: pixel = original * 0.7 + 255 * 0.3
        # stripe region expected ≈ 220 * 0.7 + 76.5 = 230.5
        expected = img * 0.7 + 255 * 0.3
        np.testing.assert_allclose(out, expected, atol=2.0)
