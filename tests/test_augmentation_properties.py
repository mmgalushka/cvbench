"""Properties every ``aug_*`` function must keep: shape, dtype, range and seed determinism.

Plus focused behaviour tests for the augmentations that had none (net, chirp, interference, mask).
"""
import inspect

import numpy as np
import pytest

import cvbench.augmentations as aug_mod
from cvbench.augmentations.chirp import aug_chirp_artifacts
from cvbench.augmentations.interference import aug_interference
from cvbench.augmentations.mask import aug_mask, aug_mask_h, aug_mask_v
from cvbench.augmentations.net import _line_mask, _stripe_mask, aug_net
from cvbench.core.errors import ConfigError

H, W = 64, 80


def _image(channels: int | None = 3, dtype=np.uint8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (H, W) if channels is None else (H, W, channels)
    img = rng.integers(0, 256, shape)
    return img.astype(dtype)


SEEDED = sorted(
    name for name in dir(aug_mod)
    if name.startswith("aug_") and "seed" in inspect.signature(getattr(aug_mod, name)).parameters
)

# Augmentations whose output actually varies with the seed (the rest are deterministic given their args).
RANDOM_BY_SEED = [
    "aug_chirp_artifacts", "aug_interference", "aug_mask", "aug_mask_h", "aug_mask_v",
    "aug_random_profile_h", "aug_random_profile_v", "aug_salt_pepper",
]


# Arguments a function has no default for.
REQUIRED = {"aug_salt_pepper": {"density": 0.05}}


def _apply(name: str, img: np.ndarray, **kwargs) -> np.ndarray:
    return getattr(aug_mod, name)(img, **REQUIRED.get(name, {}), **kwargs)


def test_every_seeded_augmentation_is_covered():
    assert set(RANDOM_BY_SEED) <= set(SEEDED)
    assert {"aug_net", "aug_chirp_artifacts", "aug_interference", "aug_mask"} <= set(SEEDED)


@pytest.mark.parametrize("name", SEEDED)
@pytest.mark.parametrize("channels", [3, 1, None], ids=["rgb", "single-channel", "2d"])
def test_shape_dtype_and_range_are_preserved(name, channels):
    img = _image(channels)

    out = _apply(name, img, seed=7)

    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert out.min() >= 0 and out.max() <= 255


@pytest.mark.parametrize("name", SEEDED)
def test_same_seed_gives_the_same_output(name):
    img = _image()

    first = _apply(name, img.copy(), seed=3)
    # Unrelated global RNG draws between the calls must not leak into a seeded call.
    np.random.rand(5)
    assert np.array_equal(_apply(name, img.copy(), seed=3), first)


@pytest.mark.parametrize("name", RANDOM_BY_SEED)
def test_different_seeds_give_different_outputs(name):
    img = _image()

    outputs = {_apply(name, img.copy(), seed=s).tobytes() for s in range(6)}

    assert len(outputs) > 1


@pytest.mark.parametrize("name", SEEDED)
def test_input_is_never_modified_in_place(name):
    img = _image()
    snapshot = img.copy()

    _apply(name, img, seed=1)

    assert np.array_equal(img, snapshot)


# ---------------------------------------------------------------------------
# net
# ---------------------------------------------------------------------------

def test_net_draws_lines_at_the_requested_brightness():
    img = np.zeros((H, W, 3), dtype=np.uint8)

    out = aug_net(img, mode="positive", angle=45, num_lines=4, line_brightness=200)

    assert set(np.unique(out)) == {0, 200}
    assert (out == 200).any()


def test_net_modes_differ_and_both_is_their_union():
    img = np.zeros((H, W), dtype=np.uint8)
    kwargs = dict(angle=30, num_lines=5, line_brightness=255)

    pos, neg, both = (aug_net(img, mode=m, **kwargs) > 0 for m in ("positive", "negative", "both"))

    assert not np.array_equal(pos, neg)
    assert np.array_equal(both, pos | neg)


def test_net_unknown_mode_falls_back_to_both():
    img = np.zeros((H, W), dtype=np.uint8)

    assert np.array_equal(aug_net(img, mode="sideways", num_lines=3), aug_net(img, mode="both", num_lines=3))


def test_net_angle_is_clipped_to_the_valid_range():
    img = np.zeros((H, W), dtype=np.uint8)

    assert np.array_equal(aug_net(img, mode="positive", angle=0, num_lines=3),
                          aug_net(img, mode="positive", angle=1, num_lines=3))
    assert np.array_equal(aug_net(img, mode="positive", angle=200, num_lines=3),
                          aug_net(img, mode="positive", angle=89, num_lines=3))


@pytest.mark.parametrize("stripe", ["horizontal", "vertical"])
def test_net_stripe_confines_the_lines_to_the_bands(stripe):
    img = np.zeros((H, W), dtype=np.uint8)
    band = _stripe_mask(stripe, 4, 50.0, H, W)

    out = aug_net(img, mode="both", num_lines=30, stripe=stripe, num_stripes=4, stripe_fill=50.0)

    assert (out > 0).any()
    assert not (out > 0)[~band].any()


@pytest.mark.parametrize("orientation, axis", [("horizontal", 0), ("vertical", 1)])
def test_stripe_mask_fill_fraction(orientation, axis):
    mask = _stripe_mask(orientation, 4, 50.0, 64, 64)

    assert mask.mean() == pytest.approx(0.5, abs=0.05)
    assert mask.shape == (64, 64)
    assert (mask.all(axis=1 - axis) | ~mask.any(axis=1 - axis)).all()  # whole rows/columns on or off


def test_line_mask_covers_more_with_wider_lines():
    thin, thick = _line_mask(30, 5, 1, H, W), _line_mask(30, 5, 5, H, W)

    assert thin.dtype == bool and thick.sum() > thin.sum() > 0


# ---------------------------------------------------------------------------
# chirp
# ---------------------------------------------------------------------------

def test_chirp_only_adds_light_screen_blend_never_darkens():
    img = _image()

    out = aug_chirp_artifacts(img, num_chirps=8, seed=2)

    assert (out >= img).all() and (out > img).any()


def test_chirp_float_input_stays_float_in_unit_range():
    img = _image(dtype=np.float32) / 255.0

    out = aug_chirp_artifacts(img, seed=4)

    assert out.dtype == np.float32 and out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_chirp_preserves_the_alpha_channel():
    img = _image(channels=4)

    out = aug_chirp_artifacts(img, seed=5)

    assert out.shape == img.shape
    assert np.array_equal(out[:, :, 3], img[:, :, 3])


@pytest.mark.parametrize("justification", ["left", "right", "center", "random"])
def test_chirp_justification_modes_run(justification):
    out = aug_chirp_artifacts(_image(), justification=justification, num_anchors=2, seed=1)

    assert out.shape == (H, W, 3)


def test_chirp_rejects_an_unknown_justification():
    with pytest.raises(ConfigError, match="justification must be one of"):
        aug_chirp_artifacts(_image(), justification="diagonal")


def test_chirp_without_vertical_clustering_and_with_swapped_bounds():
    out = aug_chirp_artifacts(
        _image(), cluster_vertically=False, min_length=60, max_length=20, min_thickness=5, max_thickness=1, seed=1
    )

    assert out.shape == (H, W, 3)


def test_chirp_with_zero_chirps_is_the_identity():
    img = _image()

    assert np.array_equal(aug_chirp_artifacts(img, num_chirps=0, seed=1), img)


# ---------------------------------------------------------------------------
# interference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", ["scanline", "stripes", "turbulent", "flow", "random"])
def test_interference_patterns_change_the_image(pattern):
    img = _image()

    out = aug_interference(img, pattern=pattern, alpha_min=0.5, alpha_max=0.5, seed=1)

    assert out.shape == img.shape and out.dtype == img.dtype
    assert not np.array_equal(out, img)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical", "both", "random"])
def test_interference_scanline_orientations(orientation):
    out = aug_interference(_image(), pattern="scanline", orientation=orientation, seed=1)

    assert out.shape == (H, W, 3)


def test_interference_zero_alpha_is_the_identity():
    img = _image()

    assert np.array_equal(aug_interference(img, alpha_min=0.0, alpha_max=0.0, seed=1), img)


def test_interference_strength_grows_with_alpha():
    img = _image().astype(np.float64)

    weak = aug_interference(_image(), pattern="stripes", alpha_min=0.1, alpha_max=0.1, seed=3)
    strong = aug_interference(_image(), pattern="stripes", alpha_min=0.9, alpha_max=0.9, seed=3)

    assert np.abs(strong - img).mean() > np.abs(weak - img).mean()


def test_interference_rejects_an_unknown_pattern():
    with pytest.raises(ConfigError, match="Unknown pattern"):
        aug_interference(_image(), pattern="plaid")


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------

def test_mask_h_blanks_whole_rows():
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    out = aug_mask_h(img, n_masks=1, max_width=10, fill_value=7, seed=1)

    blank_rows = np.where((out == 7).all(axis=(1, 2)))[0]
    assert 1 <= len(blank_rows) <= 10
    assert (out[blank_rows[0]:blank_rows[-1] + 1] == 7).all()  # one contiguous band
    assert (out[np.setdiff1d(np.arange(H), blank_rows)] == 255).all()


def test_mask_v_blanks_whole_columns():
    img = np.full((H, W), 255, dtype=np.uint8)

    out = aug_mask_v(img, n_masks=2, max_width=6, fill_value=0, seed=1)

    blank_cols = np.where((out == 0).all(axis=0))[0]
    assert 1 <= len(blank_cols) <= 12
    assert (out[:, np.setdiff1d(np.arange(W), blank_cols)] == 255).all()


def test_mask_zero_masks_is_the_identity():
    img = _image()

    assert np.array_equal(aug_mask(img, n_masks=0, seed=1), img)


def test_mask_band_width_is_bounded():
    img = np.full((H, W), 255, dtype=np.uint8)

    for seed in range(20):
        out = aug_mask(img, n_masks=1, max_width=5, orientation="h", seed=seed)
        assert 1 <= (out == 0).all(axis=1).sum() <= 5
