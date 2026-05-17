import random

import numpy as np
import keras

from cvbench.core.config import OneOfConfig

# Transforms that can erase or bury a weak signal when applied at full strength.
# Their effective probability is scaled by the per-image SNR factor so that
# low-SNR images are augmented less aggressively.
_DESTRUCTIVE_TRANSFORMS = {
    "aug_fog",
    "aug_blur",
    "aug_gamma",
    "aug_mask_h",
    "aug_mask_v",
    "aug_fade_horizontal",
    "aug_fade_vertical",
    "aug_random_profile_h",
    "aug_random_profile_v",
    "aug_interference",
    "aug_net",
}

# Which parameters control destructive intensity for each transform.
# When SNR is low, the upper bound of these parameters is compressed toward the
# lower bound: scaled_hi = lo + (hi - lo) * snr_factor.
# Transforms with no magnitude params rely solely on probability scaling.
_DESTRUCTIVE_MAGNITUDE_PARAMS: dict[str, list[str]] = {
    "aug_fog":             ["strength"],
    "aug_blur":            ["radius"],
    "aug_gamma":           [],
    "aug_mask_h":          ["max_width"],
    "aug_mask_v":          ["max_width"],
    "aug_fade_horizontal": ["strength"],
    "aug_fade_vertical":   ["strength"],
    "aug_random_profile_h":["max_delta"],
    "aug_random_profile_v":["max_delta"],
    "aug_interference":    [],
    "aug_net":             [],
}

# Below this SNR, all destructive transforms are skipped entirely (hard floor).
_ABSOLUTE_SNR_FLOOR = 0.05
# If post-augmentation SNR drops below snr_before * this ratio, blend back toward original.
_SNR_PRESERVATION_RATIO = 0.40
# Maximum fraction of the original image to blend back (never return fully unaugmented).
_MAX_BLEND_BACK = 0.80


def _compute_snr_factor(img: np.ndarray) -> float:
    """Return (95th pct - 30th pct) / 255, clamped to [0, 1].

    Approximates visible signal contrast. Strong signal → near 1.0,
    noise-floor-only image → near 0.0. For batches, returns the minimum
    across images so that any weak sample in the batch is protected.
    """
    if img.ndim == 4:
        factors = []
        for im in img:
            gray = im.mean(axis=-1) if im.ndim == 3 else im
            factors.append(float(np.percentile(gray, 95)) - float(np.percentile(gray, 30)))
        return float(np.clip(min(factors) / 255.0, 0.0, 1.0))
    gray = img.mean(axis=-1) if img.ndim == 3 else img
    return float(np.clip(
        (float(np.percentile(gray, 95)) - float(np.percentile(gray, 30))) / 255.0,
        0.0, 1.0,
    ))


_KERAS_MAP = {
    "keras_flip":        (keras.layers.RandomFlip,        {"mode": "horizontal"}),
    "keras_rotation":    (keras.layers.RandomRotation,    {"factor": 0.1}),
    "keras_zoom":        (keras.layers.RandomZoom,        {"height_factor": 0.1}),
    "keras_translation": (keras.layers.RandomTranslation, {"height_factor": 0.1, "width_factor": 0.1}),
    "keras_crop":        (keras.layers.RandomCrop,        {"height": 196, "width": 196}),
    "keras_brightness":  (keras.layers.RandomBrightness,  {"factor": 0.2}),
    "keras_contrast":    (keras.layers.RandomContrast,    {"factor": 0.2}),
    "keras_noise":       (keras.layers.GaussianNoise,     {"stddev": 0.05}),
}


def build_aug_pipeline(transforms: list) -> callable:
    """
    Build an augmentation pipeline from a list of TransformConfig / OneOfConfig objects.

    Returns apply(img: np.ndarray float32 [0,255]) -> np.ndarray float32 [0,255].
    Each step fires independently with its own prob; one_of groups pick one candidate.
    """
    steps = []
    for t in transforms:
        if isinstance(t, OneOfConfig):
            steps.append((_make_one_of_fn(t.candidates), t.prob))
        else:
            steps.append((_resolve(t.name, t.params), t.prob))

    def apply(img):
        for fn, prob in steps:
            if random.random() < prob:
                img = fn(img)
        return img

    return apply


def _make_one_of_fn(candidates: list):
    """Build a function that picks one candidate by weight and applies it."""
    fns_weights = [
        (_resolve(c.name, c.params, _DESTRUCTIVE_MAGNITUDE_PARAMS.get(c.name, [])), c.weight)
        for c in candidates
    ]
    total = sum(w for _, w in fns_weights)

    def one_of_fn(img, snr_factor=1.0):
        r = random.uniform(0, total)
        cumul = 0.0
        for fn, w in fns_weights:
            cumul += w
            if r <= cumul:
                return fn(img, snr_factor=snr_factor)
        return fns_weights[-1][0](img, snr_factor=snr_factor)

    return one_of_fn


def build_keras_aug_fn(transforms: list):
    """
    Build a tf-graph-compatible function from keras_* transforms only.

    Returns a callable suitable for use directly in tf.data.map (no numpy_function
    wrapper needed), or None if the list contains no keras_* transforms.
    Keras preprocessing layers must be applied this way — calling them inside
    tf.numpy_function strips graph context and causes internal shape errors.
    """
    import tensorflow as tf

    keras_steps = []
    for t in transforms:
        if isinstance(t, OneOfConfig):
            continue  # one_of groups contain only aug_* custom transforms
        if t.name in _KERAS_MAP:
            cls, defaults = _KERAS_MAP[t.name]
            merged = {**defaults, **t.params}
            layer = cls(**merged)
            keras_steps.append((layer, float(t.prob)))

    if not keras_steps:
        return None

    def apply(x):
        for layer, prob in keras_steps:
            x = tf.cond(
                tf.random.uniform(()) < prob,
                true_fn=lambda x=x: layer(x, training=True),
                false_fn=lambda x=x: x,
            )
        return x

    return apply


def build_custom_aug_fn(transforms: list):
    """
    Build a numpy-compatible function from aug_* transforms only.

    Returns apply(img: np.ndarray) -> np.ndarray, or None if the list contains
    no aug_* transforms.

    Destructive transforms have three layers of signal protection:
    1. Probability scaling by SNR (existing) — fires less often for weak signals.
    2. Magnitude scaling by SNR (new) — when it fires, strength is capped by SNR.
    3. Post-augmentation blend-back (new) — if cumulative SNR drop exceeds
       _SNR_PRESERVATION_RATIO, the output is blended back toward the original.
    """
    steps = []
    for t in transforms:
        if isinstance(t, OneOfConfig):
            aug_candidates = [c for c in t.candidates if c.name.startswith("aug_")]
            if aug_candidates:
                destructive = any(c.name in _DESTRUCTIVE_TRANSFORMS for c in aug_candidates)
                steps.append((_make_one_of_fn(aug_candidates), t.prob, destructive))
        elif t.name.startswith("aug_"):
            mag_params = _DESTRUCTIVE_MAGNITUDE_PARAMS.get(t.name, [])
            fn = _resolve(t.name, t.params, mag_params)
            steps.append((fn, t.prob, t.name in _DESTRUCTIVE_TRANSFORMS))

    if not steps:
        return None

    def apply(img):
        snr_before = _compute_snr_factor(img)
        original = img.copy()

        for fn, prob, destructive in steps:
            effective_prob = prob * snr_before if destructive else prob
            if random.random() < effective_prob:
                if destructive and snr_before < _ABSOLUTE_SNR_FLOOR:
                    continue  # hard floor: skip entirely for near-invisible signals
                img = fn(img, snr_factor=snr_before) if destructive else fn(img)

        # Blend-back: if cumulative augmentation degraded SNR too much, mix in original.
        snr_after = _compute_snr_factor(img)
        if snr_before > _ABSOLUTE_SNR_FLOOR:
            floor = snr_before * _SNR_PRESERVATION_RATIO
            if snr_after < floor:
                blend = min((floor - snr_after) / floor, _MAX_BLEND_BACK)
                img = img * (1.0 - blend) + original * blend

        return img

    return apply


def _resolve(name: str, params: dict, magnitude_params: list[str] | None = None) -> callable:
    if name in _KERAS_MAP:
        cls, defaults = _KERAS_MAP[name]
        merged = {**defaults, **params}
        layer = cls(**merged)

        def keras_fn(img, snr_factor=1.0, _layer=layer):  # noqa: ARG001
            if img.ndim == 3:
                return _layer(img[None], training=True)[0].numpy()
            return _layer(img, training=True).numpy()

        return keras_fn

    if name.startswith("aug_"):
        import cvbench.augmentations as aug_mod
        fn = getattr(aug_mod, name, None)
        if fn is None:
            available = [n for n in dir(aug_mod) if n.startswith("aug_")]
            raise ValueError(f"Unknown augmentation '{name}'. Available: {available}")

        mag_params = list(magnitude_params) if magnitude_params else []

        def _resolve_params(_params, snr_factor=1.0):
            resolved = {}
            for k, v in _params.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(e, str) for e in v):
                    resolved[k] = random.choice(v)
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    lo, hi = v
                    # bool must be checked before int — bool is a subclass of int
                    if isinstance(lo, bool) or isinstance(hi, bool):
                        resolved[k] = random.choice(v)
                    elif isinstance(lo, int) and isinstance(hi, int):
                        if k in mag_params and snr_factor < 1.0:
                            scaled_hi = int(round(lo + (hi - lo) * snr_factor))
                            scaled_hi = max(scaled_hi, lo)
                            resolved[k] = random.randint(lo, scaled_hi)
                        else:
                            resolved[k] = random.randint(lo, hi)
                    else:
                        if k in mag_params and snr_factor < 1.0:
                            scaled_hi = float(lo) + (float(hi) - float(lo)) * snr_factor
                            resolved[k] = random.uniform(float(lo), scaled_hi)
                        else:
                            resolved[k] = random.uniform(float(lo), float(hi))
                else:
                    resolved[k] = v
            return resolved

        def custom_fn(img, snr_factor=1.0, _fn=fn, _params=params):
            if img.ndim == 4:
                return np.stack([
                    _fn(im.astype(np.uint8), **_resolve_params(_params, snr_factor)).astype(np.float32)
                    for im in img
                ])
            return _fn(img.astype(np.uint8), **_resolve_params(_params, snr_factor)).astype(np.float32)

        return custom_fn

    known = list(_KERAS_MAP) + [n for n in dir(__import__("cvbench.augmentations", fromlist=["cvbench"])) if n.startswith("aug_")]
    raise ValueError(f"Unknown transform '{name}'. Known: {known}")
