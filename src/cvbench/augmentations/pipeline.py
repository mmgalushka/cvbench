import inspect
from collections.abc import Callable
from typing import Any

import keras
import numpy as np

from cvbench.core.config import OneOfConfig
from cvbench.core.errors import ConfigError

_KERAS_MAP: dict[str, tuple[type, dict[str, Any]]] = {
    "keras_flip":        (keras.layers.RandomFlip,        {"mode": "horizontal"}),
    "keras_rotation":    (keras.layers.RandomRotation,    {"factor": 0.1}),
    "keras_zoom":        (keras.layers.RandomZoom,        {"height_factor": 0.1}),
    "keras_translation": (keras.layers.RandomTranslation, {"height_factor": 0.1, "width_factor": 0.1}),
    "keras_crop":        (keras.layers.RandomCrop,        {"height": 196, "width": 196}),
    "keras_brightness":  (keras.layers.RandomBrightness,  {"factor": 0.2}),
    "keras_contrast":    (keras.layers.RandomContrast,    {"factor": 0.2}),
    "keras_noise":       (keras.layers.GaussianNoise,     {"stddev": 0.05}),
}

# Upper bound for seeds handed to aug_* functions (np.random.default_rng and
# random.Random both accept any non-negative int; keep it in uint32 range).
_SEED_BOUND = 2**32

Step = tuple[Callable[[np.ndarray, np.random.Generator], np.ndarray], float]


def build_aug_pipeline(transforms: list, seed: int | None = None) -> Callable:
    """
    Build an augmentation pipeline from a list of TransformConfig / OneOfConfig objects.

    Returns apply(img: np.ndarray float32 [0,255], HxWxC) -> np.ndarray float32 [0,255].
    Each step fires independently with its own prob; one_of groups pick one candidate.
    Every random decision draws from one generator seeded with SEED (a fresh
    seed when None), so the same SEED replays the same sequence of outputs.
    """
    steps: list[Step] = []
    for t in transforms:
        if isinstance(t, OneOfConfig):
            steps.append((_make_one_of_fn(t.candidates), t.prob))
        else:
            steps.append((_resolve(t.name, t.params), t.prob))

    rng = np.random.default_rng(seed)

    def apply(img):
        return _run_steps(steps, img, rng)

    return apply


def _run_steps(steps: list[Step], img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    for fn, prob in steps:
        if rng.random() < prob:
            img = fn(img, rng)
    return img


def _make_one_of_fn(candidates: list):
    """Build a function that picks one candidate by weight and applies it."""
    fns_weights = [(_resolve(c.name, c.params), c.weight) for c in candidates]
    total = sum(w for _, w in fns_weights)

    def one_of_fn(img, rng):
        r = rng.uniform(0, total)
        cumul = 0.0
        for fn, w in fns_weights:
            cumul += w
            if r <= cumul:
                return fn(img, rng)
        return fns_weights[-1][0](img, rng)

    return one_of_fn


def build_keras_aug_fn(transforms: list, seed: int | None = None):
    """
    Build a tf-graph-compatible per-image function from keras_* transforms only.

    Returns apply(img (H, W, C), element_seed int64 scalar) -> img, suitable for
    use directly in tf.data.map (no numpy_function wrapper needed), or None if
    the list contains no keras_* transforms. Each step's prob gate is a
    stateless draw from ELEMENT_SEED, so it is decided per image and replays
    under the same seed. SEED seeds the layers' own random transformations.
    Keras preprocessing layers must be applied this way — calling them inside
    tf.numpy_function strips graph context and causes internal shape errors.
    """
    import tensorflow as tf

    keras_steps: list[tuple[keras.layers.Layer, float]] = []
    for t in transforms:
        if isinstance(t, OneOfConfig):
            continue  # one_of groups contain only aug_* custom transforms
        if t.name in _KERAS_MAP:
            cls, defaults = _KERAS_MAP[t.name]
            merged = {**defaults, **t.params}
            if seed is not None:
                merged.setdefault("seed", seed + len(keras_steps))
            keras_steps.append((cls(**merged), float(t.prob)))

    if not keras_steps:
        return None

    def apply(x, element_seed):
        for i, (layer, prob) in enumerate(keras_steps):
            gate = tf.random.stateless_uniform((), seed=tf.stack([element_seed, tf.constant(i, tf.int64)]))
            x = tf.cond(
                gate < prob,
                true_fn=lambda x=x, layer=layer: layer(x[None], training=True)[0],
                false_fn=lambda x=x: x,
            )
        return x

    return apply


# keras_* transforms that move pixels: on a task with image-shaped targets (masks) they must
# move the target with the image; the other keras_* transforms and every aug_* only change colour.
GEOMETRIC_KERAS = frozenset({"keras_flip", "keras_rotation", "keras_zoom", "keras_translation", "keras_crop"})


def build_joint_keras_aug_fn(transforms: list, seed: int | None = None):
    """
    Like ``build_keras_aug_fn`` for samples whose target is an image-shaped array (a mask).

    Returns apply(x (H, W, C), y (H, W, K), element_seed) -> (x, y), or None when TRANSFORMS has
    no keras_* transforms. Geometric transforms run once on ``x`` and ``y`` stacked along the
    channel axis, so both get the same random flip/rotation/crop; nearest-neighbour
    interpolation keeps ``y`` free of in-between values. Photometric transforms touch only ``x``.
    """
    import tensorflow as tf

    steps: list[tuple[keras.layers.Layer, float, bool]] = []
    for t in transforms:
        if isinstance(t, OneOfConfig) or t.name not in _KERAS_MAP:
            continue
        cls, defaults = _KERAS_MAP[t.name]
        merged = {**defaults, **t.params}
        geometric = t.name in GEOMETRIC_KERAS
        if geometric and "interpolation" in inspect.signature(cls).parameters:
            merged["interpolation"] = "nearest"
        if seed is not None:
            merged.setdefault("seed", seed + len(steps))
        steps.append((cls(**merged), float(t.prob), geometric))

    if not steps:
        return None

    def apply(x, y, element_seed):
        channels = x.shape[-1]
        for i, (layer, prob, geometric) in enumerate(steps):
            gate = tf.random.stateless_uniform((), seed=tf.stack([element_seed, tf.constant(i, tf.int64)]))
            if geometric:
                def _both(x=x, y=y, layer=layer):
                    both = layer(tf.concat([x, y], axis=-1)[None], training=True)[0]
                    return both[..., :channels], both[..., channels:]

                x, y = tf.cond(gate < prob, true_fn=_both, false_fn=lambda x=x, y=y: (x, y))
            else:
                x = tf.cond(
                    gate < prob,
                    true_fn=lambda x=x, layer=layer: layer(x[None], training=True)[0],
                    false_fn=lambda x=x: x,
                )
        return x, y

    return apply


def build_custom_aug_fn(transforms: list):
    """
    Build a numpy-compatible per-image function from aug_* transforms only.

    Returns apply(img: np.ndarray (H, W, C), rng: np.random.Generator) -> np.ndarray,
    or None if the list contains no aug_* transforms. Every random decision —
    prob gates, one_of picks, sampled params and the aug_* functions' own
    seeds — draws from RNG.
    """
    steps: list[Step] = []
    for t in transforms:
        if isinstance(t, OneOfConfig):
            aug_candidates = [c for c in t.candidates if c.name.startswith("aug_")]
            if aug_candidates:
                steps.append((_make_one_of_fn(aug_candidates), t.prob))
        elif t.name.startswith("aug_"):
            steps.append((_resolve(t.name, t.params), t.prob))

    if not steps:
        return None

    def apply(img, rng):
        return _run_steps(steps, img, rng)

    return apply


def _sample_params(params: dict, rng: np.random.Generator) -> dict:
    """Draw one value for each ranged/choice param; pass fixed values through."""
    resolved = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(e, str) for e in v):
            resolved[k] = v[int(rng.integers(len(v)))]
        elif isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = v
            # bool must be checked before int — bool is a subclass of int
            if isinstance(lo, bool) or isinstance(hi, bool):
                resolved[k] = v[int(rng.integers(2))]
            elif isinstance(lo, int) and isinstance(hi, int):
                resolved[k] = int(rng.integers(lo, hi + 1))
            else:
                resolved[k] = float(rng.uniform(float(lo), float(hi)))
        else:
            resolved[k] = v
    return resolved


def _resolve(name: str, params: dict) -> Callable:
    if name in _KERAS_MAP:
        cls, defaults = _KERAS_MAP[name]
        merged = {**defaults, **params}
        layer = cls(**merged)

        def keras_fn(img, rng, _layer=layer):
            return _layer(img[None], training=True)[0].numpy()

        return keras_fn

    if name.startswith("aug_"):
        import cvbench.augmentations as aug_mod
        fn = getattr(aug_mod, name, None)
        if fn is None:
            available = [n for n in dir(aug_mod) if n.startswith("aug_")]
            raise ConfigError(f"Unknown augmentation '{name}'. Available: {available}")

        # An aug_* function with its own `seed` gets one drawn from the
        # pipeline's generator, so its internal randomness replays too.
        try:
            takes_seed = "seed" in inspect.signature(fn).parameters and "seed" not in params
        except (TypeError, ValueError):
            takes_seed = False

        def custom_fn(img, rng, _fn=fn, _params=params):
            kwargs = _sample_params(_params, rng)
            if takes_seed:
                kwargs["seed"] = int(rng.integers(_SEED_BOUND))
            return _fn(img.astype(np.uint8), **kwargs).astype(np.float32)

        return custom_fn

    aug_module = __import__("cvbench.augmentations", fromlist=["cvbench"])
    known = list(_KERAS_MAP) + [n for n in dir(aug_module) if n.startswith("aug_")]
    raise ConfigError(f"Unknown transform '{name}'. Known: {known}")
