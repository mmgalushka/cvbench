import random

import numpy as np
import keras

from cvbench.core.config import OneOfConfig

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
    fns_weights = [(_resolve(c.name, c.params), c.weight) for c in candidates]
    total = sum(w for _, w in fns_weights)

    def one_of_fn(img):
        r = random.uniform(0, total)
        cumul = 0.0
        for fn, w in fns_weights:
            cumul += w
            if r <= cumul:
                return fn(img)
        return fns_weights[-1][0](img)

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
    """
    steps = []
    for t in transforms:
        if isinstance(t, OneOfConfig):
            aug_candidates = [c for c in t.candidates if c.name.startswith("aug_")]
            if aug_candidates:
                steps.append((_make_one_of_fn(aug_candidates), t.prob))
        elif t.name.startswith("aug_"):
            fn = _resolve(t.name, t.params)
            steps.append((fn, t.prob))

    if not steps:
        return None

    def apply(img):
        for fn, prob in steps:
            if random.random() < prob:
                img = fn(img)
        return img

    return apply


def _resolve(name: str, params: dict) -> callable:
    if name in _KERAS_MAP:
        cls, defaults = _KERAS_MAP[name]
        merged = {**defaults, **params}
        layer = cls(**merged)

        def keras_fn(img, _layer=layer):
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

        def _resolve_params(_params):
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
                        resolved[k] = random.randint(lo, hi)
                    else:
                        resolved[k] = random.uniform(float(lo), float(hi))
                else:
                    resolved[k] = v
            return resolved

        def custom_fn(img, _fn=fn, _params=params):
            if img.ndim == 4:
                return np.stack([
                    _fn(im.astype(np.uint8), **_resolve_params(_params)).astype(np.float32)
                    for im in img
                ])
            return _fn(img.astype(np.uint8), **_resolve_params(_params)).astype(np.float32)

        return custom_fn

    known = list(_KERAS_MAP) + [n for n in dir(__import__("cvbench.augmentations", fromlist=["cvbench"])) if n.startswith("aug_")]
    raise ValueError(f"Unknown transform '{name}'. Known: {known}")
