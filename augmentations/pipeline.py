import random

import numpy as np
import keras

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
    Build an augmentation pipeline from a list of TransformConfig objects.

    Returns apply(img: np.ndarray float32 [0,255]) -> np.ndarray float32 [0,255].
    Each step fires independently with its own prob.
    """
    steps = []
    for t in transforms:
        fn = _resolve(t.name, t.params)
        steps.append((fn, t.prob))

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
            t = _layer(img[None], training=True)
            return t[0].numpy()

        return keras_fn

    if name.startswith("aug_"):
        import augmentations as aug_mod
        fn = getattr(aug_mod, name, None)
        if fn is None:
            available = [n for n in dir(aug_mod) if n.startswith("aug_")]
            raise ValueError(f"Unknown augmentation '{name}'. Available: {available}")

        def custom_fn(img, _fn=fn, _params=params):
            return _fn(img.astype(np.uint8), **_params).astype(np.float32)

        return custom_fn

    known = list(_KERAS_MAP) + [n for n in dir(__import__("augmentations")) if n.startswith("aug_")]
    raise ValueError(f"Unknown transform '{name}'. Known: {known}")
