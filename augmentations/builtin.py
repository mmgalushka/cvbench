import keras

from core.registry import register_aug


@register_aug("light")
def build_light(params: dict) -> keras.Sequential:
    """Conservative augmentation — flips and mild rotation only."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(params.get("rotation", 0.05)),
    ], name="aug_light")


@register_aug("heavy")
def build_heavy(params: dict) -> keras.Sequential:
    """Aggressive augmentation — flips, rotation, zoom, contrast, translation."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(params.get("rotation", 0.2)),
        keras.layers.RandomZoom(params.get("zoom", 0.2)),
        keras.layers.RandomTranslation(
            height_factor=params.get("translation", 0.1),
            width_factor=params.get("translation", 0.1),
        ),
        keras.layers.RandomContrast(params.get("contrast", 0.2)),
    ], name="aug_heavy")


@register_aug("cutmix")
def build_cutmix(params: dict) -> keras.Sequential:
    """CutMix-style augmentation using random crop + resize as a Keras-layer proxy.

    Note: True CutMix (mixing two images) requires custom training loop logic.
    This preset applies spatial augmentations that approximate the effect within
    a single-image Keras layer pipeline.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(params.get("rotation", 0.1)),
        keras.layers.RandomZoom(params.get("zoom", 0.15)),
        keras.layers.RandomCrop(
            height=params.get("crop_size", 196),
            width=params.get("crop_size", 196),
        ),
        keras.layers.Resizing(
            height=params.get("output_size", 224),
            width=params.get("output_size", 224),
        ),
    ], name="aug_cutmix")
