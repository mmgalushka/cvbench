from __future__ import annotations

import keras
import keras_hub

from cvbench.core.config import CVBenchConfig

# Map config backbone names to keras-hub preset identifiers
_BACKBONE_PRESETS = {
    "efficientnet_b0": "efficientnet_b0_ra_imagenet",
    "efficientnet_b1": "efficientnet_b1_ft_imagenet",
    "efficientnet_b2": "efficientnet_b2_ra_imagenet",
    "efficientnet_b3": "efficientnet_b3_ra2_imagenet",
    "efficientnet_b4": "efficientnet_b4_ra2_imagenet",
    "efficientnet_b5": "efficientnet_b5_sw_imagenet",
}


def build_model(cfg: CVBenchConfig) -> keras.Model:
    """Build and return a compiled Keras model.

    Args:
        cfg: Resolved experiment config.

    Returns:
        Compiled keras.Model ready for training.
    """
    preset = _BACKBONE_PRESETS.get(cfg.model.backbone)
    if preset is None:
        raise ValueError(
            f"Unknown backbone '{cfg.model.backbone}'. "
            f"Valid options: {', '.join(_BACKBONE_PRESETS)}"
        )

    size = cfg.model.input_size
    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = inputs

    # Rescale pixels from [0, 255] to [0, 1] — EfficientNet backbone expects this range
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # Backbone (frozen by default; fine-tune top layers if configured)
    backbone = keras_hub.models.EfficientNetBackbone.from_preset(preset)
    backbone.trainable = cfg.model.fine_tune_from_layer > 0
    if backbone.trainable:
        for layer in backbone.layers[: cfg.model.fine_tune_from_layer]:
            layer.trainable = False

    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(cfg.model.dropout)(x)
    outputs = keras.layers.Dense(cfg.model.num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
