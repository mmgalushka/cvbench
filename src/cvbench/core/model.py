from __future__ import annotations

import keras
import keras_hub

from cvbench.core.config import CVBenchConfig, LossConfig


# Map config backbone names to keras-hub preset identifiers
_BACKBONE_PRESETS = {
    "efficientnet_b0": "efficientnet_b0_ra_imagenet",
    "efficientnet_b1": "efficientnet_b1_ft_imagenet",
    "efficientnet_b2": "efficientnet_b2_ra_imagenet",
    "efficientnet_b3": "efficientnet_b3_ra2_imagenet",
    "efficientnet_b4": "efficientnet_b4_ra2_imagenet",
    "efficientnet_b5": "efficientnet_b5_sw_imagenet",
}


def _build_loss(loss_cfg: LossConfig) -> keras.losses.Loss:
    if loss_cfg.type == "focal":
        return keras.losses.CategoricalFocalCrossentropy(
            gamma=loss_cfg.focal_gamma,
            label_smoothing=loss_cfg.label_smoothing,
        )
    return keras.losses.CategoricalCrossentropy(
        label_smoothing=loss_cfg.label_smoothing,
    )


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

    if cfg.model.normalization == "internal":
        x = keras.layers.Rescaling(1.0 / 255.0)(x)

    # Backbone (frozen by default; fine-tune top layers if configured)
    # fine_tune_from_layer == 0  → fully frozen
    # fine_tune_from_layer == -1 → fully unfrozen
    # fine_tune_from_layer >  0  → layers[:N] frozen, rest trainable
    backbone = keras_hub.models.EfficientNetBackbone.from_preset(preset)
    ftl = cfg.model.fine_tune_from_layer
    if ftl == 0:
        backbone.trainable = False
    elif ftl == -1:
        backbone.trainable = True
    else:
        backbone.trainable = True
        for layer in backbone.layers[:ftl]:
            layer.trainable = False

    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(cfg.model.dropout)(x)
    outputs = keras.layers.Dense(cfg.model.num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.training.learning_rate),
        loss=_build_loss(cfg.training.loss),
        metrics=["accuracy"],
    )
    return model
