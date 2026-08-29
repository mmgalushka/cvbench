from __future__ import annotations

import keras

from cvbench.core.backbone import build_backbone_stem, build_optimizer
from cvbench.core.config import CVBenchConfig, LossConfig


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
    inputs, backbone, x = build_backbone_stem(cfg)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(cfg.model.dropout)(x)
    outputs = keras.layers.Dense(cfg.model.num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=build_optimizer(cfg),
        loss=_build_loss(cfg.training.loss),
        metrics=["accuracy"],
    )
    return model
