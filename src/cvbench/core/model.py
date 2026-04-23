from __future__ import annotations

import numpy as np
import keras
import keras_hub
import tensorflow as tf

from cvbench.core.config import CVBenchConfig, LossConfig


class LocalContrastNormalization(keras.layers.Layer):
    """Subtract local mean, divide by local std, then sigmoid-squash to (0, 1).

    Removes both additive offset (sensor brightness bias) and multiplicative
    scale (sensor sensitivity) while preserving local pattern structure.
    Output is passed through sigmoid so the backbone receives values in (0, 1).
    """

    def __init__(self, kernel_size: int = 32, epsilon: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        k = self.kernel_size
        sigma = k / 6.0
        coords = np.arange(k, dtype=np.float32) - k // 2
        g = np.exp(-0.5 * (coords / sigma) ** 2)
        kernel_2d = np.outer(g, g)
        kernel_2d /= kernel_2d.sum()
        kernel_4d = np.stack([kernel_2d] * channels, axis=2)[:, :, :, np.newaxis]
        self._kernel = self.add_weight(
            name="gaussian_kernel",
            shape=kernel_4d.shape,
            initializer=keras.initializers.Constant(kernel_4d),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        local_mean = tf.nn.depthwise_conv2d(
            x, self._kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        centered = x - local_mean
        local_std = tf.sqrt(
            tf.nn.depthwise_conv2d(
                tf.square(centered), self._kernel, strides=[1, 1, 1, 1], padding="SAME"
            )
        )
        return tf.sigmoid(centered / (local_std + self.epsilon))

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.kernel_size, "epsilon": self.epsilon}

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

    # Rescale pixels from [0, 255] to [0, 1] — EfficientNet backbone expects this range
    x = keras.layers.Rescaling(1.0 / 255.0)(x)

    if cfg.model.use_lcn:
        x = LocalContrastNormalization(
            kernel_size=cfg.model.lcn_kernel_size,
            epsilon=cfg.model.lcn_epsilon,
        )(x)

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
