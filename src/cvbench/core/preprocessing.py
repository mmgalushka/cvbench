from __future__ import annotations

import numpy as np
import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="cvbench")
class LogNormalizeLayer(keras.layers.Layer):
    """Maps [0,1] pixel values through sqrt(x) → [0,1] with compressed dynamic range.

    Compresses bright regions so that large multiplicative intensity differences
    become smaller additive ones, which the subsequent Sobel layer then removes.
    Uses only SQRT (guaranteed Hailo-safe, no transcendental LUT required).
    """

    def call(self, x):
        return tf.math.sqrt(x)


@keras.saving.register_keras_serializable(package="cvbench")
class SobelGradientLayer(keras.layers.Layer):
    """Replaces pixel values with Sobel gradient features [Gx, Gy, magnitude].

    Input:  (B, H, W, 3) grayscale-as-RGB (R=G=B)
    Output: (B, H, W, 3) — Gx, Gy, sqrt(Gx²+Gy²)

    The two Conv2D filters are fixed (non-trainable). Gradient features are
    invariant to additive intensity shifts, so combined with LogNormalizeLayer
    the network is effectively intensity-invariant.
    """

    def build(self, input_shape):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        # kernel shape for Conv2D: (kH, kW, in_channels=1, out_channels=2)
        kernel = np.stack([sobel_x, sobel_y], axis=-1)  # (3, 3, 2)
        kernel = kernel[:, :, np.newaxis, :]  # (3, 3, 1, 2)

        self._conv = keras.layers.Conv2D(
            filters=2,
            kernel_size=3,
            padding="same",
            use_bias=False,
            trainable=False,
            kernel_initializer=keras.initializers.Constant(kernel),
            name="sobel_conv",
        )
        self._conv.build(input_shape[:-1] + (1,))
        super().build(input_shape)

    def call(self, x):
        gray = x[..., :1]  # (B, H, W, 1) — all 3 channels identical for grayscale
        grads = self._conv(gray)  # (B, H, W, 2)
        gx, gy = grads[..., :1], grads[..., 1:]
        mag = tf.sqrt(tf.square(gx) + tf.square(gy) + 1e-6)
        return tf.concat([gx, gy, mag], axis=-1)  # (B, H, W, 3)
