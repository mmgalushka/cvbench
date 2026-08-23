"""CenterNet-style detection model: pyramid backbone feature -> conv neck -> heads.

Graph: Input -> Rescaling -> EfficientNet backbone (tapped at a pyramid level,
default stride 4) -> conv neck -> three heads concatenated into one output
tensor: heatmap (num_classes, sigmoid), size (2, linear), offset (2, sigmoid).
A single conv output with no NMS op in the graph — decoding happens in Python
(see decode.py) — keeps the existing TFLite/ONNX/Hailo export path usable.
"""
from __future__ import annotations

import keras

from cvbench.core.backbone import build_optimizer, build_pyramid_stem
from cvbench.core.config import CVBenchConfig
from cvbench.detection.losses import CenterNetLoss


def build_model(cfg: CVBenchConfig) -> keras.Model:
    """Build and return a compiled CenterNet-style detection model.

    Args:
        cfg: Resolved experiment config.

    Returns:
        Compiled keras.Model ready for training. Output shape
        (batch, G, G, num_classes + 4) where G = input_size // grid_stride.
    """
    inputs, _backbone, feat = build_pyramid_stem(cfg, stride=cfg.detection.grid_stride)

    num_classes = cfg.model.num_classes

    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="neck_conv1")(feat)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="neck_conv2")(x)
    x = keras.layers.Dropout(cfg.model.dropout)(x)

    heatmap = keras.layers.Conv2D(num_classes, 1, activation="sigmoid", name="heatmap")(x)
    size = keras.layers.Conv2D(2, 1, name="size")(x)
    offset = keras.layers.Conv2D(2, 1, activation="sigmoid", name="offset")(x)

    outputs = keras.layers.Concatenate(axis=-1, name="predictions")([heatmap, size, offset])

    model = keras.Model(inputs=inputs, outputs=outputs, name="centernet")
    model.compile(
        optimizer=build_optimizer(cfg),
        loss=CenterNetLoss(num_classes=num_classes),
    )
    return model
