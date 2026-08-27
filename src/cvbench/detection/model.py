"""YOLO-style detection model: 2-scale pyramid backbone features -> FPN-lite
neck -> per-scale anchor head.

Graph: Input -> Rescaling -> backbone (tapped at strides 16 and 32) -> a
coarse (stride-32) conv branch feeding its own head, upsampled and fused with
the stride-16 features for a second head. Each head emits one raw-logit
tensor of shape ``(G, G, A*(5+C))`` — ``tx, ty, tw, th, obj`` then C class
logits per anchor, no sigmoid/exp applied — so decoding (and the greedy NMS
it needs) stays entirely in Python (see ``decode.py``), keeping the
TFLite/ONNX/Hailo export path clean of any NMS/activation ops.

Assumes ``cfg.detection.strides == [s, 2*s]`` (a single ``UpSampling2D(2)``
fuses the coarse branch into the fine one) — the shipped default is
``[16, 32]``.
"""
from __future__ import annotations

import keras

from cvbench.core.backbone import build_optimizer, build_pyramid_stem
from cvbench.core.config import CVBenchConfig
from cvbench.detection.losses import YoloLoss


def build_model(cfg: CVBenchConfig) -> keras.Model:
    """Build and return a compiled 2-scale YOLO-style detection model.

    Args:
        cfg: Resolved experiment config — ``cfg.detection.anchors`` must
            already be resolved (see ``detection/anchors.py::resolve_anchors``).

    Returns:
        Compiled keras.Model with two outputs, ``[head16, head32]``, shapes
        ``(batch, G16, G16, A16*(5+C))`` and ``(batch, G32, G32, A32*(5+C))``.
    """
    stride_fine, stride_coarse = cfg.detection.strides
    inputs, _backbone, (feat_fine, feat_coarse) = build_pyramid_stem(
        cfg, stride=[stride_fine, stride_coarse]
    )

    num_classes = cfg.model.num_classes
    anchors_fine, anchors_coarse = cfg.detection.anchors
    a_fine, a_coarse = len(anchors_fine), len(anchors_coarse)

    # Coarse (stride-32) branch — smallest feature map, largest objects.
    x_coarse = keras.layers.Conv2D(
        256, 3, padding="same", activation="relu", name="neck_coarse_conv"
    )(feat_coarse)
    x_coarse = keras.layers.Dropout(cfg.model.dropout)(x_coarse)
    head_coarse = keras.layers.Conv2D(
        a_coarse * (5 + num_classes), 1, name="head_coarse"
    )(x_coarse)

    # Fine (stride-16) branch — fuses upsampled coarse features (FPN-lite)
    # with the backbone's own stride-16 features, for small-object detail.
    up = keras.layers.Conv2D(
        128, 1, padding="same", activation="relu", name="neck_fine_reduce"
    )(x_coarse)
    up = keras.layers.UpSampling2D(2, name="neck_fine_upsample")(up)
    x_fine = keras.layers.Concatenate(name="neck_fine_concat")([up, feat_fine])
    x_fine = keras.layers.Conv2D(
        256, 3, padding="same", activation="relu", name="neck_fine_conv"
    )(x_fine)
    x_fine = keras.layers.Dropout(cfg.model.dropout)(x_fine)
    head_fine = keras.layers.Conv2D(
        a_fine * (5 + num_classes), 1, name="head_fine"
    )(x_fine)

    model = keras.Model(inputs=inputs, outputs=[head_fine, head_coarse], name="yolo")
    model.compile(
        optimizer=build_optimizer(cfg),
        loss=[
            YoloLoss(
                num_classes=num_classes, num_anchors=a_fine,
                noobj_weight=cfg.detection.noobj_weight,
            ),
            YoloLoss(
                num_classes=num_classes, num_anchors=a_coarse,
                noobj_weight=cfg.detection.noobj_weight,
            ),
        ],
    )
    return model
