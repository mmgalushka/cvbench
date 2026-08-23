"""Shared backbone construction — the part of the model every task reuses.

Everything past the backbone (head, loss, metrics) is task-specific and lives
in the task package (e.g. ``cvbench.classification.model``,
``cvbench.detection.model``).
"""
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

# Output stride -> keras-hub pyramid_outputs key, for tasks that need a
# spatial feature map (e.g. detection) instead of the deepest feature.
_PYRAMID_LEVEL_BY_STRIDE = {2: "P1", 4: "P2", 8: "P3", 16: "P4", 32: "P5"}


def build_optimizer(cfg: CVBenchConfig) -> keras.optimizers.Optimizer:
    opt = cfg.training.optimizer
    lr = cfg.training.learning_rate
    if opt.type == "sgd":
        return keras.optimizers.SGD(
            learning_rate=lr,
            weight_decay=opt.weight_decay,
            momentum=opt.momentum,
        )
    return keras.optimizers.Adam(
        learning_rate=lr,
        weight_decay=opt.weight_decay,
    )


def _apply_freeze(backbone: keras.Model, fine_tune_from_layer: int) -> None:
    # fine_tune_from_layer == 0  → fully frozen
    # fine_tune_from_layer == -1 → fully unfrozen
    # fine_tune_from_layer >  0  → layers[:N] frozen, rest trainable
    if fine_tune_from_layer == 0:
        backbone.trainable = False
    elif fine_tune_from_layer == -1:
        backbone.trainable = True
    else:
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_from_layer]:
            layer.trainable = False


def _load_backbone(cfg: CVBenchConfig, name: str | None = None) -> keras.Model:
    preset = _BACKBONE_PRESETS.get(cfg.model.backbone)
    if preset is None:
        raise ValueError(
            f"Unknown backbone '{cfg.model.backbone}'. "
            f"Valid options: {', '.join(_BACKBONE_PRESETS)}"
        )
    kwargs = {"load_weights": cfg.model.weights != "none"}
    if name is not None:
        kwargs["name"] = name
    backbone = keras_hub.models.EfficientNetBackbone.from_preset(preset, **kwargs)
    _apply_freeze(backbone, cfg.model.fine_tune_from_layer)
    return backbone


def build_backbone_stem(cfg: CVBenchConfig):
    """Build Input -> Rescaling -> EfficientNet backbone (deepest feature).

    For tasks that pool the whole feature map (e.g. classification).

    Args:
        cfg: Resolved experiment config.

    Returns:
        (inputs, backbone, x): the Input tensor, the backbone sub-model (with
        fine_tune_from_layer freeze semantics already applied, named
        "backbone" so ``model.get_layer("backbone")`` finds it later), and
        the output tensor of the backbone — the point a task-specific head
        attaches to.
    """
    size = cfg.model.input_size
    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)

    backbone = _load_backbone(cfg, name="backbone")
    x = backbone(x)
    return inputs, backbone, x


def build_pyramid_stem(cfg: CVBenchConfig, stride: int = 4):
    """Build Input -> Rescaling -> EfficientNet backbone, tapped at a pyramid
    level with the given output STRIDE, instead of the deepest feature.

    For tasks that need spatial resolution (e.g. detection).

    Args:
        cfg: Resolved experiment config.
        stride: output stride of the returned feature map (2/4/8/16/32).

    Returns:
        (inputs, feature_extractor, feat): the Input tensor, the pyramid
        feature-extraction sub-model (named "backbone" — this, not the raw
        keras-hub backbone object, is what actually appears in the built
        model's layer graph, so it is what ``model.get_layer("backbone")``
        finds), and the pyramid feature tensor a task-specific neck attaches to.
    """
    level = _PYRAMID_LEVEL_BY_STRIDE.get(stride)
    if level is None:
        raise ValueError(
            f"Unsupported stride {stride}. Valid options: {sorted(_PYRAMID_LEVEL_BY_STRIDE)}"
        )

    size = cfg.model.input_size
    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)

    # Freeze semantics are applied to the raw backbone's layers before they're
    # captured into feature_extractor's subgraph, below — trainable is a
    # mutable per-layer attribute, and feature_extractor shares those same
    # layer objects by reference, so the freeze is visible either way.
    raw_backbone = _load_backbone(cfg)
    if level not in raw_backbone.pyramid_outputs:
        raise ValueError(
            f"Backbone '{cfg.model.backbone}' has no pyramid level {level} "
            f"(stride {stride}). Available: {sorted(raw_backbone.pyramid_outputs)}"
        )
    feature_extractor = keras.Model(
        inputs=raw_backbone.input,
        outputs=raw_backbone.pyramid_outputs[level],
        name="backbone",
    )
    # feature_extractor's own container-level `.trainable` defaults to True
    # regardless of its (already-frozen) sublayers, since we only mutated the
    # child layers above. Mirror raw_backbone's container flag so a reader —
    # e.g. `model.get_layer("backbone").trainable` — sees the same answer it
    # would for the classification stem's backbone.
    feature_extractor.trainable = raw_backbone.trainable
    feat = feature_extractor(x)
    return inputs, feature_extractor, feat
