"""Shared backbone construction — the part of the model every task reuses.

Everything past the backbone (head, loss, metrics) is task-specific and lives
in the task package (e.g. ``cvbench.classification.model``,
``cvbench.detection.model``).

This module is TensorFlow-free to import: ``keras`` / ``keras_hub`` are only
imported inside the functions that build a model, so the CLI can list and
validate backbone names without paying for them.
"""
from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from cvbench.core.errors import ConfigError

if TYPE_CHECKING:
    import keras

    from cvbench.core.config import CVBenchConfig


@dataclass(frozen=True)
class BackboneSpec:
    """How to build one named backbone: a keras-hub preset or a custom factory.

    Set either ``hub_class`` + ``preset``, or ``factory``.
    """

    name: str
    hub_class: str | None = None  # e.g. "keras_hub.models.ResNetBackbone"
    preset: str | None = None  # e.g. "resnet_18_imagenet"
    # "pkg.module:fn" -> fn(cfg, load_weights) -> keras.Model exposing .pyramid_outputs
    factory: str | None = None
    # Pure-ReLU, INT8-friendly graph (see docs/deployment/hailo.md). EfficientNet's
    # Swish/SiLU activations are known to quantize poorly.
    quant_friendly: bool = False


BACKBONES: dict[str, BackboneSpec] = {
    spec.name: spec
    for spec in (
        BackboneSpec("efficientnet_b0", "keras_hub.models.EfficientNetBackbone", "efficientnet_b0_ra_imagenet"),
        BackboneSpec("efficientnet_b1", "keras_hub.models.EfficientNetBackbone", "efficientnet_b1_ft_imagenet"),
        BackboneSpec("efficientnet_b2", "keras_hub.models.EfficientNetBackbone", "efficientnet_b2_ra_imagenet"),
        BackboneSpec("efficientnet_b3", "keras_hub.models.EfficientNetBackbone", "efficientnet_b3_ra2_imagenet"),
        BackboneSpec("efficientnet_b4", "keras_hub.models.EfficientNetBackbone", "efficientnet_b4_ra2_imagenet"),
        BackboneSpec("efficientnet_b5", "keras_hub.models.EfficientNetBackbone", "efficientnet_b5_sw_imagenet"),
        BackboneSpec("resnet_18", "keras_hub.models.ResNetBackbone", "resnet_18_imagenet", quant_friendly=True),
        BackboneSpec("resnet_50", "keras_hub.models.ResNetBackbone", "resnet_50_imagenet", quant_friendly=True),
    )
}


def register_backbone(spec: BackboneSpec) -> None:
    """Add SPEC to the registry so ``--backbone <spec.name>`` can select it.

    Raises ConfigError on a duplicate name or a spec that is neither a
    ``hub_class`` + ``preset`` pair nor a ``factory``. Register before the
    ``train`` CLI module is imported: its ``--backbone`` choices are read once.

    Example, a custom backbone built by ``my_pkg.backbones:build_mobilenet``::

        def build_mobilenet(cfg, load_weights):
            # must return a keras.Model with a ``.pyramid_outputs`` dict
            ...

        register_backbone(BackboneSpec(
            "mobilenet_v3", factory="my_pkg.backbones:build_mobilenet", quant_friendly=True,
        ))
    """
    if spec.name in BACKBONES:
        raise ConfigError(f"Backbone '{spec.name}' is already registered.")
    if spec.factory is None and not (spec.hub_class and spec.preset):
        raise ConfigError(
            f"Backbone '{spec.name}' needs either hub_class and preset, or a factory."
        )
    if spec.factory is not None and (spec.hub_class or spec.preset):
        raise ConfigError(
            f"Backbone '{spec.name}' sets a factory, so hub_class and preset must be empty."
        )
    BACKBONES[spec.name] = spec


BACKBONE_ENTRY_POINT_GROUP = "cvbench.backbones"
_plugins_loaded = False


def _load_plugin_backbones() -> None:
    """Register the backbones installed packages expose under the ``cvbench.backbones`` group.

    Runs once, on the first ``backbone_names()`` / ``_load_backbone`` call. An entry point's
    value is a ``BackboneSpec`` or a callable returning a list of them. A name that is
    already registered (a built-in one always is) wins: the plugin's is skipped with a warning.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group=BACKBONE_ENTRY_POINT_GROUP):
        loaded = ep.load()
        for spec in [loaded] if isinstance(loaded, BackboneSpec) else loaded():
            if spec.name in BACKBONES:
                warnings.warn(
                    f"Ignoring plugin backbone '{spec.name}' ({ep.value}): the name is already registered.",
                    stacklevel=2,
                )
                continue
            register_backbone(spec)


def backbone_names() -> list[str]:
    _load_plugin_backbones()
    return list(BACKBONES)


# Output stride -> keras-hub pyramid_outputs key, for tasks that need a
# spatial feature map (e.g. detection) instead of the deepest feature.
_PYRAMID_LEVEL_BY_STRIDE = {2: "P1", 4: "P2", 8: "P3", 16: "P4", 32: "P5"}

# The ImageNet per-channel statistics every keras-hub preset in the registry
# expects (its image converter's scale/offset are 1/(255*std) and -mean/std).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_freeze(backbone: keras.Model, fine_tune_from_layer: int) -> None:
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


def _import_attr(path: str, sep: str) -> Any:
    module_name, _, attr = path.rpartition(sep)
    return getattr(importlib.import_module(module_name), attr)


def _load_backbone(cfg: CVBenchConfig, name: str | None = None) -> keras.Model:
    _load_plugin_backbones()
    spec = BACKBONES.get(cfg.model.backbone)
    if spec is None:
        raise ConfigError(
            f"Unknown backbone '{cfg.model.backbone}'. "
            f"Valid options: {', '.join(BACKBONES)}"
        )
    load_weights = cfg.model.weights != "none"
    if spec.factory is not None:
        backbone = _import_attr(spec.factory, ":")(cfg, load_weights)
    else:
        assert spec.hub_class is not None
        kwargs: dict[str, Any] = {"load_weights": load_weights}
        if name is not None:
            kwargs["name"] = name
        backbone = _import_attr(spec.hub_class, ".").from_preset(spec.preset, **kwargs)
    apply_freeze(backbone, cfg.model.fine_tune_from_layer)
    return backbone


def build_stem(cfg: CVBenchConfig, strides: int | list[int] | None = None):
    """Build Input -> normalisation -> backbone.

    The input is float32 in [0, 255]; the per-channel ImageNet normalisation the
    keras-hub presets were trained with happens inside the model.

    Args:
        cfg: Resolved experiment config.
        strides: ``None`` taps the deepest feature (for tasks that pool the
            whole map, e.g. classification). Otherwise an output stride
            (2/4/8/16/32), or a list of them (e.g. ``[16, 32]`` for a 2-scale
            detection head), to tap pyramid levels of the same backbone in one
            pass — the backbone runs once however many levels are tapped.

    Returns:
        (inputs, backbone, features): the Input tensor, the backbone sub-model
        (fine_tune_from_layer freeze semantics already applied, named
        "backbone" so ``model.get_layer("backbone")`` finds it later), and the
        feature tensor(s) a task-specific head attaches to — a single tensor
        for ``None`` or a scalar stride, a list (same order) for a list.
    """
    import keras

    levels: list[str] = []
    if strides is not None:
        for s in [strides] if isinstance(strides, int) else strides:
            level = _PYRAMID_LEVEL_BY_STRIDE.get(s)
            if level is None:
                raise ConfigError(
                    f"Unsupported stride {s}. Valid options: {sorted(_PYRAMID_LEVEL_BY_STRIDE)}"
                )
            levels.append(level)

    size = cfg.model.input_size
    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = keras.layers.Rescaling(
        scale=[1.0 / (255.0 * s) for s in IMAGENET_STD],
        offset=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)],
    )(inputs)

    # Freeze semantics are applied to the raw backbone's layers before they're
    # captured into the wrapper's subgraph below — trainable is a mutable
    # per-layer attribute, and the wrapper shares those same layer objects by
    # reference, so the freeze is visible either way.
    raw = _load_backbone(cfg, name="backbone" if strides is None else None)
    if strides is None:
        if raw.name == "backbone":
            return inputs, raw, raw(x)
        outputs = raw.output
    else:
        missing = [lvl for lvl in levels if lvl not in raw.pyramid_outputs]
        if missing:
            raise ConfigError(
                f"Backbone '{cfg.model.backbone}' has no pyramid level(s) {missing}. "
                f"Available: {sorted(raw.pyramid_outputs)}"
            )
        outs = [raw.pyramid_outputs[lvl] for lvl in levels]
        outputs = outs if len(outs) > 1 else outs[0]
    # The wrapper is what appears in the built model's layer graph, so it is
    # what ``get_layer("backbone")`` finds. Its container-level `.trainable`
    # defaults to True regardless of its (already-frozen) sublayers, so mirror
    # the raw backbone's flag.
    extractor = keras.Model(inputs=raw.input, outputs=outputs, name="backbone")
    extractor.trainable = raw.trainable
    return inputs, extractor, extractor(x)
