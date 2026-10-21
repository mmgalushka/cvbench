import dataclasses
import json
import os
from collections.abc import Callable
from typing import Any

import click

from cvbench.cli import _help
from cvbench.core.backbone import backbone_names
from cvbench.core.config import (
    FLAG_KEYS,
    KEY_FLAGS,
    DataConfig,
    LossConfig,
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    build_config,
    resolve_key,
)
from cvbench.core.errors import CVBenchError, NotFoundError

_DATA = DataConfig()
_MODEL = ModelConfig()
_TRAIN = TrainingConfig()

# NOTE: cvbench.services.training is imported inside train() — it pulls in
# TensorFlow, and importing it at module scope would make `train --help` (and
# the `commands` overview, which enumerates every command) take ~1.5s to start.


# Click type of every flag `train` and `sweep` share, so a value is accepted or rejected
# identically by both (sweep converts each comma-separated item through the same type).
FLAG_TYPES: dict[str, click.ParamType] = {
    "backbone": click.Choice(backbone_names()),
    "weights": click.Choice(["imagenet", "none"]),
    "epochs": click.IntRange(min=1),
    "lr": click.FloatRange(min=0, min_open=True),
    "batch_size": click.IntRange(min=1),
    "input_size": click.IntRange(min=1),
    "dropout": click.FloatRange(min=0, max=1, max_open=True),
    "fine_tune_from_layer": click.IntRange(min=-1),
    "val_split": click.FloatRange(min=0, max=1, min_open=True, max_open=True),
    "seed": click.IntRange(min=0),
}


def _parse_class_weight(value: str | None):
    """Parse --class-weight CLI value: null → None, auto → 'auto', JSON → dict."""
    if value is None or value.lower() in ("null", "none"):
        return None
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and all(
        isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0 for w in parsed.values()
    ):
        return parsed
    raise click.BadParameter(
        f"Expected null, auto, or a JSON dict of positive weights like "
        f"'{{\"cat\": 1.0, \"dog\": 2.5}}', got: {value!r}",
        param_hint="--class-weight",
    )


def _parse_params(
    raw: str, spec: dict[str, click.ParamType], hint: str, example: str,
) -> dict[str, Any]:
    """Parse `key=value,key=value` into typed values; every failure is a BadParameter with an example."""
    params: dict[str, Any] = {}
    for item in raw.split(","):
        key, sep, val = item.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key or not val:
            raise click.BadParameter(
                f"expected key=value, got {item.strip()!r} (e.g. {example})", param_hint=hint
            )
        if key not in spec:
            raise click.BadParameter(
                f"unknown parameter {key!r}; valid: {', '.join(spec)} (e.g. {example})", param_hint=hint
            )
        try:
            params[key] = spec[key].convert(val, None, None)
        except click.BadParameter as e:
            raise click.BadParameter(f"{key}: {e.message}", param_hint=hint) from None
    return params


def _split_typed(
    value: str, known: tuple[str, ...], hint: str, example: str,
) -> tuple[str, str]:
    """Split `type` / `type:params` and check the type name; returns (type, params_str)."""
    kind, _, params_str = value.partition(":")
    kind = kind.strip()
    if kind not in known:
        raise click.BadParameter(
            f"unknown type {kind!r}; valid: {', '.join(known)} (e.g. {example})", param_hint=hint
        )
    if ":" in value and not params_str.strip():
        raise click.BadParameter(f"nothing after ':' (e.g. {example})", param_hint=hint)
    return kind, params_str


def _parse_optimizer(value: str | None) -> OptimizerConfig | None:
    if value is None:
        return None
    example = "sgd:momentum=0.9,weight_decay=1e-4"
    kind, params_str = _split_typed(value, ("adam", "sgd"), "--optimizer", example)
    params = _parse_params(params_str, {
        "weight_decay": click.FloatRange(min=0),
        "momentum": click.FloatRange(min=0, max=1),
    }, "--optimizer", example) if params_str else {}
    return OptimizerConfig(
        type=kind,
        weight_decay=params.get("weight_decay", 0.0),
        momentum=params.get("momentum", 0.9),
    )


def _parse_lr_scheduler(value: str | None) -> LRSchedulerConfig | None:
    if value is None:
        return None
    params = _parse_params(value, {
        "patience": click.IntRange(min=0),
        "factor": click.FloatRange(min=0, max=1, min_open=True, max_open=True),
        "min": click.FloatRange(min=0),
    }, "--lr-scheduler", "patience=5,factor=0.5,min=1e-7")
    return LRSchedulerConfig(
        patience=params.get("patience", 5),
        factor=params.get("factor", 0.5),
        min_lr=params.get("min", 1e-7),
    )


def _parse_loss(value: str | None) -> LossConfig | None:
    if value is None:
        return None
    example = "focal:gamma=2.0,label_smoothing=0.1"
    kind, params_str = _split_typed(value, ("crossentropy", "focal"), "--loss", example)
    params = _parse_params(params_str, {
        "label_smoothing": click.FloatRange(min=0, max=1),
        "gamma": click.FloatRange(min=0),
    }, "--loss", example) if params_str else {}
    return LossConfig(
        type=kind,
        label_smoothing=params.get("label_smoothing", 0.0),
        focal_gamma=params.get("gamma", 2.0),
    )


# Flags whose text is a `type:k=v` spec naming a whole section; each expands to one override per field.
_SPEC_PARSERS: dict[str, Callable[[str], Any]] = {
    "training.optimizer": _parse_optimizer,
    "training.loss": _parse_loss,
    "training.lr_scheduler": _parse_lr_scheduler,
}


def parse_override(key: str, raw: Any) -> dict[str, Any]:
    """Command-line text for the canonical config ``key`` -> dotted overrides for ``apply_overrides``.

    Keys with a flag are checked like that flag (ranges, choices, specs); any other key
    passes through and is type-checked against its dataclass field later.
    """
    hint = f"--{KEY_FLAGS[key].replace('_', '-')}" if key in KEY_FLAGS else "--set"
    flag = KEY_FLAGS.get(key)
    if key in _SPEC_PARSERS:
        section = _SPEC_PARSERS[key](raw)
        return {f"{key}.{f.name}": getattr(section, f.name) for f in dataclasses.fields(section)}
    if key == "training.class_weight" and isinstance(raw, str):
        return {key: _parse_class_weight(raw)}
    if key == "augmentation" and not os.path.isfile(raw):
        raise click.BadParameter(f"augmentation file not found: {raw!r}", param_hint=hint)
    if flag in FLAG_TYPES and isinstance(raw, str):
        try:
            raw = FLAG_TYPES[flag].convert(raw, None, None)
        except click.BadParameter as e:
            raise click.BadParameter(e.message, param_hint=hint) from None
    return {key: raw}


def parse_set_option(item: str) -> dict[str, Any]:
    """One ``--set key=value`` -> overrides; ``key`` may be a dotted config key or a flag name."""
    key, sep, raw = item.partition("=")
    if not sep or not key.strip():
        raise click.BadParameter(f"expected key=value, got {item!r} (e.g. training.epochs=5)", param_hint="--set")
    try:
        key = resolve_key(key.strip())
    except CVBenchError as e:
        raise click.BadParameter(e.message, param_hint="--set") from None
    return parse_override(key, raw.strip())


def validate_dataset(data_dir: str, from_dir: str | None = None) -> None:
    """Fail with a usage error (exit 2) if DATA_DIR cannot be trained on.

    Runs before TensorFlow is imported or anything is written. A `--from` baseline
    keeps the task recorded in its config instead of re-sniffing the layout, as
    `run_training` does.
    """
    from cvbench.core.config import load_config
    from cvbench.tasks import layout_errors

    task = None
    if from_dir is not None:
        try:
            task = load_config(from_dir).task
        except NotFoundError as e:
            raise click.BadParameter(e.message, param_hint="--from") from None
    errors = layout_errors(data_dir, task)
    if errors:
        raise click.BadParameter("\n".join(errors), param_hint="DATA_DIR")


@_help.command(
    examples=[
        ("train data/synthetic --epochs 5",
         "Smoke-test the whole pipeline on the synthetic dataset"),
        ("train data/my_dataset --epochs 20 --backbone efficientnet_b0",
         "Train on your own dataset (needs a train/ subfolder; val/ and test/ optional)"),
        ("train data/my_dataset --augmentation workspace/augmentation.yaml --loss focal:gamma=2.0",
         "Add a saved augmentation config (see 'data aug') and swap the loss function"),
    ],
    see_also=[
        ("runs list", "find the name of the run you just created"),
        ("evaluate <run>", "score it on the held-out test split"),
        ("runs export <run> --format tflite", "package it for a device"),
    ],
)
@click.argument("data_dir")
@click.option("--output", "output_dir", default=None,
              help="Run output directory; its last component is the run name "
                   "(default: experiments/<auto-name>/).")
@click.option("--overwrite", is_flag=True, default=False,
              help="Replace the --output run if it already exists (default: refuse).")
@click.option("--from", "from_dir", default=None, type=click.Path(exists=True),
              help="Load config from an existing run as baseline (default: none).")
@click.option("--backbone", default=None, type=FLAG_TYPES["backbone"],
              help="Backbone name (efficientnet_b0..b5, resnet_18, resnet_50) "
                   "(default: task default (efficientnet_b0; resnet_18 for detection)).")
@click.option("--weights", default=None, type=FLAG_TYPES["weights"],
              help=f"Backbone weight init: imagenet (pretrained) or none (random/scratch) "
                   f"(default: {_MODEL.weights}).")
@click.option("--epochs", default=None, type=FLAG_TYPES["epochs"],
              help=f"Number of training epochs (default: {_TRAIN.epochs}).")
@click.option("--lr", default=None, type=FLAG_TYPES["lr"],
              help=f"Learning rate (default: {_TRAIN.learning_rate:g}).")
@click.option("--batch-size", default=None, type=FLAG_TYPES["batch_size"],
              help=f"Batch size (default: {_DATA.batch_size}).")
@click.option("--input-size", default=None, type=FLAG_TYPES["input_size"],
              help=f"Image input size in pixels (default: {_MODEL.input_size}).")
@click.option("--dropout", default=None, type=FLAG_TYPES["dropout"], help=f"Dropout rate (default: {_MODEL.dropout}).")
@click.option("--augmentation", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to an augmentation YAML file, see 'data aug' (default: none).")
@click.option("--resume", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Path to a checkpoint file to resume training from (default: none).")
@click.option("--class-weight", default=None,
              help="Class weighting: null | auto | '{\"cat\": 1.0, \"dog\": 2.5}' (default: null).")
@click.option("--loss", default=None,
              help="Loss function: crossentropy | focal | focal:gamma=2.0 | "
                   "focal:gamma=2.0,label_smoothing=0.1 (default: crossentropy)")
@click.option("--optimizer", default=None,
              help="Optimizer: adam | sgd | adam:weight_decay=1e-4 | "
                   "sgd:weight_decay=1e-4,momentum=0.9 (default: adam)")
@click.option("--lr-scheduler", default=None,
              help="LR scheduler: patience=5 | patience=5,factor=0.5,min=1e-7 (default: disabled).")
@click.option("--fine-tune-from-layer", default=None, type=FLAG_TYPES["fine_tune_from_layer"],
              help="Unfreeze the backbone from this layer index onward: 0 keeps it FROZEN "
                   "(only the new head trains), -1 unfreezes all layers "
                   f"(default: {_MODEL.fine_tune_from_layer}, frozen). "
                   "A frozen backbone often ends near chance accuracy on small or "
                   "unusual datasets; try -1 with a lower --lr.")
@click.option("--val-split", default=None, type=FLAG_TYPES["val_split"],
              help="Fraction of train set to use for validation when no val/ directory "
                   f"exists (default: {_DATA.val_split}).")
@click.option("--seed", default=None, type=FLAG_TYPES["seed"],
              help="Random seed for reproducibility, sets Python, NumPy and TensorFlow seeds "
                   "(default: none, not seeded).")
@click.option("--set", "set_options", multiple=True, metavar="KEY=VALUE",
              help="Set any config key by its dotted name, e.g. --set training.optimizer.weight_decay=1e-4 "
                   "(repeatable; the flags above are shortcuts for common keys; --set wins over a flag).")
def train(data_dir, output_dir, overwrite, from_dir, resume, set_options, **flags):
    """Train a model on DATA_DIR.

    DATA_DIR must contain a train/ subdirectory. val/ is optional (without it,
    --val-split carves one out of train/; detection requires it) and test/ is
    only needed by 'evaluate'. It can be a full path (data/my_dataset) or a bare
    dataset name (my_dataset), resolved under data/.

    The task is auto-detected: a dataset with images/ and labels/ folders is
    trained as detection, a dataset of per-class folders as classification.

    Every option has a default (shown in its help). The default backbone is
    frozen, so only the new head trains; if accuracy ends near chance, unfreeze
    it with --fine-tune-from-layer -1 and a lower --lr, or train longer with
    --epochs, a stronger --augmentation, or --class-weight auto on imbalanced
    data.
    """
    from cvbench.core.data_store import resolve_data_dir
    from cvbench.core.exp_store import check_output_dir

    data_dir = resolve_data_dir(data_dir)
    if output_dir is not None:
        try:
            check_output_dir(output_dir, resume=bool(resume), overwrite=overwrite)
        except CVBenchError as e:
            raise click.BadParameter(e.message, param_hint="--output") from None

    overrides: dict[str, Any] = {}
    for flag, value in flags.items():
        if value is not None:
            overrides.update(parse_override(FLAG_KEYS[flag], value))
    for item in set_options:
        overrides.update(parse_set_option(item))
    validate_dataset(data_dir, from_dir)
    build_config(data_dir, from_dir, overrides)  # reject a bad --set before TensorFlow loads

    from cvbench.cli._listeners import ConsoleListener
    from cvbench.services.training import run_training  # deferred: pulls in TensorFlow

    run_training(
        data_dir=data_dir,
        output_dir=output_dir,
        from_dir=from_dir,
        overrides=overrides,
        resume=resume,
        overwrite=overwrite,
        listener=ConsoleListener(),
    )
