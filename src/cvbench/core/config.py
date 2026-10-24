from __future__ import annotations

import contextlib
import dataclasses
import difflib
import functools
import os
import time
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cvbench.core.errors import ConfigError, CVBenchError, NotFoundError
from cvbench.tasks import DEFAULT_TASK

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    data_dir: str = "data"
    train_dir: str = ""
    val_dir: str = ""
    test_dir: str = ""
    classes: list = field(default_factory=list)
    batch_size: int = 16
    val_split: float = 0.2  # fraction of train used for val when val/ dir is absent
    val_split_explicit: bool = False  # True when val_split was passed explicitly by the user


@dataclass
class ModelConfig:
    backbone: str = ""  # "" = the task's default_backbone, resolved by run_training
    weights: str = "imagenet"  # "imagenet" | "none"
    input_size: int = 224
    num_classes: int = 0
    dropout: float = 0.2
    fine_tune_from_layer: int = 0


@dataclass
class TransformConfig:
    name: str
    prob: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class OneOfCandidateConfig:
    name: str
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class OneOfConfig:
    prob: float = 1.0
    candidates: list = field(default_factory=list)  # list[OneOfCandidateConfig]


@dataclass
class AugmentationConfig:
    transforms: list = field(default_factory=list)  # list[TransformConfig | OneOfConfig]


@dataclass
class InterruptConfig:
    enabled: bool = True
    save_checkpoint: bool = True


@dataclass
class CheckpointsConfig:
    strategy: str = "best_only"
    every_n_epochs: int = 5
    monitor: str = "val_loss"
    mode: str = "min"
    keep_last_n: int = 1


@dataclass
class LRSchedulerConfig:
    patience: int = 0       # 0 = disabled; >0 enables ReduceLROnPlateau
    factor: float = 0.5
    min_lr: float = 1e-7
    monitor: str = "val_loss"


@dataclass
class LossConfig:
    type: str = "crossentropy"   # "crossentropy" | "focal"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0     # only used when type="focal"


@dataclass
class OptimizerConfig:
    type: str = "adam"           # "adam" | "sgd"
    weight_decay: float = 0.0
    momentum: float = 0.9        # only used when type="sgd"


@dataclass
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    seed: int | None = None
    class_weight: Any = None  # null | "auto" | {class_name: weight, ...}
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    interrupt: InterruptConfig = field(default_factory=InterruptConfig)
    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)


@dataclass
class RunConfig:
    name: str = ""
    date: str = ""
    status: str = "running"
    pid: int = 0  # PID of the process that set status="running"; used to detect a
    # crashed/killed run whose status was never updated (see load_config).
    epochs_run: int = 0
    # {metric name: value} — the headline numbers of the run. Which names exist is the
    # task's call (Task.metrics_schema); `primary_metric` names the one that stands for
    # the run as a whole (the test score, or the val score a sweep ranks on).
    metrics: dict = field(default_factory=dict)
    primary_metric: str = "accuracy"
    resumable: bool = False
    resume_checkpoint: Any = None
    notes: str = ""
    cli_command: str = ""

    @property
    def primary_value(self) -> Any:
        return self.metrics.get(self.primary_metric)

    @primary_value.setter
    def primary_value(self, value: Any) -> None:
        self._set_metric(self.primary_metric, value)

    def _set_metric(self, name: str, value: Any) -> None:
        if value is None:
            self.metrics.pop(name, None)
        else:
            self.metrics[name] = value


@dataclass
class CVBenchConfig:
    task: str = DEFAULT_TASK  # any name registered in cvbench.tasks
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run: RunConfig = field(default_factory=RunConfig)
    # Task-owned sections (``Task.config_section``), keyed by section name. Only the run's
    # own task has one; it is read as an attribute (``cfg.<section>``) and written at the
    # top level of config.yaml. ``task`` must be known before sections are built, so an
    # override that changes ``task`` resets the sections to the new task's defaults.
    sections: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sync_sections()

    def sync_sections(self) -> None:
        """Keep exactly the section the current task owns (defaults when it is missing)."""
        owned = _task_section(self.task)
        self.sections = (
            {owned[0]: self.sections.get(owned[0]) or owned[1]()} if owned else {}
        )

    def __getattr__(self, name: str) -> Any:  # only called for attributes that are missing
        sections = self.__dict__.get("sections", {})
        if name in sections:
            return sections[name]
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# dict <-> config (generic: driven by the dataclass fields and their types)
# ---------------------------------------------------------------------------

# `train` / `sweep` flag -> dotted config key. Pure sugar: any config key is also
# settable (`--set`) and sweepable by its dotted name. A flag whose key is a section
# (optimizer, loss, lr_scheduler) takes a `type:k=v` spec; `augmentation` takes a file.
FLAG_KEYS: dict[str, str] = {
    "backbone": "model.backbone",
    "weights": "model.weights",
    "epochs": "training.epochs",
    "lr": "training.learning_rate",
    "batch_size": "data.batch_size",
    "input_size": "model.input_size",
    "dropout": "model.dropout",
    "optimizer": "training.optimizer",
    "loss": "training.loss",
    "lr_scheduler": "training.lr_scheduler",
    "class_weight": "training.class_weight",
    "fine_tune_from_layer": "model.fine_tune_from_layer",
    "val_split": "data.val_split",
    "seed": "training.seed",
    "augmentation": "augmentation",
}
KEY_FLAGS = {key: flag for flag, key in FLAG_KEYS.items()}

# Setting these also records that the user chose the value (see ModelConfig / DataConfig).
_EXPLICIT = {"data.val_split": "data.val_split_explicit"}


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    hints = typing.get_type_hints(cls)
    if cls is CVBenchConfig:
        hints.pop("sections")  # task sections are addressed by their own names, see _task_sections
    return hints


def _is_section(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """``X | None`` -> (X, True); anything else -> (tp, False)."""
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _unknown_key(name: str, valid: typing.Iterable[str], prefix: str) -> CVBenchError:
    valid = list(valid)
    msg = f"Unknown config key '{prefix}{name}'."
    close = difflib.get_close_matches(name, valid, n=1, cutoff=0.6)
    msg += f" Did you mean '{prefix}{close[0]}'?" if close else f" Valid keys here: {', '.join(valid)}."
    return CVBenchError(msg)


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Check ``value`` against the field type ``tp`` (ints widen to float); raise a CVBenchError naming ``path``."""
    tp, optional = _unwrap_optional(tp)
    if value is None and optional:
        return None
    if tp is Any:
        return value
    if tp is float:
        if isinstance(value, str):  # YAML reads `1e-4` as a string
            with contextlib.suppress(ValueError):
                value = float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    ok = isinstance(value, tp) and not (tp is int and isinstance(value, bool))
    if not ok:
        raise CVBenchError(f"Invalid value for '{path}': expected {tp.__name__}, got {value!r}.")
    return value


def _task_sections() -> dict[str, tuple[str, type]]:
    """``{section name: (owning task, section dataclass)}`` over every registered task."""
    from cvbench.tasks import get_task, task_names

    return {s[0]: (n, s[1]) for n in task_names() if (s := get_task(n).config_section)}


def _task_section(task: str) -> tuple[str, type] | None:
    """``(name, dataclass)`` of the section TASK owns; None if it has none or is unregistered."""
    from cvbench.tasks import get_task

    try:
        return get_task(task).config_section
    except ConfigError:
        return None


def _section_owners() -> dict[str, str]:
    """``{config section name: task that owns it}`` over every registered task."""
    return {name: owner for name, (owner, _) in _task_sections().items()}


def _parse_transforms(raw: list) -> list:
    transforms: list[TransformConfig | OneOfConfig] = []
    for t in raw or []:
        if "one_of" in t:
            group = t["one_of"]
            candidates = []
            for c in group.get("candidates", []):
                name = c["name"]
                weight = c.get("weight", 1.0)
                params = {k: v for k, v in c.items() if k not in ("name", "weight")}
                candidates.append(OneOfCandidateConfig(name=name, weight=weight, params=params))
            transforms.append(OneOfConfig(prob=group.get("prob", 1.0), candidates=candidates))
        else:
            name = t["name"]
            prob = t.get("prob", 1.0)
            params = {k: v for k, v in t.items() if k not in ("name", "prob")}
            transforms.append(TransformConfig(name=name, prob=prob, params=params))
    return transforms


# Fields whose YAML shape is not their dataclass shape.
_FIELD_LOADERS: dict[tuple[type, str], typing.Callable[[Any], Any]] = {
    (AugmentationConfig, "transforms"): _parse_transforms,
}


def _run_metrics(r: dict) -> dict:
    """The run's ``metrics`` dict, migrating the pre-``metrics`` keys of an older config.yaml.

    Older runs stored ``val_loss`` / ``val_accuracy`` flat and the test score as
    ``test_accuracy`` labelled by ``test_metric``; the score is re-keyed by that label.
    """
    if isinstance(r.get("metrics"), dict):
        return dict(r["metrics"])
    metrics = {k: r[k] for k in ("val_loss", "val_accuracy") if r.get(k) is not None}
    if r.get("test_accuracy") is not None:
        metrics[r.get("test_metric") or "accuracy"] = r["test_accuracy"]
    return metrics


_LEGACY_RUN_KEYS = ("val_loss", "val_accuracy", "test_accuracy", "test_metric")


def _migrate_run(r: dict) -> dict:
    out = {k: v for k, v in r.items() if k not in _LEGACY_RUN_KEYS}
    out["metrics"] = _run_metrics(r)
    if "primary_metric" not in out and r.get("test_metric"):
        out["primary_metric"] = r["test_metric"]
    return out


# Fields removed from the dataclasses that older run directories still carry; dropped on load
# so those runs stay readable (any other unknown key is an error).
_REMOVED_KEYS = frozenset({
    "detection.grid_stride", "training.grad_accum", "training.mixup",
    "model.use_lcn", "model.lcn_kernel_size", "model.lcn_epsilon", "model.normalization",
    "model.backbone_explicit", "interrupt.save_optimizer_state",
})


def _build(cls: type, raw: Any, path: str = "") -> Any:
    """Build dataclass ``cls`` from a dict; unknown keys and mistyped values raise a CVBenchError."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise CVBenchError(f"Invalid value for '{path or 'config'}': expected a mapping, got {raw!r}.")
    hints = _hints(cls)
    prefix = f"{path}." if path else ""
    kwargs = {}
    for name, value in raw.items():
        if f"{prefix}{name}" in _REMOVED_KEYS:
            continue
        if name not in hints:
            raise _unknown_key(name, hints, prefix)
        tp = hints[name]
        if (cls, name) in _FIELD_LOADERS:
            kwargs[name] = _FIELD_LOADERS[(cls, name)](value)
        elif _is_section(tp):
            kwargs[name] = _build(tp, value, f"{prefix}{name}")
        else:
            kwargs[name] = _coerce(value, tp, f"{prefix}{name}")
    return cls(**kwargs)


def config_from_dict(d: dict) -> CVBenchConfig:
    """Build a config from parsed YAML.

    The run's task section is built from its top-level block (defaults when absent) and
    stored in ``sections``; sections owned by another task are ignored (older configs
    carry another task's block on every run); the pre-``metrics`` run keys are migrated.
    """
    d = dict(d)
    if "sections" in d:
        raise _unknown_key("sections", _hints(CVBenchConfig), "")
    task = d.get("task") or DEFAULT_TASK
    owned = _task_section(task)
    raw_section = None
    for section in _task_sections():
        value = d.pop(section, None)
        if owned and section == owned[0]:
            raw_section = value
    if isinstance(d.get("run"), dict):
        d["run"] = _migrate_run(d["run"])
    cfg = _build(CVBenchConfig, d)
    if owned:
        cfg.sections = {owned[0]: _build(owned[1], raw_section, owned[0])}
    return cfg


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, TransformConfig):
        return {"name": obj.name, "prob": obj.prob, **obj.params}
    if isinstance(obj, OneOfConfig):
        return {"one_of": {
            "prob": obj.prob,
            "candidates": [{"name": c.name, "weight": c.weight, **c.params} for c in obj.candidates],
        }}
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_plain(item) for item in obj]
    return obj


def config_to_dict(cfg: CVBenchConfig) -> dict:
    """The exact inverse of ``config_from_dict``: plain dicts, the run's task section at the top level."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        if f.name == "sections":
            continue
        if f.name == "run":  # task sections sit just before the run bookkeeping
            out.update({name: _to_plain(section) for name, section in cfg.sections.items()})
        out[f.name] = _to_plain(getattr(cfg, f.name))
    return out


# ---------------------------------------------------------------------------
# Dotted-key overrides
# ---------------------------------------------------------------------------

def _key_type(key: str) -> Any:
    """Declared type of the config field at dotted ``key``; unknown keys raise a did-you-mean CVBenchError."""
    tp: Any = CVBenchConfig
    parts: list[str] = []
    for part in key.split("."):
        if not _is_section(tp):
            raise CVBenchError(f"Unknown config key '{key}': '{'.'.join(parts)}' is not a section.")
        hints = _hints(tp)
        if tp is CVBenchConfig:  # task-owned sections resolve through their owner's dataclass
            hints = {**hints, **{n: cls for n, (_, cls) in _task_sections().items()}}
        if part not in hints:
            raise _unknown_key(part, hints, "".join(f"{p}." for p in parts))
        tp = hints[part]
        parts.append(part)
    return tp


def resolve_key(name: str) -> str:
    """Canonical dotted key for a flag alias (``lr``) or a dotted key; unknown names raise a CVBenchError."""
    key = FLAG_KEYS.get(name, name)
    _key_type(key)
    return key


def key_type(key: str) -> Any:
    """Declared type of ``key`` with ``| None`` stripped (``int`` for ``training.seed``)."""
    return _unwrap_optional(_key_type(key))[0]


def _parse_text(value: Any, tp: Any) -> Any:
    """Command-line text is read as YAML into the field's type; strings stay strings; non-text passes through."""
    if not isinstance(value, str):
        return value
    inner, optional = _unwrap_optional(tp)
    if optional and value.strip().lower() in ("none", "null", ""):
        return None
    if inner is str:
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_key(cfg: CVBenchConfig, key: str, value: Any) -> None:
    tp = _key_type(key)
    *parents, leaf = key.split(".")
    owner = _section_owners().get(parents[0] if parents else key)
    if owner is not None and owner != cfg.task:
        raise CVBenchError(
            f"'{key}' only applies to {owner} runs, but this is a {cfg.task} run.", param_hint=key
        )
    if tp is AugmentationConfig:
        if isinstance(value, str):
            value = load_aug_file(value)
        elif not isinstance(value, AugmentationConfig):
            raise CVBenchError(f"'{key}' takes an augmentation YAML file path.")
    elif _is_section(tp):
        raise CVBenchError(
            f"'{key}' is a section; set one of its keys: "
            + ", ".join(f"{key}.{n}" for n in _hints(tp)) + "."
        )
    else:
        value = _coerce(_parse_text(value, tp), tp, key)
        if key == "task":
            from cvbench.tasks import task_names

            if value not in task_names():
                raise CVBenchError(f"Unknown task '{value}'. Valid options: {', '.join(task_names())}.")
    obj: Any = cfg
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, leaf, value)
    if key == "task":
        cfg.sync_sections()
    if key in _EXPLICIT:
        _set_key(cfg, _EXPLICIT[key], True)


def apply_overrides(cfg: CVBenchConfig, overrides: dict[str, Any]) -> CVBenchConfig:
    """Set each dotted key (``{"training.learning_rate": 1e-3}``) on ``cfg`` in place and return it.

    Values are checked against the field type; command-line strings are parsed first.
    ``task`` is applied before the rest so task-owned sections can be checked against it.
    """
    for key in sorted(overrides, key=lambda k: k != "task"):
        _set_key(cfg, key, overrides[key])
    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_config(
    data_dir: str,
    from_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> CVBenchConfig:
    """Build a CVBenchConfig for ``data_dir``.

    The baseline is the ``from_dir`` experiment's config.yaml, or defaults with the task
    sniffed from the dataset layout. ``overrides`` (dotted key -> value) are applied on top.
    Data directories are always derived from ``data_dir``.
    """
    from cvbench.tasks import detect_task_name

    cfg = load_config(from_dir) if from_dir is not None else CVBenchConfig(task=detect_task_name(data_dir))

    cfg.data.data_dir = data_dir
    cfg.data.train_dir = str(Path(data_dir) / "train")
    cfg.data.val_dir = str(Path(data_dir) / "val")
    cfg.data.test_dir = str(Path(data_dir) / "test")

    return apply_overrides(cfg, overrides or {})


def save_config(cfg: CVBenchConfig, exp_dir: str):
    """Write the fully resolved config to config.yaml in exp_dir."""
    path = Path(exp_dir) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_to_dict(cfg), f, default_flow_style=False, sort_keys=False)


def _is_pid_alive(pid: int) -> bool:
    """Best-effort check for whether a process is still alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by someone else — still alive.
        return True
    except OSError:
        return False
    return True


# Legacy runs (created before RunConfig.pid existed) have no PID to check, so a
# stuck status="running" is instead inferred from file inactivity: no training run
# should go this long without touching a single file in its directory (CSVLogger
# writes at least once per epoch).
_LEGACY_STALE_SECONDS = 6 * 3600  # 6 hours


def _dir_last_activity(exp_dir: str) -> float | None:
    """Latest mtime among all files under exp_dir, or None if it has none."""
    latest = None
    for root, _dirs, files in os.walk(exp_dir):
        for name in files:
            try:
                mtime = os.path.getmtime(os.path.join(root, name))
            except OSError:
                continue
            if latest is None or mtime > latest:
                latest = mtime
    return latest


def load_config(exp_dir: str) -> CVBenchConfig:
    """Load config.yaml from an experiment directory.

    Corrects a stale status="running" left behind by a run that crashed or was
    killed externally (kill -9, OOM, closed terminal) and so never got the chance
    to update its own status:
    - Runs with a recorded PID (started after this check was added): stale if
      that PID is no longer alive.
    - Legacy runs with no recorded PID (pid=0): stale if nothing in the run
      directory has been touched in over _LEGACY_STALE_SECONDS.

    The correction is in-memory only: reading never rewrites config.yaml. It
    reaches disk the next time a writer (e.g. update_run_status) saves the config.
    """
    path = Path(exp_dir) / "config.yaml"
    if not path.exists():
        raise NotFoundError(f"No config.yaml found in {exp_dir}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = config_from_dict(raw)
    if cfg.run.status == "running":
        if cfg.run.pid:
            stale = not _is_pid_alive(cfg.run.pid)
        else:
            latest = _dir_last_activity(exp_dir)
            stale = latest is not None and (time.time() - latest) > _LEGACY_STALE_SECONDS
        if stale:
            cfg.run.status = "failed"
    return cfg


def load_aug_file(path: str) -> AugmentationConfig:
    """Load an augmentation YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AugmentationConfig(transforms=_parse_transforms(raw.get("transforms", [])))


def update_run_status(exp_dir: str, **kwargs):
    """Update fields in the run: section of config.yaml in-place.

    ``metrics`` is merged into the run's metrics rather than replacing them.
    """
    cfg = load_config(exp_dir)
    for k, v in kwargs.items():
        if k == "metrics":
            cfg.run.metrics.update(v)
        elif hasattr(cfg.run, k):
            setattr(cfg.run, k, v)
    save_config(cfg, exp_dir)
