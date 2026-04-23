from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
    backbone: str = "efficientnet_b0"
    input_size: int = 224
    num_classes: int = 0
    dropout: float = 0.2
    fine_tune_from_layer: int = 0
    use_lcn: bool = False
    lcn_kernel_size: int = 32
    lcn_epsilon: float = 1e-3


@dataclass
class TransformConfig:
    name: str
    prob: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    transforms: list = field(default_factory=list)  # list[TransformConfig]


@dataclass
class InterruptConfig:
    enabled: bool = True
    save_checkpoint: bool = True
    save_optimizer_state: bool = True


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
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    class_weight: Any = None  # null | "auto" | {class_name: weight, ...}
    loss: LossConfig = field(default_factory=LossConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    interrupt: InterruptConfig = field(default_factory=InterruptConfig)
    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)


@dataclass
class RunConfig:
    name: str = ""
    date: str = ""
    status: str = "running"
    epochs_run: int = 0
    val_accuracy: Any = None
    val_loss: Any = None
    test_accuracy: Any = None
    resumable: bool = False
    resume_checkpoint: Any = None
    notes: str = ""


@dataclass
class CVBenchConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run: RunConfig = field(default_factory=RunConfig)


# ---------------------------------------------------------------------------
# Internal: dict → config
# ---------------------------------------------------------------------------

def _dict_to_config(d: dict) -> CVBenchConfig:
    cfg = CVBenchConfig()

    dt = d.get("data", {})
    cfg.data = DataConfig(
        data_dir=dt.get("data_dir", cfg.data.data_dir),
        train_dir=dt.get("train_dir", cfg.data.train_dir),
        val_dir=dt.get("val_dir", cfg.data.val_dir),
        test_dir=dt.get("test_dir", cfg.data.test_dir),
        classes=dt.get("classes", cfg.data.classes),
        batch_size=dt.get("batch_size", cfg.data.batch_size),
    )

    m = d.get("model", {})
    cfg.model = ModelConfig(
        backbone=m.get("backbone", cfg.model.backbone),
        input_size=m.get("input_size", cfg.model.input_size),
        num_classes=m.get("num_classes", cfg.model.num_classes),
        dropout=m.get("dropout", cfg.model.dropout),
        fine_tune_from_layer=m.get("fine_tune_from_layer", cfg.model.fine_tune_from_layer),
        use_lcn=m.get("use_lcn", cfg.model.use_lcn),
        lcn_kernel_size=m.get("lcn_kernel_size", cfg.model.lcn_kernel_size),
        lcn_epsilon=m.get("lcn_epsilon", cfg.model.lcn_epsilon),
    )

    aug = d.get("augmentation", {})
    raw_transforms = aug.get("transforms", [])
    transforms = []
    for t in raw_transforms:
        name = t["name"]
        prob = t.get("prob", 1.0)
        params = {k: v for k, v in t.items() if k not in ("name", "prob")}
        transforms.append(TransformConfig(name=name, prob=prob, params=params))
    cfg.augmentation = AugmentationConfig(transforms=transforms)

    tr = d.get("training", {})
    intr = tr.get("interrupt", {})
    ckpt = tr.get("checkpoints", {})
    lrs = tr.get("lr_scheduler", {})
    loss = tr.get("loss", {})
    cfg.training = TrainingConfig(
        epochs=tr.get("epochs", cfg.training.epochs),
        learning_rate=tr.get("learning_rate", cfg.training.learning_rate),
        class_weight=tr.get("class_weight", cfg.training.class_weight),
        loss=LossConfig(
            type=loss.get("type", cfg.training.loss.type),
            label_smoothing=loss.get("label_smoothing", cfg.training.loss.label_smoothing),
            focal_gamma=loss.get("focal_gamma", cfg.training.loss.focal_gamma),
        ),
        lr_scheduler=LRSchedulerConfig(
            patience=lrs.get("patience", cfg.training.lr_scheduler.patience),
            factor=lrs.get("factor", cfg.training.lr_scheduler.factor),
            min_lr=lrs.get("min_lr", cfg.training.lr_scheduler.min_lr),
            monitor=lrs.get("monitor", cfg.training.lr_scheduler.monitor),
        ),
        interrupt=InterruptConfig(
            enabled=intr.get("enabled", cfg.training.interrupt.enabled),
            save_checkpoint=intr.get("save_checkpoint", cfg.training.interrupt.save_checkpoint),
            save_optimizer_state=intr.get(
                "save_optimizer_state", cfg.training.interrupt.save_optimizer_state
            ),
        ),
        checkpoints=CheckpointsConfig(
            strategy=ckpt.get("strategy", cfg.training.checkpoints.strategy),
            every_n_epochs=ckpt.get("every_n_epochs", cfg.training.checkpoints.every_n_epochs),
            monitor=ckpt.get("monitor", cfg.training.checkpoints.monitor),
            mode=ckpt.get("mode", cfg.training.checkpoints.mode),
            keep_last_n=ckpt.get("keep_last_n", cfg.training.checkpoints.keep_last_n),
        ),
    )

    r = d.get("run", {})
    cfg.run = RunConfig(
        name=r.get("name", cfg.run.name),
        date=r.get("date", cfg.run.date),
        status=r.get("status", cfg.run.status),
        epochs_run=r.get("epochs_run", cfg.run.epochs_run),
        val_accuracy=r.get("val_accuracy", cfg.run.val_accuracy),
        val_loss=r.get("val_loss", cfg.run.val_loss),
        test_accuracy=r.get("test_accuracy", cfg.run.test_accuracy),
        resumable=r.get("resumable", cfg.run.resumable),
        resume_checkpoint=r.get("resume_checkpoint", cfg.run.resume_checkpoint),
        notes=r.get("notes", cfg.run.notes),
    )

    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_config(
    data_dir: str,
    from_dir: str | None = None,
    backbone: str | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    input_size: int | None = None,
    dropout: float | None = None,
    class_weight: Any = None,
    loss: "LossConfig | None" = None,
    lr_patience: int | None = None,
    lr_factor: float | None = None,
    lr_min: float | None = None,
    fine_tune_from_layer: int | None = None,
    use_lcn: bool | None = None,
    lcn_kernel_size: int | None = None,
    lcn_epsilon: float | None = None,
    val_split: float | None = None,
) -> CVBenchConfig:
    """Build a CVBenchConfig from CLI options.

    If from_dir is given, loads that experiment's config.yaml as a baseline.
    Any non-None CLI options override the loaded/default values.
    Data directories are always derived from data_dir.
    """
    if from_dir is not None:
        cfg = load_config(from_dir)
    else:
        cfg = CVBenchConfig()

    cfg.data.data_dir = data_dir
    cfg.data.train_dir = str(Path(data_dir) / "train")
    cfg.data.val_dir = str(Path(data_dir) / "val")
    cfg.data.test_dir = str(Path(data_dir) / "test")

    if backbone is not None:
        cfg.model.backbone = backbone
    if epochs is not None:
        cfg.training.epochs = epochs
    if lr is not None:
        cfg.training.learning_rate = lr
    if batch_size is not None:
        cfg.data.batch_size = batch_size
    if input_size is not None:
        cfg.model.input_size = input_size
    if dropout is not None:
        cfg.model.dropout = dropout
    if class_weight is not None:
        cfg.training.class_weight = class_weight
    if loss is not None:
        cfg.training.loss = loss
    if lr_patience is not None:
        cfg.training.lr_scheduler.patience = lr_patience
    if lr_factor is not None:
        cfg.training.lr_scheduler.factor = lr_factor
    if lr_min is not None:
        cfg.training.lr_scheduler.min_lr = lr_min
    if fine_tune_from_layer is not None:
        cfg.model.fine_tune_from_layer = fine_tune_from_layer
    if use_lcn is not None:
        cfg.model.use_lcn = use_lcn
    if lcn_kernel_size is not None:
        cfg.model.lcn_kernel_size = lcn_kernel_size
    if lcn_epsilon is not None:
        cfg.model.lcn_epsilon = lcn_epsilon
    if val_split is not None:
        cfg.data.val_split = val_split
        cfg.data.val_split_explicit = True

    return cfg


def save_config(cfg: CVBenchConfig, exp_dir: str):
    """Write the fully resolved config to config.yaml in exp_dir."""
    path = Path(exp_dir) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_dict(obj):
        if isinstance(obj, TransformConfig):
            d = {"name": obj.name, "prob": obj.prob}
            d.update(obj.params)
            return d
        if dataclasses.is_dataclass(obj):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        return obj

    with open(path, "w") as f:
        yaml.dump(_to_dict(cfg), f, default_flow_style=False, sort_keys=False)


def load_config(exp_dir: str) -> CVBenchConfig:
    """Load config.yaml from an experiment directory."""
    path = Path(exp_dir) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config.yaml found in {exp_dir}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_config(raw)


def load_aug_file(path: str) -> AugmentationConfig:
    """Load an augmentation YAML file and return an AugmentationConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw_transforms = raw.get("transforms", [])
    transforms = []
    for t in raw_transforms:
        name = t["name"]
        prob = t.get("prob", 1.0)
        params = {k: v for k, v in t.items() if k not in ("name", "prob")}
        transforms.append(TransformConfig(name=name, prob=prob, params=params))
    return AugmentationConfig(transforms=transforms)


def update_run_status(exp_dir: str, **kwargs):
    """Update fields in the run: section of config.yaml in-place."""
    cfg = load_config(exp_dir)
    for k, v in kwargs.items():
        if hasattr(cfg.run, k):
            setattr(cfg.run, k, v)
    save_config(cfg, exp_dir)
