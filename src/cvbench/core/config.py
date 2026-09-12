from __future__ import annotations

import dataclasses
import os
import time
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
    backbone_explicit: bool = False  # True when --backbone was passed explicitly by the user
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
    val_accuracy: Any = None
    val_loss: Any = None
    test_accuracy: Any = None
    test_metric: str = "accuracy"  # label for test_accuracy: "accuracy" | "map50" | ...
    resumable: bool = False
    resume_checkpoint: Any = None
    notes: str = ""
    cli_command: str = ""


@dataclass
class DetectionConfig:
    """Detection-only settings. Present (with defaults) even for classification runs —
    it is simply unused, the same way TrainingConfig.class_weight is unused by detection."""
    strides: list = field(default_factory=lambda: [16, 32])  # output strides of the YOLO heads
    anchors: list = field(default_factory=list)  # resolved once (see detection/anchors.py),
    # then persisted here — [[ [w, h] x anchors_per_scale ] x len(strides)], smallest-area
    # anchors first (assigned to the smallest stride).
    anchors_per_scale: int = 3
    ignore_iou_threshold: float = 0.5  # non-assigned anchors overlapping a GT above this are
    # excluded from the objectness loss entirely (neither positive nor negative)
    noobj_weight: float = 0.5     # objectness loss weight on negative (non-object) cells
    conf_threshold: float = 0.25  # minimum score for a decoded detection to count
    iou_threshold: float = 0.5    # IoU at which a prediction counts as matching a ground truth
    max_detections: int = 100     # top-k cap on decoded detections per image


@dataclass
class CVBenchConfig:
    task: str = "classification"  # "classification" | "detection" | ...
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    run: RunConfig = field(default_factory=RunConfig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_transforms(raw: list) -> list:
    transforms: list[TransformConfig | OneOfConfig] = []
    for t in raw:
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


# ---------------------------------------------------------------------------
# Internal: dict → config
# ---------------------------------------------------------------------------

def _dict_to_config(d: dict) -> CVBenchConfig:
    cfg = CVBenchConfig()

    cfg.task = d.get("task", cfg.task)

    dt = d.get("data", {})
    cfg.data = DataConfig(
        data_dir=dt.get("data_dir", cfg.data.data_dir),
        train_dir=dt.get("train_dir", cfg.data.train_dir),
        val_dir=dt.get("val_dir", cfg.data.val_dir),
        test_dir=dt.get("test_dir", cfg.data.test_dir),
        classes=dt.get("classes", cfg.data.classes),
        batch_size=dt.get("batch_size", cfg.data.batch_size),
        val_split=dt.get("val_split", cfg.data.val_split),
        val_split_explicit=dt.get("val_split_explicit", cfg.data.val_split_explicit),
    )

    m = d.get("model", {})
    cfg.model = ModelConfig(
        backbone=m.get("backbone", cfg.model.backbone),
        backbone_explicit=m.get("backbone_explicit", cfg.model.backbone_explicit),
        weights=m.get("weights", cfg.model.weights),
        input_size=m.get("input_size", cfg.model.input_size),
        num_classes=m.get("num_classes", cfg.model.num_classes),
        dropout=m.get("dropout", cfg.model.dropout),
        fine_tune_from_layer=m.get("fine_tune_from_layer", cfg.model.fine_tune_from_layer),
    )

    aug = d.get("augmentation", {})
    cfg.augmentation = AugmentationConfig(transforms=_parse_transforms(aug.get("transforms", [])))

    tr = d.get("training", {})
    intr = tr.get("interrupt", {})
    ckpt = tr.get("checkpoints", {})
    lrs = tr.get("lr_scheduler", {})
    loss = tr.get("loss", {})
    opt = tr.get("optimizer", {})
    cfg.training = TrainingConfig(
        epochs=tr.get("epochs", cfg.training.epochs),
        learning_rate=tr.get("learning_rate", cfg.training.learning_rate),
        seed=tr.get("seed", cfg.training.seed),
        class_weight=tr.get("class_weight", cfg.training.class_weight),
        loss=LossConfig(
            type=loss.get("type", cfg.training.loss.type),
            label_smoothing=loss.get("label_smoothing", cfg.training.loss.label_smoothing),
            focal_gamma=loss.get("focal_gamma", cfg.training.loss.focal_gamma),
        ),
        optimizer=OptimizerConfig(
            type=opt.get("type", cfg.training.optimizer.type),
            weight_decay=opt.get("weight_decay", cfg.training.optimizer.weight_decay),
            momentum=opt.get("momentum", cfg.training.optimizer.momentum),
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

    det = d.get("detection", {})
    cfg.detection = DetectionConfig(
        strides=det.get("strides", cfg.detection.strides),
        anchors=det.get("anchors", cfg.detection.anchors),
        anchors_per_scale=det.get("anchors_per_scale", cfg.detection.anchors_per_scale),
        ignore_iou_threshold=det.get("ignore_iou_threshold", cfg.detection.ignore_iou_threshold),
        noobj_weight=det.get("noobj_weight", cfg.detection.noobj_weight),
        conf_threshold=det.get("conf_threshold", cfg.detection.conf_threshold),
        iou_threshold=det.get("iou_threshold", cfg.detection.iou_threshold),
        max_detections=det.get("max_detections", cfg.detection.max_detections),
    )

    r = d.get("run", {})
    cfg.run = RunConfig(
        name=r.get("name", cfg.run.name),
        date=r.get("date", cfg.run.date),
        status=r.get("status", cfg.run.status),
        pid=r.get("pid", cfg.run.pid),
        epochs_run=r.get("epochs_run", cfg.run.epochs_run),
        val_accuracy=r.get("val_accuracy", cfg.run.val_accuracy),
        val_loss=r.get("val_loss", cfg.run.val_loss),
        test_accuracy=r.get("test_accuracy", cfg.run.test_accuracy),
        test_metric=r.get("test_metric", cfg.run.test_metric),
        resumable=r.get("resumable", cfg.run.resumable),
        resume_checkpoint=r.get("resume_checkpoint", cfg.run.resume_checkpoint),
        notes=r.get("notes", cfg.run.notes),
        cli_command=r.get("cli_command", cfg.run.cli_command),
    )

    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_config(
    data_dir: str,
    from_dir: str | None = None,
    task: str | None = None,
    backbone: str | None = None,
    weights: str | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    input_size: int | None = None,
    dropout: float | None = None,
    class_weight: Any = None,
    loss: LossConfig | None = None,
    optimizer: OptimizerConfig | None = None,
    lr_scheduler: LRSchedulerConfig | None = None,
    fine_tune_from_layer: int | None = None,
    val_split: float | None = None,
    seed: int | None = None,
) -> CVBenchConfig:
    """Build a CVBenchConfig from CLI options.

    If from_dir is given, loads that experiment's config.yaml as a baseline.
    Any non-None CLI options override the loaded/default values.
    Data directories are always derived from data_dir.
    """
    cfg = load_config(from_dir) if from_dir is not None else CVBenchConfig()

    cfg.data.data_dir = data_dir
    cfg.data.train_dir = str(Path(data_dir) / "train")
    cfg.data.val_dir = str(Path(data_dir) / "val")
    cfg.data.test_dir = str(Path(data_dir) / "test")

    if task is not None:
        cfg.task = task
    if backbone is not None:
        cfg.model.backbone = backbone
        cfg.model.backbone_explicit = True
    if weights is not None:
        cfg.model.weights = weights
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
    if optimizer is not None:
        cfg.training.optimizer = optimizer
    if lr_scheduler is not None:
        cfg.training.lr_scheduler = lr_scheduler
    if fine_tune_from_layer is not None:
        cfg.model.fine_tune_from_layer = fine_tune_from_layer
    if val_split is not None:
        cfg.data.val_split = val_split
        cfg.data.val_split_explicit = True
    if seed is not None:
        cfg.training.seed = seed

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
        if isinstance(obj, OneOfConfig):
            return {
                "one_of": {
                    "prob": obj.prob,
                    "candidates": [
                        {"name": c.name, "weight": c.weight, **c.params}
                        for c in obj.candidates
                    ],
                }
            }
        if dataclasses.is_dataclass(obj):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        return obj

    with open(path, "w") as f:
        yaml.dump(_to_dict(cfg), f, default_flow_style=False, sort_keys=False)


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
    """
    path = Path(exp_dir) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config.yaml found in {exp_dir}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _dict_to_config(raw)
    if cfg.run.status == "running":
        if cfg.run.pid:
            stale = not _is_pid_alive(cfg.run.pid)
        else:
            latest = _dir_last_activity(exp_dir)
            stale = latest is not None and (time.time() - latest) > _LEGACY_STALE_SECONDS
        if stale:
            cfg.run.status = "failed"
            save_config(cfg, exp_dir)
    return cfg


def load_aug_file(path: str) -> AugmentationConfig:
    """Load an augmentation YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AugmentationConfig(transforms=_parse_transforms(raw.get("transforms", [])))


def update_run_status(exp_dir: str, **kwargs):
    """Update fields in the run: section of config.yaml in-place."""
    cfg = load_config(exp_dir)
    for k, v in kwargs.items():
        if hasattr(cfg.run, k):
            setattr(cfg.run, k, v)
    save_config(cfg, exp_dir)
