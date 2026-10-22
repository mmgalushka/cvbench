import os

import pytest
import yaml

from cvbench.core.config import (
    build_config,
    load_config,
    save_config,
    update_run_status,
)
from cvbench.core.errors import NotFoundError


def test_build_config_defaults():
    cfg = build_config("data")
    assert cfg.data.data_dir == "data"
    assert cfg.data.train_dir == "data/train"
    assert cfg.data.val_dir == "data/val"
    assert cfg.data.test_dir == "data/test"
    assert cfg.model.backbone == ""  # "" = the task default, resolved by run_training
    assert cfg.training.epochs == 10
    assert cfg.training.learning_rate == 1e-4
    assert cfg.augmentation.transforms == []


def test_build_config_cli_overrides():
    cfg = build_config("data", overrides={
        "model.backbone": "efficientnet_b3",
        "training.epochs": 20,
        "training.learning_rate": 5e-5,
        "data.batch_size": 32,
        "model.dropout": 0.3,
    })
    assert cfg.model.backbone == "efficientnet_b3"
    assert cfg.training.epochs == 20
    assert cfg.training.learning_rate == 5e-5
    assert cfg.data.batch_size == 32
    assert cfg.model.dropout == 0.3


def test_build_config_from_dir(tmp_path):
    # Create an existing experiment with a config.yaml
    cfg_orig = build_config("data", overrides={"model.backbone": "efficientnet_b3", "training.epochs": 50})
    cfg_orig.run.name = "test_exp"
    save_config(cfg_orig, str(tmp_path))

    # Load from that dir as baseline, override just epochs
    cfg_new = build_config("data2", from_dir=str(tmp_path), overrides={"training.epochs": 5})
    assert cfg_new.model.backbone == "efficientnet_b3"  # inherited
    assert cfg_new.training.epochs == 5                 # overridden
    assert cfg_new.data.data_dir == "data2"             # always updated


def test_build_config_from_dir_data_dir_always_updated(tmp_path):
    cfg_orig = build_config("old_data")
    save_config(cfg_orig, str(tmp_path))

    cfg_new = build_config("new_data", from_dir=str(tmp_path))
    assert cfg_new.data.data_dir == "new_data"
    assert cfg_new.data.train_dir == "new_data/train"


def test_save_and_load_config(tmp_path):
    cfg = build_config("data", overrides={"model.backbone": "efficientnet_b0", "training.epochs": 5})
    cfg.data.classes = ["cat", "dog"]
    cfg.model.num_classes = 2
    cfg.run.name = "effnet_b0_lr1e4_2026_03_28"

    save_config(cfg, str(tmp_path))
    assert (tmp_path / "config.yaml").exists()

    reloaded = load_config(str(tmp_path))
    assert reloaded.model.backbone == "efficientnet_b0"
    assert reloaded.training.epochs == 5
    assert reloaded.data.classes == ["cat", "dog"]
    assert reloaded.model.num_classes == 2
    assert reloaded.run.name == "effnet_b0_lr1e4_2026_03_28"


def test_load_config_missing_raises(tmp_path):
    with pytest.raises(NotFoundError):
        load_config(str(tmp_path))


def test_update_run_status(tmp_path):
    cfg = build_config("data")
    save_config(cfg, str(tmp_path))

    update_run_status(str(tmp_path), status="done", epochs_run=10, metrics={"val_accuracy": 0.92})

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.status == "done"
    assert reloaded.run.epochs_run == 10
    assert reloaded.run.metrics.get("val_accuracy") == 0.92


def test_update_run_status_interrupted(tmp_path):
    cfg = build_config("data")
    save_config(cfg, str(tmp_path))

    update_run_status(
        str(tmp_path),
        status="interrupted",
        resumable=True,
        resume_checkpoint="/path/to/ckpt.keras",
        epochs_run=5,
    )

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.status == "interrupted"
    assert reloaded.run.resumable is True
    assert reloaded.run.resume_checkpoint == "/path/to/ckpt.keras"


def _dead_pid() -> int:
    """Spawn and wait on a short-lived subprocess, returning its now-dead PID."""
    import subprocess
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_load_config_marks_stale_running_as_failed(tmp_path):
    # Simulates a run whose process was killed (kill -9, OOM, crash) without
    # ever getting the chance to update its own status.
    cfg = build_config("data")
    cfg.run.status = "running"
    cfg.run.pid = _dead_pid()
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.status == "failed"

    # Reading never mutates the file: the correction is in-memory only.
    on_disk = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert on_disk["run"]["status"] == "running"


def test_load_config_leaves_running_alone_when_pid_alive(tmp_path):
    cfg = build_config("data")
    cfg.run.status = "running"
    cfg.run.pid = os.getpid()
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.status == "running"


def test_load_config_leaves_recent_legacy_running_alone(tmp_path):
    # Runs created before the pid field existed default to pid=0, so liveness
    # falls back to file-activity — a just-saved config counts as "recent".
    cfg = build_config("data")
    cfg.run.status = "running"
    cfg.run.pid = 0
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.status == "running"


def test_load_config_marks_inactive_legacy_running_as_failed(tmp_path):
    import os
    import time

    from cvbench.core.config import _LEGACY_STALE_SECONDS

    cfg = build_config("data")
    cfg.run.status = "running"
    cfg.run.pid = 0
    save_config(cfg, str(tmp_path))

    stale_time = time.time() - _LEGACY_STALE_SECONDS - 60
    config_path = tmp_path / "config.yaml"
    os.utime(config_path, (stale_time, stale_time))

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.status == "failed"


# ---------------------------------------------------------------------------
# task field
# ---------------------------------------------------------------------------

def test_build_config_defaults_to_classification_task():
    cfg = build_config("data")
    assert cfg.task == "classification"


def test_build_config_task_override():
    cfg = build_config("data", overrides={"task": "detection"})
    assert cfg.task == "detection"


def test_task_round_trips(tmp_path):
    cfg = build_config("data", overrides={"task": "detection"})
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.task == "detection"


def test_config_yaml_without_task_key_loads_as_classification(tmp_path):
    # Simulates a config.yaml written before the `task` field existed.
    (tmp_path / "config.yaml").write_text("data:\n  data_dir: data\n")

    reloaded = load_config(str(tmp_path))
    assert reloaded.task == "classification"


def test_detection_config_round_trips(tmp_path):
    cfg = build_config("data", overrides={"task": "detection"})
    cfg.detection.strides = [8, 16]
    cfg.detection.anchors = [[[0.1, 0.1], [0.2, 0.2]], [[0.4, 0.4], [0.6, 0.6]]]
    cfg.detection.anchors_per_scale = 2
    cfg.detection.ignore_iou_threshold = 0.4
    cfg.detection.noobj_weight = 0.3
    cfg.detection.conf_threshold = 0.4
    cfg.detection.iou_threshold = 0.6
    cfg.detection.max_detections = 50
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.detection.strides == [8, 16]
    assert reloaded.detection.anchors == [[[0.1, 0.1], [0.2, 0.2]], [[0.4, 0.4], [0.6, 0.6]]]
    assert reloaded.detection.anchors_per_scale == 2
    assert reloaded.detection.ignore_iou_threshold == 0.4
    assert reloaded.detection.noobj_weight == 0.3
    assert reloaded.detection.conf_threshold == 0.4
    assert reloaded.detection.iou_threshold == 0.6
    assert reloaded.detection.max_detections == 50


def test_run_metrics_round_trip(tmp_path):
    cfg = build_config("data")
    cfg.run.primary_metric = "map50"
    cfg.run.primary_value = 0.5
    cfg.run.metrics["val_loss"] = 0.25
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.primary_metric == "map50"
    assert reloaded.run.metrics == {"map50": 0.5, "val_loss": 0.25}
    assert reloaded.run.primary_value == 0.5


def test_legacy_run_keys_migrate_to_metrics(tmp_path):
    cfg = build_config("data")
    save_config(cfg, str(tmp_path))
    path = tmp_path / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["run"] = {
        "name": "old", "val_accuracy": 0.9, "val_loss": 0.3,
        "test_accuracy": 0.5, "test_metric": "map50",
    }
    path.write_text(yaml.safe_dump(raw))

    run = load_config(str(tmp_path)).run
    assert run.metrics == {"val_accuracy": 0.9, "val_loss": 0.3, "map50": 0.5}
    assert run.primary_metric == "map50"
    assert run.primary_value == 0.5
    assert run.metrics.get("val_accuracy") == 0.9


def test_update_run_status_merges_metrics(tmp_path):
    cfg = build_config("data")
    cfg.run.metrics["val_loss"] = 0.4
    save_config(cfg, str(tmp_path))

    update_run_status(str(tmp_path), metrics={"map50": 0.6}, primary_metric="map50")

    run = load_config(str(tmp_path)).run
    assert run.metrics == {"val_loss": 0.4, "map50": 0.6}
    assert run.primary_metric == "map50"


def test_val_split_round_trips(tmp_path):
    # Regression test: the config loader used to silently drop val_split and
    # val_split_explicit, so --from would reset val_split to the 0.2 default.
    cfg = build_config("data", overrides={"data.val_split": 0.3})
    assert cfg.data.val_split == 0.3
    assert cfg.data.val_split_explicit is True
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.data.val_split == 0.3
    assert reloaded.data.val_split_explicit is True


# ---------------------------------------------------------------------------
# Reflective round-trip: every scalar field must survive save_config/load_config.
#
# The loader once silently dropped val_split/val_split_explicit. It is now driven by
# the dataclass fields, and this test keeps it that way for every field present today
# and any added later — including a future task's own config section.
# ---------------------------------------------------------------------------

def _mutate_scalar_fields(obj):
    """Set every scalar leaf field on a (possibly nested) dataclass to a value
    distinct from its default, recursing into nested dataclasses. List/dict
    fields are left alone — they're exercised by dedicated tests elsewhere."""
    import dataclasses

    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if dataclasses.is_dataclass(value):
            _mutate_scalar_fields(value)
        elif isinstance(value, bool):
            setattr(obj, f.name, not value)
        elif isinstance(value, int):
            setattr(obj, f.name, value + 1)
        elif isinstance(value, float):
            setattr(obj, f.name, value + 0.5)
        elif isinstance(value, str):
            setattr(obj, f.name, value + "_x")
        # None, list, dict fields: left untouched.


def test_every_scalar_field_survives_round_trip(tmp_path):
    import dataclasses

    cfg = build_config("data", overrides={"task": "detection"})  # a detection run owns every section
    _mutate_scalar_fields(cfg)
    cfg.task = "detection"

    save_config(cfg, str(tmp_path))
    reloaded = load_config(str(tmp_path))

    assert dataclasses.asdict(reloaded) == dataclasses.asdict(cfg)


# ---------------------------------------------------------------------------
# Generic loader, round trips, overrides
# ---------------------------------------------------------------------------

def _round_trip(cfg, tmp_path):
    save_config(cfg, str(tmp_path))
    reloaded = load_config(str(tmp_path))
    assert reloaded == cfg
    save_config(reloaded, str(tmp_path))  # and the second save is stable too
    assert load_config(str(tmp_path)) == cfg
    return reloaded


def test_full_config_round_trips(tmp_path):
    from cvbench.core.config import OneOfCandidateConfig, OneOfConfig, TransformConfig

    cfg = build_config("data", overrides={
        "training.optimizer.type": "sgd", "training.loss.type": "focal", "training.seed": 7,
        "training.class_weight": {"cat": 2.0}, "training.lr_scheduler.patience": 3,
    })
    cfg.augmentation.transforms = [
        TransformConfig(name="keras_flip", prob=0.5, params={"mode": "horizontal"}),
        OneOfConfig(prob=0.3, candidates=[OneOfCandidateConfig(name="aug_blur", weight=2.0, params={"k": 3})]),
    ]
    cfg.run.metrics = {"val_loss": 0.4, "accuracy": 0.9}
    _round_trip(cfg, tmp_path)


def test_legacy_flat_run_metrics_are_migrated(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "run:\n  name: old\n  status: done\n  val_loss: 0.5\n  val_accuracy: 0.8\n  test_accuracy: 0.7\n"
    )
    cfg = load_config(str(tmp_path))
    assert cfg.run.metrics == {"val_loss": 0.5, "val_accuracy": 0.8, "accuracy": 0.7}
    _round_trip(cfg, tmp_path)


def test_legacy_test_metric_label_names_the_primary_metric(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "task: detection\nrun:\n  status: done\n  test_accuracy: 0.4\n  test_metric: map50\n"
    )
    cfg = load_config(str(tmp_path))
    assert cfg.run.metrics == {"map50": 0.4} and cfg.run.primary_metric == "map50"
    _round_trip(cfg, tmp_path)


def test_detection_block_on_a_classification_run_is_ignored(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "task: classification\ndetection:\n  strides: [8, 16]\n  anchors: []\nrun:\n  status: done\n"
    )
    cfg = load_config(str(tmp_path))
    assert cfg.task == "classification"
    _round_trip(cfg, tmp_path)
    assert "detection" not in yaml.safe_load((tmp_path / "config.yaml").read_text())


def test_detection_section_is_saved_only_for_detection_runs(tmp_path):
    save_config(build_config("data"), str(tmp_path))
    assert "detection" not in yaml.safe_load((tmp_path / "config.yaml").read_text())
    det = build_config("data", overrides={"task": "detection", "detection.conf_threshold": 0.4})
    reloaded = _round_trip(det, tmp_path)
    assert reloaded.detection.conf_threshold == 0.4


def test_unknown_yaml_key_names_the_path_and_suggests_a_fix(tmp_path):
    from cvbench.core.errors import CVBenchError

    (tmp_path / "config.yaml").write_text("training:\n  optimizer:\n    wieght_decay: 0.1\n")
    with pytest.raises(CVBenchError, match=r"wieght_decay.*Did you mean 'training\.optimizer\.weight_decay'"):
        load_config(str(tmp_path))


def test_mistyped_yaml_value_is_rejected(tmp_path):
    from cvbench.core.errors import CVBenchError

    (tmp_path / "config.yaml").write_text("training:\n  epochs: many\n")
    with pytest.raises(CVBenchError, match=r"training\.epochs.*expected int"):
        load_config(str(tmp_path))


def test_apply_overrides_sets_nested_keys_and_parses_text():
    from cvbench.core.config import apply_overrides

    cfg = apply_overrides(build_config("data"), {
        "training.optimizer.weight_decay": "1e-4", "training.seed": "5", "training.interrupt.enabled": "false",
        "model.backbone": "resnet_18", "training.class_weight": "auto",
    })
    assert cfg.training.optimizer.weight_decay == 1e-4
    assert cfg.training.seed == 5 and cfg.training.interrupt.enabled is False
    assert cfg.training.class_weight == "auto"
    assert cfg.model.backbone == "resnet_18"


def test_apply_overrides_none_for_optional_field():
    from cvbench.core.config import apply_overrides

    cfg = build_config("data", overrides={"training.seed": 3})
    assert apply_overrides(cfg, {"training.seed": "null"}).training.seed is None


@pytest.mark.parametrize("key, value, message", [
    ("training.learning_rte", 1e-3, r"Did you mean 'training\.learning_rate'"),
    ("training.epochs", "many", r"expected int"),
    ("training.epochs", True, r"expected int"),
    ("training", 1, r"is a section"),
    ("detection.conf_threshold", 0.3, r"only applies to detection runs"),
    ("task", "segmentation", r"Unknown task"),
])
def test_apply_overrides_rejects_bad_input(key, value, message):
    from cvbench.core.config import apply_overrides
    from cvbench.core.errors import CVBenchError

    with pytest.raises(CVBenchError, match=message):
        apply_overrides(build_config("data"), {key: value})


def test_every_field_of_every_section_can_be_overridden():
    # Guards the "adding a field needs only the dataclass change" promise: nothing is hand-mapped.
    import dataclasses

    from cvbench.core.config import CVBenchConfig, _hints, _is_section, _task_sections, apply_overrides

    def leaves(cls, prefix=""):
        for name, tp in _hints(cls).items():
            if _is_section(tp):
                yield from leaves(tp, f"{prefix}{name}.")
            else:
                yield f"{prefix}{name}"

    cfg = build_config("data", overrides={"task": "detection"})
    task_keys = [k for name, (_, cls) in _task_sections().items() for k in leaves(cls, f"{name}.")]
    assert "detection.conf_threshold" in task_keys
    for key in [*leaves(CVBenchConfig), *task_keys]:
        current = cfg
        for part in key.split("."):
            current = getattr(current, part)
        if not dataclasses.is_dataclass(current):
            apply_overrides(cfg, {key: current})  # re-setting the current value must type-check


def test_keys_removed_from_the_dataclasses_still_load(tmp_path):
    # Older run directories carry settings that no longer exist; they must stay readable.
    (tmp_path / "config.yaml").write_text(
        "task: detection\nmodel:\n  use_lcn: true\ntraining:\n  mixup:\n    alpha: 0.2\ndetection:\n  grid_stride: 4\n"
    )
    assert load_config(str(tmp_path)).task == "detection"


def test_legacy_backbone_explicit_key_still_loads(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n  backbone: resnet_18\n  backbone_explicit: true\n"
    )
    assert load_config(str(tmp_path)).model.backbone == "resnet_18"
