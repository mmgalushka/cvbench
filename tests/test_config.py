from pathlib import Path

import pytest
import yaml

from cvbench.core.config import (
    CVBenchConfig,
    build_config,
    load_config,
    save_config,
    update_run_status,
)


def test_build_config_defaults():
    cfg = build_config("data")
    assert cfg.data.data_dir == "data"
    assert cfg.data.train_dir == "data/train"
    assert cfg.data.val_dir == "data/val"
    assert cfg.data.test_dir == "data/test"
    assert cfg.model.backbone == "efficientnet_b0"
    assert cfg.training.epochs == 10
    assert cfg.training.learning_rate == 1e-4
    assert cfg.augmentation.transforms == []


def test_build_config_cli_overrides():
    cfg = build_config(
        "data",
        backbone="efficientnet_b3",
        epochs=20,
        lr=5e-5,
        batch_size=32,
        dropout=0.3,
    )
    assert cfg.model.backbone == "efficientnet_b3"
    assert cfg.training.epochs == 20
    assert cfg.training.learning_rate == 5e-5
    assert cfg.data.batch_size == 32
    assert cfg.model.dropout == 0.3


def test_build_config_from_dir(tmp_path):
    # Create an existing experiment with a config.yaml
    cfg_orig = build_config("data", backbone="efficientnet_b3", epochs=50)
    cfg_orig.run.name = "test_exp"
    save_config(cfg_orig, str(tmp_path))

    # Load from that dir as baseline, override just epochs
    cfg_new = build_config("data2", from_dir=str(tmp_path), epochs=5)
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
    cfg = build_config("data", backbone="efficientnet_b0", epochs=5)
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
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path))


def test_update_run_status(tmp_path):
    cfg = build_config("data")
    save_config(cfg, str(tmp_path))

    update_run_status(str(tmp_path), status="done", epochs_run=10, val_accuracy=0.92)

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.status == "done"
    assert reloaded.run.epochs_run == 10
    assert reloaded.run.val_accuracy == 0.92


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


# ---------------------------------------------------------------------------
# task field
# ---------------------------------------------------------------------------

def test_build_config_defaults_to_classification_task():
    cfg = build_config("data")
    assert cfg.task == "classification"


def test_build_config_task_override():
    cfg = build_config("data", task="detection")
    assert cfg.task == "detection"


def test_task_round_trips(tmp_path):
    cfg = build_config("data", task="detection")
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.task == "detection"


def test_config_yaml_without_task_key_loads_as_classification(tmp_path):
    # Simulates a config.yaml written before the `task` field existed.
    (tmp_path / "config.yaml").write_text("data:\n  data_dir: data\n")

    reloaded = load_config(str(tmp_path))
    assert reloaded.task == "classification"


def test_detection_config_round_trips(tmp_path):
    cfg = build_config("data", task="detection")
    cfg.detection.grid_stride = 8
    cfg.detection.conf_threshold = 0.4
    cfg.detection.iou_threshold = 0.6
    cfg.detection.max_detections = 50
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.detection.grid_stride == 8
    assert reloaded.detection.conf_threshold == 0.4
    assert reloaded.detection.iou_threshold == 0.6
    assert reloaded.detection.max_detections == 50


def test_test_metric_round_trips(tmp_path):
    cfg = build_config("data")
    cfg.run.test_metric = "map50"
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.run.test_metric == "map50"


def test_val_split_round_trips(tmp_path):
    # Regression test: _dict_to_config used to silently drop val_split and
    # val_split_explicit, so --from would reset val_split to the 0.2 default.
    cfg = build_config("data", val_split=0.3)
    assert cfg.data.val_split == 0.3
    assert cfg.data.val_split_explicit is True
    save_config(cfg, str(tmp_path))

    reloaded = load_config(str(tmp_path))
    assert reloaded.data.val_split == 0.3
    assert reloaded.data.val_split_explicit is True


# ---------------------------------------------------------------------------
# Reflective round-trip: every scalar field must survive save_config/load_config.
#
# _dict_to_config builds each section explicitly from a raw dict and silently
# falls back to the default for any key it forgets to read (it already lost
# val_split/val_split_explicit this way once). This test makes that bug class
# structurally impossible to reintroduce, for every field present today and
# any added later — including a future task's own config section.
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

    cfg = build_config("data")
    _mutate_scalar_fields(cfg)

    save_config(cfg, str(tmp_path))
    reloaded = load_config(str(tmp_path))

    assert dataclasses.asdict(reloaded) == dataclasses.asdict(cfg)
