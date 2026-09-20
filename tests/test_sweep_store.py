"""Sweep store tests — axis parsing, grid expansion, manifest, summary (no TensorFlow)."""
from pathlib import Path

import pytest

from cvbench.core.config import build_config, save_config
from cvbench.core.sweep_store import (
    SweepError,
    SweepManifest,
    best_trial,
    default_metric,
    expand_grid,
    metric_direction,
    read_manifest,
    split_axis_values,
    summarize,
    trial_dir_name,
    validate_sweep_name,
    write_manifest,
)


def _write_exp(parent: Path, name: str, **run_kwargs) -> Path:
    exp_dir = parent / name
    exp_dir.mkdir(parents=True)
    cfg = build_config("data")
    cfg.run.name = name
    cfg.run.date = "2026-01-01"
    cfg.run.status = "done"
    for k, v in run_kwargs.items():
        setattr(cfg.run, k, v)
    save_config(cfg, str(exp_dir))
    return exp_dir


def _manifest(name="sw", metric="val_loss", direction="min", axes=None) -> SweepManifest:
    return SweepManifest(
        version=1, name=name, date="2026-01-01", data_dir="data", strategy="grid",
        metric=metric, direction=direction, axes=axes or {"lr": ["0.1", "0.01", "0.001"]},
    )


def _sweep(tmp_path, **kw) -> Path:
    m = _manifest(**kw)
    d = tmp_path / m.name
    d.mkdir()
    write_manifest(d, m)
    return d


# ---------------------------------------------------------------------------
# split_axis_values
# ---------------------------------------------------------------------------

def test_split_plain_list():
    assert split_axis_values("lr", "1e-3,1e-4") == ["1e-3", "1e-4"]


def test_split_single_value():
    assert split_axis_values("backbone", "resnet50") == ["resnet50"]


def test_split_strips_whitespace():
    assert split_axis_values("backbone", " a , b ") == ["a", "b"]


@pytest.mark.parametrize("raw", ["a,,b", "a,", ",a", ""])
def test_split_empty_item_error(raw):
    with pytest.raises(SweepError, match="empty"):
        split_axis_values("backbone", raw)


def test_split_optimizer_rejoins_params():
    assert split_axis_values("optimizer", "adam,sgd:momentum=0.9,weight_decay=1e-4") == [
        "adam",
        "sgd:momentum=0.9,weight_decay=1e-4",
    ]


def test_split_optimizer_multiple_specs():
    assert split_axis_values("optimizer", "sgd:momentum=0.9,adamw:weight_decay=1e-4") == [
        "sgd:momentum=0.9",
        "adamw:weight_decay=1e-4",
    ]


def test_split_loss_rejoins_params():
    assert split_axis_values("loss", "cross_entropy,focal:gamma=2.0,alpha=0.25") == [
        "cross_entropy",
        "focal:gamma=2.0,alpha=0.25",
    ]


def test_split_lr_scheduler_rejoins_params():
    assert split_axis_values("lr_scheduler", "patience=5,factor=0.5") == ["patience=5,factor=0.5"]


def test_split_lr_scheduler_repeated_key_starts_new_item():
    assert split_axis_values("lr_scheduler", "patience=5,factor=0.5,patience=10") == [
        "patience=5,factor=0.5",
        "patience=10",
    ]


def test_split_class_weight_json_intact():
    raw = '{"a": 1.0, "b": 2.0},balanced'
    assert split_axis_values("class_weight", raw) == ['{"a": 1.0, "b": 2.0}', "balanced"]


@pytest.mark.parametrize("flag", ["lr", "epochs", "dropout"])
def test_split_numeric_range_rejected(flag):
    with pytest.raises(SweepError, match="range"):
        split_axis_values(flag, "0.1:0.5")


def test_split_unknown_flag():
    with pytest.raises(SweepError, match="Unknown sweep flag"):
        split_axis_values("bogus", "1,2")


# ---------------------------------------------------------------------------
# expand_grid / names
# ---------------------------------------------------------------------------

def test_expand_grid_count_and_order():
    grid = expand_grid({"lr": ["1", "2"], "backbone": ["a", "b", "c"]})
    assert len(grid) == 6
    assert grid[0] == {"lr": "1", "backbone": "a"}
    assert grid[1] == {"lr": "1", "backbone": "b"}  # last axis fastest
    assert grid[3] == {"lr": "2", "backbone": "a"}
    assert grid[-1] == {"lr": "2", "backbone": "c"}


def test_expand_grid_no_axes():
    with pytest.raises(SweepError, match="at least one axis"):
        expand_grid({})


def test_expand_grid_empty_axis():
    with pytest.raises(SweepError, match="no values"):
        expand_grid({"lr": []})


def test_expand_grid_duplicates():
    with pytest.raises(SweepError, match="duplicate"):
        expand_grid({"lr": ["1", "1"]})


def test_trial_dir_name():
    assert trial_dir_name("shapes_lr", 3) == "shapes_lr_003"
    assert trial_dir_name("s", 1234) == "s_1234"


def test_validate_sweep_name_ok():
    validate_sweep_name("shapes_lr_backbone")


@pytest.mark.parametrize("name", ["", "a/b", "..", "a b"])
def test_validate_sweep_name_bad(name):
    with pytest.raises(SweepError):
        validate_sweep_name(name)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_default_metric():
    assert default_metric("classification") == ("val_loss", "min")
    assert default_metric("detection") == ("map50", "max")


def test_metric_direction():
    assert metric_direction("val_loss") == "min"
    assert metric_direction("val_accuracy") == "max"
    assert metric_direction("map50") == "max"


def test_metric_direction_unknown():
    with pytest.raises(SweepError, match="Unknown metric"):
        metric_direction("f1")


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_manifest_round_trip(tmp_path):
    m = _manifest(axes={"lr": ["1e-3", "1e-4"], "epochs": ["1", "2"]})
    write_manifest(tmp_path, m)
    assert read_manifest(tmp_path) == m


def test_manifest_values_stringified(tmp_path):
    m = _manifest(axes={"epochs": [1, 2]})  # type: ignore[list-item]
    write_manifest(tmp_path, m)
    assert read_manifest(tmp_path).axes == {"epochs": ["1", "2"]}


def test_manifest_missing(tmp_path):
    with pytest.raises(SweepError, match="Not a sweep directory"):
        read_manifest(tmp_path)


def test_manifest_bad_version(tmp_path):
    (tmp_path / "sweep.yaml").write_text("version: 99\nname: x\n")
    with pytest.raises(SweepError, match="version"):
        read_manifest(tmp_path)


def test_manifest_not_a_mapping(tmp_path):
    (tmp_path / "sweep.yaml").write_text("- a\n- b\n")
    with pytest.raises(SweepError, match="mapping"):
        read_manifest(tmp_path)


def test_manifest_missing_field(tmp_path):
    (tmp_path / "sweep.yaml").write_text("version: 1\nname: x\n")
    with pytest.raises(SweepError, match="Invalid"):
        read_manifest(tmp_path)


# ---------------------------------------------------------------------------
# summarize / best_trial
# ---------------------------------------------------------------------------

def test_summarize_min_marks_best(tmp_path):
    d = _sweep(tmp_path)
    _write_exp(d, "sw_001", val_loss=0.5)
    _write_exp(d, "sw_002", val_loss=0.2)
    _write_exp(d, "sw_003", val_loss=0.9)
    manifest, rows = summarize(d)
    assert manifest.name == "sw"
    assert [r.index for r in rows] == [1, 2, 3]
    assert [r.params for r in rows] == [{"lr": "0.1"}, {"lr": "0.01"}, {"lr": "0.001"}]
    assert [r.is_best for r in rows] == [False, True, False]
    assert best_trial(rows).dir == "sw_002"


def test_summarize_max_direction(tmp_path):
    d = _sweep(tmp_path, metric="val_accuracy", direction="max")
    _write_exp(d, "sw_001", val_accuracy=0.6)
    _write_exp(d, "sw_002", val_accuracy=0.95)
    _write_exp(d, "sw_003", val_accuracy=0.7)
    _, rows = summarize(d)
    assert best_trial(rows).dir == "sw_002"


def test_summarize_failed_never_best(tmp_path):
    d = _sweep(tmp_path)
    _write_exp(d, "sw_001", status="failed", val_loss=0.01)
    _write_exp(d, "sw_002", val_loss=0.8)
    _write_exp(d, "sw_003", status="failed")
    _, rows = summarize(d)
    assert [r.status for r in rows] == ["failed", "done", "failed"]
    assert best_trial(rows).dir == "sw_002"


def test_summarize_no_best_when_all_failed(tmp_path):
    d = _sweep(tmp_path)
    _write_exp(d, "sw_001", status="failed")
    _, rows = summarize(d)
    assert best_trial(rows) is None


def test_summarize_missing_and_unreadable(tmp_path):
    d = _sweep(tmp_path)
    _write_exp(d, "sw_001", val_loss=0.4)
    (d / "sw_002").mkdir()  # no config.yaml
    # sw_003 never created
    _, rows = summarize(d)
    assert [r.status for r in rows] == ["done", "unreadable", "missing"]
    assert rows[1].value is None and rows[2].value is None
    assert best_trial(rows).dir == "sw_001"


def test_summarize_done_without_metric_not_best(tmp_path):
    d = _sweep(tmp_path)
    _write_exp(d, "sw_001")  # val_loss None
    _write_exp(d, "sw_002", val_loss=0.3)
    _, rows = summarize(d)
    assert rows[0].value is None
    assert best_trial(rows).dir == "sw_002"


def test_summarize_map50_uses_test_metric(tmp_path):
    d = _sweep(tmp_path, metric="map50", direction="max", axes={"lr": ["1", "2", "3"]})
    _write_exp(d, "sw_001", test_metric="map50", test_accuracy=0.3)
    _write_exp(d, "sw_002", test_metric="map50", test_accuracy=0.7)
    _write_exp(d, "sw_003", test_metric="accuracy", test_accuracy=0.99)  # wrong metric: ignored
    _, rows = summarize(d)
    assert [r.value for r in rows] == [0.3, 0.7, None]
    assert best_trial(rows).dir == "sw_002"


def test_summarize_two_axes_params(tmp_path):
    d = _sweep(tmp_path, axes={"lr": ["1", "2"], "backbone": ["a", "b"]})
    for i in range(1, 5):
        _write_exp(d, f"sw_{i:03d}", val_loss=float(i))
    _, rows = summarize(d)
    assert rows[1].params == {"lr": "1", "backbone": "b"}
    assert best_trial(rows).index == 1
