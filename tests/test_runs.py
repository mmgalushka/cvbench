import json
from datetime import date
from pathlib import Path

from cvbench.core.config import build_config, save_config
from cvbench.core.runs import (
    best_experiment,
    make_run_name,
    make_unique_dir,
    scan_experiments,
)

# ---------------------------------------------------------------------------
# Run name generation
# ---------------------------------------------------------------------------

def _make_cfg(backbone="efficientnet_b3", lr=5e-5, task="classification"):
    cfg = build_config("data", backbone=backbone, lr=lr)
    cfg.task = task
    return cfg


def test_run_name_contains_backbone_slug():
    cfg = _make_cfg(backbone="efficientnet_b3")
    name = make_run_name(cfg)
    assert "effnet_b3" in name


def test_run_name_contains_lr_slug():
    cfg = _make_cfg(lr=1e-4)
    name = make_run_name(cfg)
    assert "lr1e4" in name


def test_run_name_contains_date():
    cfg = _make_cfg()
    name = make_run_name(cfg)
    today = date.today().strftime("%Y_%m_%d")
    assert today in name


def test_run_name_format():
    cfg = _make_cfg(backbone="efficientnet_b0", lr=1e-4)
    name = make_run_name(cfg)
    parts = name.split("_")
    assert parts[0] == "cls"
    assert parts[1] == "effnet"
    assert parts[2] == "b0"


def test_run_name_classification_prefix():
    cfg = _make_cfg(task="classification")
    name = make_run_name(cfg)
    assert name.startswith("cls_")


def test_run_name_detection_prefix():
    cfg = _make_cfg(backbone="resnet_18", task="detection")
    name = make_run_name(cfg)
    assert name.startswith("det_")
    assert "resnet_18" in name


def test_run_name_same_backbone_lr_differs_by_task():
    cls_name = make_run_name(_make_cfg(backbone="resnet_18", lr=1e-4, task="classification"))
    det_name = make_run_name(_make_cfg(backbone="resnet_18", lr=1e-4, task="detection"))
    assert cls_name != det_name


# ---------------------------------------------------------------------------
# make_unique_dir
# ---------------------------------------------------------------------------

def test_make_unique_dir_no_conflict(tmp_path):
    result = make_unique_dir(str(tmp_path), "exp_01")
    assert result == tmp_path / "exp_01"


def test_make_unique_dir_with_conflict(tmp_path):
    (tmp_path / "exp_01").mkdir()
    result = make_unique_dir(str(tmp_path), "exp_01")
    assert result == tmp_path / "exp_01_2"


def test_make_unique_dir_multiple_conflicts(tmp_path):
    (tmp_path / "exp_01").mkdir()
    (tmp_path / "exp_01_2").mkdir()
    result = make_unique_dir(str(tmp_path), "exp_01")
    assert result == tmp_path / "exp_01_3"


# ---------------------------------------------------------------------------
# scan_experiments
# ---------------------------------------------------------------------------

def _write_exp(parent: Path, name: str, **run_kwargs) -> Path:
    exp_dir = parent / name
    exp_dir.mkdir(parents=True)
    cfg = build_config("data")
    cfg.run.name = name
    cfg.run.date = "2026-01-01"
    for k, v in run_kwargs.items():
        setattr(cfg.run, k, v)
    save_config(cfg, str(exp_dir))
    return exp_dir


def test_scan_experiments_empty_dir(tmp_path):
    assert scan_experiments(str(tmp_path)) == []


def test_scan_experiments_missing_dir(tmp_path):
    assert scan_experiments(str(tmp_path / "nonexistent")) == []


def test_scan_experiments_skips_dirs_without_config(tmp_path):
    (tmp_path / "not_an_exp").mkdir()
    results = scan_experiments(str(tmp_path))
    assert results == []


def test_scan_experiments_finds_experiments(tmp_path):
    _write_exp(tmp_path, "exp_a", val_accuracy=0.8)
    _write_exp(tmp_path, "exp_b", val_accuracy=0.9)
    results = scan_experiments(str(tmp_path))
    assert len(results) == 2
    names = {r["name"] for r in results}
    assert names == {"exp_a", "exp_b"}


def test_scan_experiments_sort_by_val_accuracy(tmp_path):
    _write_exp(tmp_path, "low", val_accuracy=0.7)
    _write_exp(tmp_path, "high", val_accuracy=0.95)
    results = scan_experiments(str(tmp_path), sort_by="val_accuracy")
    assert results[0]["name"] == "high"


def test_scan_experiments_sort_by_date(tmp_path):
    _write_exp(tmp_path, "old")
    results = scan_experiments(str(tmp_path), sort_by="date")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# best_experiment
# ---------------------------------------------------------------------------

def test_best_experiment_by_val_accuracy(tmp_path):
    _write_exp(tmp_path, "low", val_accuracy=0.7)
    _write_exp(tmp_path, "high", val_accuracy=0.95)
    b = best_experiment(str(tmp_path), "val_accuracy")
    assert b["name"] == "high"


def test_best_experiment_by_val_loss(tmp_path):
    _write_exp(tmp_path, "good", val_loss=0.1)
    _write_exp(tmp_path, "bad", val_loss=0.9)
    b = best_experiment(str(tmp_path), "val_loss")
    assert b["name"] == "good"  # lower loss is better


def test_best_experiment_no_metric(tmp_path):
    _write_exp(tmp_path, "no_metric")  # val_accuracy=None
    b = best_experiment(str(tmp_path), "val_accuracy")
    assert b is None


# ---------------------------------------------------------------------------
# task / test_metric fields, and eval_report.json envelope back-compat
# ---------------------------------------------------------------------------

def test_scan_experiments_reports_task_and_test_metric(tmp_path):
    exp_dir = tmp_path / "det_run"
    exp_dir.mkdir(parents=True)
    cfg = build_config("data", task="detection")
    cfg.run.name = "det_run"
    cfg.run.date = "2026-01-01"
    cfg.run.test_metric = "map50"
    save_config(cfg, str(exp_dir))

    results = scan_experiments(str(tmp_path))
    assert results[0]["task"] == "detection"
    assert results[0]["test_metric"] == "map50"


def test_scan_experiments_defaults_task_to_classification(tmp_path):
    _write_exp(tmp_path, "plain")
    results = scan_experiments(str(tmp_path))
    assert results[0]["task"] == "classification"
    assert results[0]["test_metric"] == "accuracy"


def test_test_accuracy_reads_new_overall_envelope(tmp_path):
    exp_dir = _write_exp(tmp_path, "with_envelope")  # cfg.run.test_accuracy left None
    (exp_dir / "eval_report.json").write_text(json.dumps({
        "task": "classification",
        "overall": {"metric": "accuracy", "value": 0.42, "label": "Overall Accuracy"},
        "overall_accuracy": 0.42,  # legacy mirror, also present
    }))
    results = scan_experiments(str(tmp_path))
    assert results[0]["test_accuracy"] == 0.42


def test_test_accuracy_falls_back_to_legacy_report_shape(tmp_path):
    """A report written before the 'overall' envelope existed (no 'overall' key)
    must still resolve — this is the back-compat path for old runs."""
    exp_dir = _write_exp(tmp_path, "legacy_report")  # cfg.run.test_accuracy left None
    (exp_dir / "eval_report.json").write_text(json.dumps({
        "overall_accuracy": 0.77,
    }))
    results = scan_experiments(str(tmp_path))
    assert results[0]["test_accuracy"] == 0.77


def test_test_accuracy_prefers_config_value_over_report(tmp_path):
    exp_dir = _write_exp(tmp_path, "config_wins", test_accuracy=0.99)
    (exp_dir / "eval_report.json").write_text(json.dumps({
        "overall": {"metric": "accuracy", "value": 0.11, "label": "Overall Accuracy"},
    }))
    results = scan_experiments(str(tmp_path))
    assert results[0]["test_accuracy"] == 0.99
