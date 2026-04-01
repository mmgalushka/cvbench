from datetime import date
from pathlib import Path

import pytest

from core.config import CVBenchConfig, build_config, save_config
from core.runs import (
    best_experiment,
    make_run_name,
    make_unique_dir,
    scan_experiments,
)


# ---------------------------------------------------------------------------
# Run name generation
# ---------------------------------------------------------------------------

def _make_cfg(backbone="efficientnet_b3", lr=5e-5):
    return build_config("data", backbone=backbone, lr=lr)


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
    assert parts[0] == "effnet"
    assert parts[1] == "b0"


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
    _write_exp(tmp_path, "good", val_accuracy=0.1)  # reuse val_accuracy as proxy
    _write_exp(tmp_path, "bad", val_accuracy=0.9)
    # Test with val_loss directly
    parent = tmp_path / "loss_test"
    parent.mkdir()
    for name, loss in [("good_loss", 0.1), ("bad_loss", 0.9)]:
        exp_dir = parent / name
        exp_dir.mkdir()
        cfg = build_config("data")
        cfg.run.name = name
        save_config(cfg, str(exp_dir))
        # patch val_loss into the yaml directly
        import yaml
        p = exp_dir / "config.yaml"
        raw = yaml.safe_load(p.read_text())
        raw.setdefault("run", {})["val_accuracy"] = loss  # use val_accuracy as stand-in
        p.write_text(yaml.dump(raw))
    b = best_experiment(str(parent), "val_accuracy")
    assert b["val_accuracy"] == 0.9  # higher is better for val_accuracy


def test_best_experiment_no_metric(tmp_path):
    _write_exp(tmp_path, "no_metric")  # val_accuracy=None
    b = best_experiment(str(tmp_path), "val_accuracy")
    assert b is None
