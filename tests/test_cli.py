"""CLI tests using click's CliRunner — no Docker, no real training."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.runs import runs
from cvbench.core.config import build_config, save_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------

def test_runs_list_empty(tmp_path):
    runner = CliRunner()
    result = runner.invoke(runs, ["list", str(tmp_path)])
    assert result.exit_code == 0
    assert "No experiments found" in result.output


def test_runs_list_shows_entries(tmp_path):
    _write_exp(tmp_path, "exp_a", val_accuracy=0.91, epochs_run=10)
    runner = CliRunner()
    result = runner.invoke(runs, ["list", str(tmp_path)])
    assert result.exit_code == 0
    assert "exp_a" in result.output


def test_runs_list_sort_by_val_accuracy(tmp_path):
    _write_exp(tmp_path, "exp_low", val_accuracy=0.7)
    _write_exp(tmp_path, "exp_high", val_accuracy=0.95)
    runner = CliRunner()
    result = runner.invoke(runs, ["list", str(tmp_path), "--sort", "val_accuracy"])
    assert result.exit_code == 0
    assert result.output.index("exp_high") < result.output.index("exp_low")


# ---------------------------------------------------------------------------
# runs best
# ---------------------------------------------------------------------------

def test_runs_best(tmp_path):
    _write_exp(tmp_path, "low", val_accuracy=0.7)
    _write_exp(tmp_path, "high", val_accuracy=0.95)
    runner = CliRunner()
    result = runner.invoke(runs, ["best", str(tmp_path)])
    assert result.exit_code == 0
    assert "high" in result.output


def test_runs_best_no_metric(tmp_path):
    _write_exp(tmp_path, "no_metric")  # val_accuracy stays None
    runner = CliRunner()
    result = runner.invoke(runs, ["best", str(tmp_path)])
    assert result.exit_code == 0
    assert "No experiments" in result.output


# ---------------------------------------------------------------------------
# runs compare
# ---------------------------------------------------------------------------

def test_runs_compare(tmp_path):
    exp_a = _write_exp(tmp_path, "exp_a", val_accuracy=0.8)
    exp_b = _write_exp(tmp_path, "exp_b", val_accuracy=0.9)
    runner = CliRunner()
    result = runner.invoke(runs, ["compare", str(exp_a), str(exp_b)])
    assert result.exit_code == 0
    assert "exp_a" in result.output
    assert "exp_b" in result.output


def test_runs_compare_no_config(tmp_path):
    exp_a = _write_exp(tmp_path, "exp_a")
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(runs, ["compare", str(exp_a), str(empty_dir)])
    assert result.exit_code != 0
    assert "config.yaml" in result.output
