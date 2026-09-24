"""`runs` commands and name resolution with sweep directories (trials live in experiments/<sweep>/)."""
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from cvbench.cli.runs import runs
from cvbench.core.config import build_config, save_config
from cvbench.core.exp_store import (
    assert_name_available,
    assert_renamable,
    is_sweep_dir,
    resolve_run_dir,
)
from cvbench.core.sweep_store import SweepManifest, scan_sweeps, write_manifest


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


def _sweep(experiments: Path, name="sw", date="2026-02-01") -> Path:
    d = experiments / name
    d.mkdir(parents=True)
    write_manifest(d, SweepManifest(
        version=1, name=name, date=date, data_dir="data", strategy="grid",
        metric="val_loss", direction="min", axes={"lr": ["0.1", "0.01"]},
    ))
    return d


@pytest.fixture
def exps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "experiments"
    d.mkdir()
    return d


def test_is_sweep_dir(exps):
    sweep = _sweep(exps)
    run = _write_exp(exps, "manual")
    assert is_sweep_dir(sweep)
    assert not is_sweep_dir(run)


def test_resolve_trial_by_bare_name(exps):
    sweep = _sweep(exps)
    trial = _write_exp(sweep, "sw_001")
    assert Path(resolve_run_dir("sw_001")).resolve() == trial.resolve()


def test_resolve_top_level_still_wins(exps):
    _sweep(exps)
    run = _write_exp(exps, "manual")
    assert Path(resolve_run_dir("manual")).resolve() == run.resolve()


def test_resolve_unknown_still_errors(exps):
    _sweep(exps)
    with pytest.raises(click.BadParameter):
        resolve_run_dir("nope")


def test_name_conflicts_with_trial(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001")
    with pytest.raises(ValueError, match="trial"):
        assert_name_available("SW_001")
    assert_name_available("fresh")


def test_renamable_guards(exps):
    sweep = _sweep(exps)
    trial = _write_exp(sweep, "sw_001")
    manual = _write_exp(exps, "manual")
    with pytest.raises(ValueError, match="is a sweep"):
        assert_renamable(sweep)
    with pytest.raises(ValueError, match="trial of sweep"):
        assert_renamable(trial)
    assert_renamable(manual)


def test_scan_sweeps_uses_best_trial(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001", val_loss=0.5, epochs_run=3)
    _write_exp(sweep, "sw_002", val_loss=0.2, epochs_run=4)
    (entry,) = scan_sweeps(str(exps))
    assert entry["name"] == "sw"
    assert entry["is_sweep"] is True
    assert entry["val_loss"] == 0.2
    assert entry["epochs_run"] == 4
    assert entry["status"] == "done"


def test_scan_sweeps_running_and_no_result(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001", status="running")
    (entry,) = scan_sweeps(str(exps))
    assert entry["status"] == "running"
    assert entry["val_loss"] is None


def test_runs_list_shows_sweep_row(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001", val_loss=0.25, epochs_run=3)
    _write_exp(exps, "manual", val_loss=0.4)
    result = CliRunner().invoke(runs, ["list"])
    assert result.exit_code == 0
    assert "cls·sweep" in result.output
    assert "sw_001" not in result.output
    assert "manual" in result.output
    assert "0.2500" in result.output
    assert "shows their trials" in result.output


def test_runs_list_sweep_drilldown_shows_trials(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001", val_loss=0.25)
    result = CliRunner().invoke(runs, ["list", str(sweep)])
    assert result.exit_code == 0
    assert "sw_001" in result.output
    by_name = CliRunner().invoke(runs, ["list", "sw"])
    assert by_name.exit_code == 0
    assert "sw_001" in by_name.output
    assert "sweep" not in result.output.replace("sw_001", "")


def test_runs_show_trial_by_bare_name(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001", val_loss=0.25)
    result = CliRunner().invoke(runs, ["show", "sw_001"])
    assert result.exit_code == 0, result.output


def test_runs_rename_refuses_sweep_and_trial(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001")
    r1 = CliRunner().invoke(runs, ["rename", "sw", "other"])
    r2 = CliRunner().invoke(runs, ["rename", "sw_001", "other"])
    assert r1.exit_code != 0 and "is a sweep" in r1.output
    assert r2.exit_code != 0 and "trial of sweep" in r2.output
    assert (sweep / "sw_001").is_dir()


def test_runs_delete_sweep_labels_and_removes(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001")
    _write_exp(sweep, "sw_002")
    result = CliRunner().invoke(runs, ["delete", "sw", "--yes"])
    assert result.exit_code == 0
    assert "sweep 'sw' and its 2 trial(s)" in result.output
    assert not sweep.exists()


def test_runs_show_sweep_prints_summary_not_error(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001", val_loss=0.5)
    _write_exp(sweep, "sw_002", val_loss=0.2)
    result = CliRunner().invoke(runs, ["show", "sw"])
    assert result.exit_code == 0, result.output
    assert "No config.yaml" not in result.output
    assert "sw_002" in result.output
    assert "Best:" in result.output


def test_runs_show_sweep_by_path(exps):
    sweep = _sweep(exps)
    _write_exp(sweep, "sw_001", val_loss=0.5)
    result = CliRunner().invoke(runs, ["show", str(sweep)])
    assert result.exit_code == 0, result.output


def test_resolve_run_dir_rejects_sweep_unless_allowed(exps):
    _sweep(exps)
    with pytest.raises(click.BadParameter, match="is a sweep"):
        resolve_run_dir("sw")
    assert resolve_run_dir("sw", allow_sweep=True).endswith("sw")
