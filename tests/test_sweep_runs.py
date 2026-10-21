"""`runs` commands and name resolution with sweep directories (trials live in experiments/<sweep>/)."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.runs import runs
from cvbench.core.errors import AmbiguousNameError, ConfigError, ConflictError, NotFoundError, WrongKindError
from cvbench.core.exp_store import (
    assert_name_available,
    assert_renamable,
    is_sweep_dir,
    resolve_run_dir,
)
from cvbench.core.sweep_store import scan_sweeps


def test_is_sweep_dir(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    run = write_exp(exps, "manual")
    assert is_sweep_dir(sweep)
    assert not is_sweep_dir(run)


def test_resolve_trial_by_bare_name(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    trial = write_exp(sweep, "sw_001")
    assert Path(resolve_run_dir("sw_001")).resolve() == trial.resolve()


def test_resolve_top_level_still_wins(exps, write_exp, make_sweep):
    make_sweep(exps)
    run = write_exp(exps, "manual")
    assert Path(resolve_run_dir("manual")).resolve() == run.resolve()


def test_resolve_unknown_still_errors(exps, make_sweep):
    make_sweep(exps)
    with pytest.raises(NotFoundError):
        resolve_run_dir("nope")


def test_name_conflicts_with_trial(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001")
    with pytest.raises(ConflictError, match="trial"):
        assert_name_available("SW_001")
    assert_name_available("fresh")


def test_renamable_guards(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    trial = write_exp(sweep, "sw_001")
    manual = write_exp(exps, "manual")
    with pytest.raises(ConfigError, match="is a sweep"):
        assert_renamable(sweep)
    with pytest.raises(ConfigError, match="trial of sweep"):
        assert_renamable(trial)
    assert_renamable(manual)


def test_scan_sweeps_uses_best_trial(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001", val_loss=0.5, epochs_run=3)
    write_exp(sweep, "sw_002", val_loss=0.2, epochs_run=4)
    (entry,) = scan_sweeps(str(exps))
    assert entry["name"] == "sw"
    assert entry["is_sweep"] is True
    assert entry["val_loss"] == 0.2
    assert entry["epochs_run"] == 4
    assert entry["status"] == "done"


def test_scan_sweeps_running_and_no_result(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001", status="running")
    (entry,) = scan_sweeps(str(exps))
    assert entry["status"] == "running"
    assert entry["val_loss"] is None


def test_runs_list_shows_sweep_row(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001", val_loss=0.25, epochs_run=3)
    write_exp(exps, "manual", val_loss=0.4)
    result = CliRunner().invoke(runs, ["list"])
    assert result.exit_code == 0
    assert "cls·sweep" in result.output
    assert "sw_001" not in result.output
    assert "manual" in result.output
    assert "0.2500" in result.output
    assert "shows their trials" in result.output


def test_runs_list_sweep_drilldown_shows_trials(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001", val_loss=0.25)
    result = CliRunner().invoke(runs, ["list", str(sweep)])
    assert result.exit_code == 0
    assert "sw_001" in result.output
    by_name = CliRunner().invoke(runs, ["list", "sw"])
    assert by_name.exit_code == 0
    assert "sw_001" in by_name.output
    assert "sweep" not in result.output.replace("sw_001", "")


def test_runs_show_trial_by_bare_name(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001", val_loss=0.25)
    result = CliRunner().invoke(runs, ["show", "sw_001"])
    assert result.exit_code == 0, result.output


def test_runs_rename_refuses_sweep_and_trial(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001")
    r1 = CliRunner().invoke(runs, ["rename", "sw", "other"])
    r2 = CliRunner().invoke(runs, ["rename", "sw_001", "other"])
    assert r1.exit_code != 0 and "is a sweep" in r1.output
    assert r2.exit_code != 0 and "trial of sweep" in r2.output
    assert (sweep / "sw_001").is_dir()


def test_runs_delete_sweep_labels_and_removes(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001")
    write_exp(sweep, "sw_002")
    result = CliRunner().invoke(runs, ["delete", "sw", "--yes"])
    assert result.exit_code == 0
    assert "sweep 'sw' and its 2 trial(s)" in result.output
    assert not sweep.exists()


def test_runs_show_sweep_prints_summary_not_error(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001", val_loss=0.5)
    write_exp(sweep, "sw_002", val_loss=0.2)
    result = CliRunner().invoke(runs, ["show", "sw"])
    assert result.exit_code == 0, result.output
    assert "No config.yaml" not in result.output
    assert "sw_002" in result.output
    assert "Best:" in result.output


def test_runs_show_sweep_by_path(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001", val_loss=0.5)
    result = CliRunner().invoke(runs, ["show", str(sweep)])
    assert result.exit_code == 0, result.output


def test_resolve_run_dir_rejects_sweep_unless_allowed(exps, make_sweep):
    make_sweep(exps)
    with pytest.raises(WrongKindError, match="is a sweep"):
        resolve_run_dir("sw")
    assert resolve_run_dir("sw", allow_sweep=True).endswith("sw")


@pytest.mark.parametrize("args", [["show", "nosuch"], ["rename", "nosuch", "x"], ["delete", "nosuch", "--yes"]])
def test_runs_unknown_run_exits_2_and_points_to_runs_list(exps, args):
    result = CliRunner().invoke(runs, args)
    assert result.exit_code == 2
    assert "Run not found: 'nosuch'" in result.output
    assert "runs list" in result.output
    assert "also tried" not in result.output.lower()


def test_runs_unknown_run_suggests_close_run_and_trial(exps, write_exp, make_sweep):
    write_exp(exps, "resnet_run")
    sweep = make_sweep(exps)
    write_exp(sweep, "sw_001")
    assert "Did you mean 'resnet_run'?" in CliRunner().invoke(runs, ["show", "resnet_rum"]).output
    assert "Did you mean 'sw_001'?" in CliRunner().invoke(runs, ["show", "sw_01"]).output


# ---------------------------------------------------------------------------
# prefix lookup
# ---------------------------------------------------------------------------

def test_resolve_unique_prefix(exps, write_exp):
    run = write_exp(exps, "cls_resnet_lr1e4_2026_03_28")
    write_exp(exps, "det_resnet_lr1e4_2026_03_28")
    assert Path(resolve_run_dir("cls_")).resolve() == run.resolve()


def test_resolve_exact_name_wins_over_longer_names(exps, write_exp):
    short = write_exp(exps, "run")
    write_exp(exps, "run_2")
    assert Path(resolve_run_dir("run")).resolve() == short.resolve()


def test_resolve_ambiguous_prefix_lists_matches(exps, write_exp):
    write_exp(exps, "cls_a")
    write_exp(exps, "cls_b")
    with pytest.raises(AmbiguousNameError) as exc:
        resolve_run_dir("cls_")
    assert exc.value.candidates == ["cls_a", "cls_b"]
    assert "cls_a, cls_b" in exc.value.message


def test_resolve_sweep_trial_by_prefix(exps, write_exp, make_sweep):
    sweep = make_sweep(exps)
    trial = write_exp(sweep, "sw_001")
    write_exp(sweep, "other_002")
    assert Path(resolve_run_dir("sw_0")).resolve() == trial.resolve()


def test_resolve_prefix_matching_sweep_is_rejected_as_wrong_kind(exps, make_sweep):
    make_sweep(exps, name="sweep_one")
    with pytest.raises(WrongKindError):
        resolve_run_dir("sweep_o")
    assert resolve_run_dir("sweep_o", allow_sweep=True).endswith("sweep_one")


def test_resolve_no_prefix_match_suggests_close_name(exps, write_exp):
    write_exp(exps, "resnet_run")
    with pytest.raises(NotFoundError) as exc:
        resolve_run_dir("resnet_rum")
    assert exc.value.suggestion == "resnet_run"


def test_cli_maps_domain_error_to_usage_error(exps, write_exp):
    write_exp(exps, "cls_a")
    write_exp(exps, "cls_b")
    result = CliRunner().invoke(runs, ["show", "cls_"])
    assert result.exit_code == 2
    assert "matches 2 runs" in result.output
    result = CliRunner().invoke(runs, ["show", "zzz"])
    assert result.exit_code == 2
    assert "Run not found: 'zzz'" in result.output


def test_runs_list_never_truncates_run_names(exps, write_exp):
    name = "cls_efficientnet_b0_lr1e4_2026_03_28_extra_long_suffix_here"
    write_exp(exps, name)
    result = CliRunner().invoke(runs, ["list"])
    assert "…" not in result.output
    lines = result.output.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith("Run")) + 2
    width = lines[first - 2].index("Task")
    assert "".join(ln[:width].strip() for ln in lines[first:]) == name
