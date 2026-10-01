"""`cvbench sweep` CLI tests — training is mocked, so no TensorFlow is needed."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from cvbench.cli.generate import generate
from cvbench.cli.sweep import sweep
from cvbench.core.config import build_config, save_config


@pytest.fixture
def data_dir(tmp_path) -> Path:
    out = tmp_path / "data"
    result = CliRunner().invoke(
        generate, [str(out), "--train", "1", "--val", "1", "--test", "1", "--image-size", "32"]
    )
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def _install_fake(monkeypatch, fail_lr=None, interrupt_lr=None):
    """Replace run_training with a fake that writes a valid experiment dir.

    val_loss = lr * 1000 (so the smallest lr wins). If `fail_lr` matches, the trial is
    recorded as failed and the fake raises, like the real service does. If `interrupt_lr`
    matches, the trial is recorded as "interrupted" and the fake raises KeyboardInterrupt,
    like Ctrl-C under `interrupt_enabled=False` (see services/training.py).
    """
    calls: list[dict] = []

    def fake_run_training(data_dir, output_dir=None, **kwargs):
        calls.append({"data_dir": data_dir, "output_dir": output_dir, **kwargs})
        out = Path(output_dir)
        out.mkdir(parents=True)
        cfg = build_config(str(data_dir))
        cfg.run.name = out.name
        cfg.run.date = "2026-01-01"
        if kwargs.get("lr") is not None:
            cfg.training.learning_rate = kwargs["lr"]
        if kwargs.get("backbone"):
            cfg.model.backbone = kwargs["backbone"]
        lr = kwargs.get("lr") or 1.0
        if fail_lr is not None and lr == fail_lr:
            cfg.run.status = "failed"
            save_config(cfg, str(out))
            raise RuntimeError("boom")
        if interrupt_lr is not None and lr == interrupt_lr:
            cfg.run.status = "interrupted"
            save_config(cfg, str(out))
            raise KeyboardInterrupt()
        cfg.run.status = "done"
        cfg.run.val_loss = lr * 1000
        save_config(cfg, str(out))
        return str(out)

    monkeypatch.setattr("cvbench.services.training.run_training", fake_run_training)
    return calls


def _run(*args):
    return CliRunner().invoke(sweep, [str(a) for a in args])


# ---------------------------------------------------------------------------
# help / imports
# ---------------------------------------------------------------------------

def test_help():
    result = CliRunner().invoke(sweep, ["--help"])
    assert result.exit_code == 0
    assert "sweep" in result.output.lower()


def test_import_does_not_pull_tensorflow():
    code = "import sys\nimport cvbench.cli.sweep\nassert 'tensorflow' not in sys.modules\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# --show
# ---------------------------------------------------------------------------

def test_show_prints_trials_and_creates_nothing(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    result = _run(data_dir, "--lr", "1e-3,1e-4", "--backbone", "a,b,c", "--name", "s1", "--show")
    assert result.exit_code == 0, result.output
    assert "6" in result.output  # trial count
    assert not (workdir / "experiments").exists()
    assert calls == []


# ---------------------------------------------------------------------------
# grid run
# ---------------------------------------------------------------------------

def test_grid_run_creates_layout(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    result = _run(data_dir, "--lr", "1e-3,1e-4", "--epochs", "2", "--name", "s1")
    assert result.exit_code == 0, result.output

    sweep_dir = workdir / "experiments" / "s1"
    assert (sweep_dir / "sweep.yaml").is_file()
    assert (sweep_dir / "s1_001" / "config.yaml").is_file()
    assert (sweep_dir / "s1_002" / "config.yaml").is_file()
    assert not (sweep_dir / "s1_003").exists()
    assert not (sweep_dir / "config.yaml").exists()

    assert [Path(c["output_dir"]).name for c in calls] == ["s1_001", "s1_002"]
    assert [c["lr"] for c in calls] == [1e-3, 1e-4]
    assert all(c["epochs"] == 2 for c in calls)  # single-valued flag applies to every trial
    # Ctrl-C must stop the whole sweep, not just finish the current trial's batch and
    # move on — see services/training.py's `interrupt_enabled` and test_training_service.py.
    assert all(c["interrupt_enabled"] is False for c in calls)


def test_manifest_axes_only_multi_valued(data_dir, workdir, monkeypatch):
    _install_fake(monkeypatch)
    result = _run(data_dir, "--lr", "1e-3,1e-4", "--epochs", "2", "--name", "s1")
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load((workdir / "experiments" / "s1" / "sweep.yaml").read_text())
    assert manifest["name"] == "s1"
    assert manifest["strategy"] == "grid"
    assert manifest["metric"] == "val_loss"
    assert manifest["direction"] == "min"
    assert list(manifest["axes"]) == ["lr"]
    assert manifest["axes"]["lr"] == ["1e-3", "1e-4"]


def test_summary_marks_best(data_dir, workdir, monkeypatch):
    _install_fake(monkeypatch)
    result = _run(data_dir, "--lr", "1e-3,1e-4", "--name", "s1")
    assert result.exit_code == 0, result.output
    assert "best" in result.output.lower()
    assert "s1_002" in result.output  # lr=1e-4 has the lowest val_loss


def test_failed_trial_does_not_abort(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch, fail_lr=1e-3)
    result = _run(data_dir, "--lr", "1e-3,1e-4,1e-5", "--name", "s1")
    assert len(calls) == 3  # all trials attempted
    assert "failed" in result.output.lower()
    sweep_dir = workdir / "experiments" / "s1"
    cfg = yaml.safe_load((sweep_dir / "s1_001" / "config.yaml").read_text())
    assert cfg["run"]["status"] == "failed"
    assert (sweep_dir / "s1_003" / "config.yaml").is_file()
    assert "s1_003" in result.output  # best among the successful trials


def test_interrupted_trial_stops_the_sweep(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch, interrupt_lr=1e-4)
    result = _run(data_dir, "--lr", "1e-3,1e-4,1e-5", "--name", "s1")
    assert len(calls) == 2  # stopped right after the interrupted trial; s1_003 never ran
    assert "interrupted" in result.output.lower()
    sweep_dir = workdir / "experiments" / "s1"
    cfg = yaml.safe_load((sweep_dir / "s1_002" / "config.yaml").read_text())
    assert cfg["run"]["status"] == "interrupted"
    assert not (sweep_dir / "s1_003").exists()


def test_interrupted_trial_skips_map50_eval_and_stops(data_dir, workdir, monkeypatch):
    # An interrupted trial raises before returning, so the map50 eval step (which needs
    # the trial's best.keras) must never run for it.
    calls = _install_fake(monkeypatch, interrupt_lr=1e-4)
    monkeypatch.setattr("cvbench.datasets.layout.detect_task_name", lambda d: "detection")
    eval_calls = []
    monkeypatch.setattr("cvbench.cli.sweep._evaluate_on_val", lambda trial_dir: eval_calls.append(trial_dir))

    result = _run(data_dir, "--lr", "1e-3,1e-4,1e-5", "--metric", "map50", "--name", "s1")

    assert len(calls) == 2  # stopped right after the interrupted trial
    assert "interrupted" in result.output.lower()
    sweep_dir = workdir / "experiments" / "s1"
    assert eval_calls == ["experiments/s1/s1_001"]  # only the successfully done trial was evaluated
    assert not (sweep_dir / "s1_003").exists()


def test_map50_eval_failure_does_not_mark_training_failed(data_dir, workdir, monkeypatch):
    # A trial can train fine (status "done") but have no best.keras if its monitored
    # metric never improved. That must be reported as a val-evaluation failure, and the
    # trial's own "done" training status must not be overwritten or misreported, and the
    # sweep must still continue to the remaining trials (this isn't a Ctrl-C).
    calls = _install_fake(monkeypatch)
    monkeypatch.setattr("cvbench.datasets.layout.detect_task_name", lambda d: "detection")

    def _boom(trial_dir):
        raise ValueError(f"File not found: filepath={trial_dir}/best.keras")

    monkeypatch.setattr("cvbench.cli.sweep._evaluate_on_val", _boom)

    result = _run(data_dir, "--lr", "1e-3,1e-4", "--metric", "map50", "--name", "s1")

    assert len(calls) == 2  # both trials still attempted
    assert "trained but val evaluation failed" in result.output.lower()
    assert "trial s1_001 failed" not in result.output.lower()
    sweep_dir = workdir / "experiments" / "s1"
    cfg = yaml.safe_load((sweep_dir / "s1_001" / "config.yaml").read_text())
    assert cfg["run"]["status"] == "done"  # training itself was never touched


# ---------------------------------------------------------------------------
# random run
# ---------------------------------------------------------------------------

def test_random_run_samples_trials(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    result = _run(data_dir, "--strategy", "random", "--trials", "3", "--sample-seed", "42", "--name", "s1")
    assert result.exit_code == 0, result.output

    sweep_dir = workdir / "experiments" / "s1"
    manifest = yaml.safe_load((sweep_dir / "sweep.yaml").read_text())
    assert manifest["strategy"] == "random"
    assert manifest["sample_seed"] == 42
    assert len(manifest["trials"]) == 3
    assert (sweep_dir / "s1_001" / "config.yaml").is_file()
    assert (sweep_dir / "s1_003" / "config.yaml").is_file()
    assert not (sweep_dir / "s1_004").exists()
    assert len(calls) == 3


def test_random_run_is_reproducible_with_same_seed(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _run(data_dir, "--strategy", "random", "--trials", "3", "--sample-seed", "7", "--name", "s1")
    first = [{k: v for k, v in c.items() if k not in ("data_dir", "output_dir")} for c in calls]

    calls.clear()
    _run(data_dir, "--strategy", "random", "--trials", "3", "--sample-seed", "7", "--name", "s2")
    second = [{k: v for k, v in c.items() if k not in ("data_dir", "output_dir")} for c in calls]
    assert first == second


def test_random_overrides_default_axis(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    result = _run(
        data_dir, "--strategy", "random", "--trials", "2", "--lr", "1e-3,1e-4", "--name", "s1",
    )
    assert result.exit_code == 0, result.output
    assert {c["lr"] for c in calls} <= {1e-3, 1e-4}


def test_random_single_value_fixes_flag(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    result = _run(
        data_dir, "--strategy", "random", "--trials", "2", "--lr", "1e-3", "--name", "s1",
    )
    assert result.exit_code == 0, result.output
    assert all(c["lr"] == 1e-3 for c in calls)


def test_random_trials_exceeds_space_rejected(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _assert_rejected(
        _run(data_dir, "--strategy", "random", "--trials", "999999", "--name", "s1"),
        "distinct combinations",
        calls,
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _assert_rejected(result, fragment, calls):
    assert result.exit_code != 0
    assert fragment.lower() in result.output.lower(), result.output
    assert calls == []


def test_no_multi_valued_flag(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _assert_rejected(_run(data_dir, "--lr", "1e-3", "--name", "s1"), "nothing to sweep", calls)
    assert not (workdir / "experiments" / "s1").exists()


def test_no_flags_at_all(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _assert_rejected(_run(data_dir, "--name", "s1"), "nothing to sweep", calls)


def test_range_rejected(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _assert_rejected(_run(data_dir, "--dropout", "0.1:0.5", "--name", "s1"), "range", calls)


def test_bad_strategy(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    result = _run(data_dir, "--lr", "1e-3,1e-4", "--strategy", "bogus", "--name", "s1")
    assert result.exit_code != 0
    assert calls == []


def test_random_requires_trials(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _assert_rejected(_run(data_dir, "--strategy", "random", "--name", "s1"), "--trials", calls)


def test_trials_rejected_under_grid(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _assert_rejected(
        _run(data_dir, "--lr", "1e-3,1e-4", "--trials", "1", "--name", "s1"), "--trials", calls
    )


def test_existing_sweep_name(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    (workdir / "experiments" / "s1").mkdir(parents=True)
    _assert_rejected(_run(data_dir, "--lr", "1e-3,1e-4", "--name", "s1"), "s1", calls)


def test_empty_item(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _assert_rejected(_run(data_dir, "--lr", "1e-3,,1e-4", "--name", "s1"), "empty", calls)


def test_duplicate_values(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    _assert_rejected(_run(data_dir, "--lr", "1e-3,1e-3", "--name", "s1"), "duplicate", calls)


def test_unknown_metric(data_dir, workdir, monkeypatch):
    calls = _install_fake(monkeypatch)
    result = _run(data_dir, "--lr", "1e-3,1e-4", "--metric", "f1", "--name", "s1")
    assert result.exit_code != 0
    assert calls == []
