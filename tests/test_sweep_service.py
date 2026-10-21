"""`services.sweep.run_sweep` driven directly — no click, training is faked."""
from pathlib import Path

import pytest

from cvbench.core.config import apply_overrides, build_config, save_config
from cvbench.core.events import SweepFinished, TrialFailed, TrialFinished, TrialStarted
from cvbench.core.sweep_store import SweepManifest, summarize, trial_dir_name
from cvbench.services import sweep as sweep_service


@pytest.fixture
def fake_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run_training(data_dir, output_dir=None, **kwargs):
        calls.append(kwargs)
        lr = kwargs["overrides"]["training.learning_rate"]
        out = Path(output_dir)
        out.mkdir(parents=True)
        cfg = build_config(str(data_dir))
        cfg.run.name = out.name
        cfg.run.date = "2026-01-01"
        apply_overrides(cfg, kwargs["overrides"])
        if lr == 0.5:
            cfg.run.status = "failed"
            save_config(cfg, str(out))
            raise RuntimeError("boom")
        cfg.run.status = "done"
        cfg.run.metrics["val_loss"] = lr * 1000
        save_config(cfg, str(out))
        return str(out)

    monkeypatch.setattr(sweep_service._training, "run_training", fake_run_training)
    cleared = []
    monkeypatch.setattr(sweep_service, "_release_memory", lambda: cleared.append(True))
    return calls, cleared


def _manifest(lrs):
    return SweepManifest(
        version=1, name="s1", date="2026-01-01", data_dir="data", strategy="grid",
        metric="val_loss", direction="min", axes={"training.learning_rate": [str(v) for v in lrs]},
    )


def _run(lrs):
    events = []
    trials = [{"training.learning_rate": str(v)} for v in lrs]
    values = [{"training.learning_rate": v} for v in lrs]
    sweep_dir = sweep_service.run_sweep(
        _manifest(lrs), trials, values, task_name="classification", listener=events.append,
    )
    return sweep_dir, events


def test_two_trial_sweep_trains_ranks_and_reports_events(fake_training):
    calls, cleared = fake_training

    sweep_dir, events = _run([0.001, 0.0001])

    assert [type(e) for e in events] == [
        TrialStarted, TrialFinished, TrialStarted, TrialFinished, SweepFinished,
    ]
    assert events[0].name == trial_dir_name("s1", 1) and events[0].total == 2
    assert events[-1].sweep_dir == str(sweep_dir)
    assert all(c["overrides"]["training.interrupt.enabled"] is False for c in calls)
    assert all(c["listener"] is not None for c in calls)  # each trial's training events are forwarded
    assert len(cleared) == 2  # memory is released after every trial
    _, rows = summarize(sweep_dir)
    assert [r.is_best for r in rows] == [False, True]


def test_a_failing_trial_is_reported_and_the_sweep_continues(fake_training):
    _, cleared = fake_training

    sweep_dir, events = _run([0.5, 0.001])

    failed = [e for e in events if isinstance(e, TrialFailed)]
    assert len(failed) == 1 and failed[0].stage == "training" and failed[0].message == "boom"
    assert sum(isinstance(e, TrialFinished) for e in events) == 1
    assert isinstance(events[-1], SweepFinished)
    assert len(cleared) == 2
