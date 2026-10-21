"""A real, tiny classification run through the services — no patched internals.

Cheap enough for the default suite (one epoch on a 32px generated dataset, no pretrained
weights), and it exercises what the faked-task tests in ``test_training_service.py`` can't:
the artifacts a finished run leaves behind and the evaluate step that reads them back.
"""
import pytest

pytestmark = pytest.mark.tf

from cvbench.core.config import load_config  # noqa: E402
from cvbench.core.errors import ConflictError
from cvbench.core.events import (  # noqa: E402
    DeviceDetected,
    EpochEnded,
    EvalProgress,
    EvalStarted,
    RunFinished,
    RunStarted,
    SeedSet,
)
from cvbench.services.evaluation import run_evaluation  # noqa: E402
from cvbench.services.training import run_training  # noqa: E402


def _train(data_dir, **kwargs):
    return run_training(
        str(data_dir), overrides={
            "model.backbone": "efficientnet_b0", "model.weights": "none", "training.epochs": 1,
            "data.batch_size": 4, "model.input_size": 32, "training.seed": 3,
        }, **kwargs,
    )


def test_train_then_evaluate_a_classification_run(exps, tiny_cls_root):
    from pathlib import Path

    exp_dir = Path(_train(tiny_cls_root))

    assert exp_dir.resolve().parent == exps.resolve()
    assert (exp_dir / "best.keras").is_file() and (exp_dir / "training_log.csv").is_file()
    cfg = load_config(str(exp_dir))
    assert cfg.task == "classification" and cfg.run.status == "done"
    assert cfg.run.epochs_run == 1 and cfg.run.val_loss is not None
    assert cfg.run.primary_metric == "accuracy"

    report = run_evaluation(str(exp_dir))

    assert report["task"] == "classification" and report["n_images"] == 2 * len(cfg.data.classes)
    assert 0.0 <= report["overall"]["value"] <= 1.0
    assert load_config(str(exp_dir)).run.metrics["accuracy"] == report["overall"]["value"]
    assert (exp_dir / "eval_report.json").is_file()


def test_overwrite_replaces_an_existing_run(tmp_path, tiny_cls_root, write_exp):
    out = write_exp(tmp_path, "out_run", val_loss=9.9)
    stale = out / "stale.txt"
    stale.write_text("left over from the previous run")

    _train(tiny_cls_root, output_dir=str(out), overwrite=True)

    assert not stale.exists()
    cfg = load_config(str(out))
    assert cfg.run.status == "done" and cfg.run.val_loss != 9.9


def test_an_existing_run_is_not_replaced_without_overwrite(tmp_path, tiny_cls_root, write_exp):
    out = write_exp(tmp_path, "out_run")

    with pytest.raises(ConflictError, match="already exists"):
        _train(tiny_cls_root, output_dir=str(out))


def test_training_and_evaluation_report_progress_to_the_listener(exps, tiny_cls_root):
    events = []
    exp_dir = _train(tiny_cls_root, listener=events.append)

    kinds = [type(e) for e in events]
    assert kinds == [DeviceDetected, SeedSet, RunStarted, EpochEnded, RunFinished]
    started, epoch, finished = events[2], events[3], events[4]
    assert started.exp_dir == exp_dir and started.resume_checkpoint is None
    assert epoch.epoch == 0 and "val_loss" in epoch.logs
    assert finished.status == "done" and finished.run_name == load_config(exp_dir).run.name
    assert finished.final_metrics["val_loss"] == load_config(exp_dir).run.val_loss
    assert finished.best_events and finished.interrupt_checkpoint is None

    events.clear()
    run_evaluation(exp_dir, listener=events.append)

    assert isinstance(events[0], DeviceDetected) and isinstance(events[1], EvalStarted)
    progress = [e for e in events if isinstance(e, EvalProgress)]
    assert progress[0].done == 0 and not progress[0].finished
    assert progress[-1].finished and progress[-1].done == progress[-1].total
