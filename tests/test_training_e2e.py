"""A real, tiny classification run through the services — no patched internals.

Cheap enough for the default suite (one epoch on a 32px generated dataset, no pretrained
weights), and it exercises what the faked-task tests in ``test_training_service.py`` can't:
the artifacts a finished run leaves behind and the evaluate step that reads them back.
"""
import pytest

pytestmark = pytest.mark.tf

from cvbench.core.config import load_config  # noqa: E402
from cvbench.services.evaluation import run_evaluation  # noqa: E402
from cvbench.services.training import run_training  # noqa: E402


def _train(data_dir, **kwargs):
    return run_training(
        str(data_dir), backbone="efficientnet_b0", weights="none", epochs=1, batch_size=4,
        input_size=32, seed=3, **kwargs,
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

    with pytest.raises(ValueError, match="already exists"):
        _train(tiny_cls_root, output_dir=str(out))
