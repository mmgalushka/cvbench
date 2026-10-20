from cvbench.classification import ClassificationTask
from cvbench.cli._listeners import ConsoleListener, fine_tune_label, print_header
from cvbench.core.config import CVBenchConfig
from cvbench.core.events import RunFinished, RunStarted


def _cfg(fine_tune=0, classes=4):
    cfg = CVBenchConfig()
    cfg.model.fine_tune_from_layer = fine_tune
    cfg.model.num_classes = classes
    return cfg


def _advisories(cfg, metrics):
    return ClassificationTask().training_advisories(cfg, metrics)


def test_advisory_when_frozen_run_ends_near_chance():
    msgs = _advisories(_cfg(), {"val_accuracy": 0.27})
    assert len(msgs) == 1 and "--fine-tune-from-layer -1" in msgs[0]


def test_no_advisory_when_accuracy_is_good_or_backbone_unfrozen():
    assert _advisories(_cfg(), {"val_accuracy": 0.9}) == []
    assert _advisories(_cfg(fine_tune=-1), {"val_accuracy": 0.27}) == []
    assert _advisories(_cfg(), {"val_loss": 1.2}) == []  # e.g. detection


def test_fine_tune_label():
    assert "frozen" in fine_tune_label(0)
    assert "all layers" in fine_tune_label(-1)
    assert "layer 50" in fine_tune_label(50)


def test_header_shows_task_finetune_seed_and_small_val_warning(capsys):
    cfg = _cfg()
    cfg.training.seed = 7
    print_header("experiments/run", cfg, val_images=5)
    out = capsys.readouterr().out
    assert "classification" in out and "frozen backbone" in out
    assert "Seed      : 7" in out
    assert "only 5 image(s)" in out


def test_console_listener_renders_start_and_summary(capsys):
    listener = ConsoleListener()
    listener(RunStarted("experiments/run", _cfg(), 100, "experiments/old/best.keras"))
    listener(RunFinished(
        exp_dir="experiments/run", run_name="run", status="done", final_metrics={"val_loss": 0.5},
        best_events=[{"metric": "val_loss", "epoch": 2, "value": 0.5}], advisories=["careful"],
    ))
    out = capsys.readouterr().out
    assert "Resuming from: experiments/old/best.keras" in out
    assert "Training complete" in out and "val_loss" in out and "Best weights history" in out
    assert "careful" in out and "evaluate run" in out
