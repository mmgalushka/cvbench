from cvbench.core.config import CVBenchConfig
from cvbench.core.trainer import _fine_tune_label, _near_chance_advisory, _print_header


def _cfg(fine_tune=0, classes=4):
    cfg = CVBenchConfig()
    cfg.model.fine_tune_from_layer = fine_tune
    cfg.model.num_classes = classes
    return cfg


def test_advisory_when_frozen_run_ends_near_chance():
    msg = _near_chance_advisory(_cfg(), {"val_accuracy": 0.27})
    assert msg and "--fine-tune-from-layer -1" in msg


def test_no_advisory_when_accuracy_is_good_or_backbone_unfrozen():
    assert _near_chance_advisory(_cfg(), {"val_accuracy": 0.9}) is None
    assert _near_chance_advisory(_cfg(fine_tune=-1), {"val_accuracy": 0.27}) is None
    assert _near_chance_advisory(_cfg(), {"val_loss": 1.2}) is None  # e.g. detection


def test_fine_tune_label():
    assert "frozen" in _fine_tune_label(0)
    assert "all layers" in _fine_tune_label(-1)
    assert "layer 50" in _fine_tune_label(50)


def test_header_shows_task_finetune_seed_and_small_val_warning(capsys):
    cfg = _cfg()
    cfg.training.seed = 7
    _print_header("experiments/run", cfg, val_images=5)
    out = capsys.readouterr().out
    assert "classification" in out and "frozen backbone" in out
    assert "Seed      : 7" in out
    assert "only 5 image(s)" in out
