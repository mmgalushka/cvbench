from pathlib import Path

import pytest

from cvbench.core.checkpoint import prune_checkpoints
from cvbench.core.config import CVBenchConfig


def _make_cfg(strategy="best_only", keep_last_n=3, every_n=5):
    cfg = CVBenchConfig()
    cfg.training.checkpoints.strategy = strategy
    cfg.training.checkpoints.keep_last_n = keep_last_n
    cfg.training.checkpoints.every_n_epochs = every_n
    return cfg


def _create_epoch_files(run_dir: Path, epochs: list[int]):
    for e in epochs:
        (run_dir / f"epoch_{e:03d}.keras").touch()


def test_prune_keeps_last_n(tmp_path):
    _create_epoch_files(tmp_path, [1, 2, 3, 4, 5])
    cfg = _make_cfg(keep_last_n=3)
    prune_checkpoints(str(tmp_path), cfg)

    remaining = sorted(tmp_path.glob("epoch_*.keras"))
    assert len(remaining) == 3
    assert remaining[0].name == "epoch_003.keras"
    assert remaining[-1].name == "epoch_005.keras"


def test_prune_never_removes_best(tmp_path):
    _create_epoch_files(tmp_path, [1, 2, 3, 4, 5])
    (tmp_path / "best.keras").touch()
    cfg = _make_cfg(keep_last_n=2)
    prune_checkpoints(str(tmp_path), cfg)

    assert (tmp_path / "best.keras").exists()


def test_prune_no_op_when_few_checkpoints(tmp_path):
    _create_epoch_files(tmp_path, [1, 2])
    cfg = _make_cfg(keep_last_n=5)
    prune_checkpoints(str(tmp_path), cfg)

    remaining = list(tmp_path.glob("epoch_*.keras"))
    assert len(remaining) == 2


def test_prune_all_when_keep_zero(tmp_path):
    _create_epoch_files(tmp_path, [1, 2, 3])
    cfg = _make_cfg(keep_last_n=0)
    prune_checkpoints(str(tmp_path), cfg)

    remaining = list(tmp_path.glob("epoch_*.keras"))
    assert len(remaining) == 0


def test_build_checkpoint_callback_best_only(tmp_path):
    from cvbench.core.checkpoint import build_checkpoint_callback
    cfg = _make_cfg(strategy="best_only")
    cb = build_checkpoint_callback(cfg, str(tmp_path))
    assert cb is not None
    assert "best.keras" in cb.filepath


def test_build_checkpoint_callback_every_epoch(tmp_path):
    from cvbench.core.checkpoint import build_checkpoint_callback
    import keras
    cfg = _make_cfg(strategy="every_epoch")
    cb = build_checkpoint_callback(cfg, str(tmp_path))
    assert cb is not None
    assert isinstance(cb, keras.callbacks.ModelCheckpoint)
    assert "epoch_" in cb.filepath


def test_build_checkpoint_callback_unknown_strategy(tmp_path):
    from cvbench.core.checkpoint import build_checkpoint_callback
    cfg = _make_cfg(strategy="unknown_strategy")
    with pytest.raises(ValueError, match="Unknown checkpoint strategy"):
        build_checkpoint_callback(cfg, str(tmp_path))
