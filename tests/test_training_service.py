import pytest

from cvbench.core.config import load_config
from cvbench.core.task import DatasetSpec
from cvbench.services import training as training_service


class _CrashingTask:
    """Minimal fake Task whose dataset build blows up, simulating a crash
    partway through run_training (before the real training loop even starts)."""

    name = "classification"

    def validate_config(self, cfg):
        return []

    def resolve_layout(self, cfg):
        cfg.data.classes = ["cat", "dog"]
        cfg.model.num_classes = 2
        return DatasetSpec(
            class_names=["cat", "dog"],
            train_dir=cfg.data.train_dir,
            val_dir=cfg.data.val_dir,
            test_dir=cfg.data.test_dir,
            num_train_images=10,
        )

    def fit_class_weight(self, cfg, spec):
        return None

    def build_datasets(self, cfg, spec):
        raise RuntimeError("simulated crash during dataset build")

    def filter_transforms(self, transforms):
        return transforms

    def build_model(self, cfg):
        raise AssertionError("should not be reached — build_datasets crashes first")


def test_run_training_leaves_no_run_dir_on_dataset_crash(tmp_path, monkeypatch):
    exp_dir = tmp_path / "crashed_run"
    monkeypatch.setattr(training_service, "detect_task_name", lambda data_dir: "classification")
    monkeypatch.setattr(training_service, "get_task", lambda name: _CrashingTask())

    with pytest.raises(RuntimeError, match="simulated crash"):
        training_service.run_training(data_dir=str(tmp_path / "data"), output_dir=str(exp_dir))

    # Dataset build runs before the run dir is created: nothing left behind.
    assert not exp_dir.exists()


class _InterruptingTask(_CrashingTask):
    """Dataset build raises KeyboardInterrupt instead of crashing, simulating Ctrl-C."""

    def build_datasets(self, cfg, spec):
        raise KeyboardInterrupt()


def test_run_training_leaves_no_run_dir_on_keyboard_interrupt(tmp_path, monkeypatch):
    exp_dir = tmp_path / "interrupted_run"
    monkeypatch.setattr(training_service, "detect_task_name", lambda data_dir: "classification")
    monkeypatch.setattr(training_service, "get_task", lambda name: _InterruptingTask())

    with pytest.raises(KeyboardInterrupt):
        training_service.run_training(data_dir=str(tmp_path / "data"), output_dir=str(exp_dir))

    assert not exp_dir.exists()


def test_run_training_interrupt_enabled_overrides_config(tmp_path, monkeypatch):
    exp_dir = tmp_path / "run"
    monkeypatch.setattr(training_service, "detect_task_name", lambda data_dir: "classification")
    seen_cfg = {}

    class _RecordingTask(_CrashingTask):
        def build_datasets(self, cfg, spec):
            seen_cfg["enabled"] = cfg.training.interrupt.enabled
            raise RuntimeError("stop before the real training loop")

    monkeypatch.setattr(training_service, "get_task", lambda name: _RecordingTask())

    with pytest.raises(RuntimeError, match="stop before"):
        training_service.run_training(
            data_dir=str(tmp_path / "data"), output_dir=str(exp_dir), interrupt_enabled=False,
        )

    assert seen_cfg["enabled"] is False


class _TrainingCrashTask(_CrashingTask):
    """Datasets and model build fine; only the training loop itself fails."""

    def build_datasets(self, cfg, spec):
        return None, None, 10

    def build_model(self, cfg):
        return None


def test_run_training_marks_status_failed_when_training_loop_crashes(tmp_path, monkeypatch):
    exp_dir = tmp_path / "crashed_run"
    monkeypatch.setattr(training_service, "detect_task_name", lambda data_dir: "classification")
    monkeypatch.setattr(training_service, "get_task", lambda name: _TrainingCrashTask())
    monkeypatch.setattr(training_service, "apply_augmentation", lambda ds, t: ds)

    def _boom(**kwargs):
        raise RuntimeError("simulated crash in training loop")

    monkeypatch.setattr(training_service._trainer, "train", _boom)
    with pytest.raises(RuntimeError, match="training loop"):
        training_service.run_training(data_dir=str(tmp_path / "data"), output_dir=str(exp_dir))

    assert load_config(str(exp_dir)).run.status == "failed"


def test_run_training_rejects_existing_run_without_overwrite(tmp_path):
    exp_dir = tmp_path / "existing"
    exp_dir.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        training_service.run_training(data_dir=str(tmp_path / "data"), output_dir=str(exp_dir))


def test_run_training_rejects_unsafe_run_name(tmp_path):
    with pytest.raises(ValueError, match="letters, digits"):
        training_service.run_training(
            data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "x --lr 1e-3"),
        )
    assert list(tmp_path.iterdir()) == []
