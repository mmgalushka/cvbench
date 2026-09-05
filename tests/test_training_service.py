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


def test_run_training_marks_status_failed_on_crash(tmp_path, monkeypatch):
    exp_dir = tmp_path / "crashed_run"
    monkeypatch.setattr(training_service, "detect_task_name", lambda data_dir: "classification")
    monkeypatch.setattr(training_service, "get_task", lambda name: _CrashingTask())

    with pytest.raises(RuntimeError, match="simulated crash"):
        training_service.run_training(data_dir=str(tmp_path / "data"), output_dir=str(exp_dir))

    cfg = load_config(str(exp_dir))
    assert cfg.run.status == "failed"
