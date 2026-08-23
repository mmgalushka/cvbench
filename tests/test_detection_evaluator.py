"""End-to-end detection train + evaluate — the eval_report.json envelope contract."""
import json

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.tf

from cvbench.cli.generate import generate


@pytest.fixture
def yolo_project(tmp_path, monkeypatch):
    """A tiny generated YOLO dataset, with experiments/ scoped to tmp_path."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        generate,
        ["data/yolo", "--format", "yolo", "--train", "6", "--val", "3", "--test", "4",
         "--image-size", "64", "--max-objects", "2", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    return tmp_path


def test_train_then_evaluate_produces_a_valid_envelope(yolo_project):
    from cvbench.services.evaluation import run_evaluation
    from cvbench.services.training import run_training

    exp_dir = run_training(
        "data/yolo", backbone="efficientnet_b0", epochs=1, batch_size=2, input_size=64,
    )

    report = run_evaluation(exp_dir)

    # Shared envelope keys — the same contract classification's report satisfies.
    assert report["task"] == "detection"
    assert report["split"] == "test"
    assert report["n_images"] == 4
    assert set(report["overall"]) == {"metric", "value", "label"}
    assert report["overall"]["metric"] == "map50"
    assert isinstance(report["per_class"], dict)
    assert isinstance(report["samples"], list)

    # Detection-specific block.
    det = report["detection"]
    assert det["map50"] == report["overall"]["value"]
    assert set(det["counts"]) == {"tp", "fp", "fn"}

    # Persisted to disk and to config.yaml.
    from pathlib import Path
    written = json.loads((Path(exp_dir) / "eval_report.json").read_text())
    assert written["task"] == "detection"

    from cvbench.core.config import load_config
    cfg = load_config(exp_dir)
    assert cfg.task == "detection"
    assert cfg.run.test_metric == "map50"
    assert cfg.run.test_accuracy == report["overall"]["value"]
    assert cfg.run.val_accuracy is None  # detection never sets this


def test_scan_experiments_surfaces_the_detection_run(yolo_project):
    from cvbench.core.runs import scan_experiments
    from cvbench.services.evaluation import run_evaluation
    from cvbench.services.training import run_training

    exp_dir = run_training("data/yolo", backbone="efficientnet_b0", epochs=1, batch_size=2, input_size=64)
    run_evaluation(exp_dir)

    entries = scan_experiments("experiments")
    assert len(entries) == 1
    assert entries[0]["task"] == "detection"
    assert entries[0]["test_metric"] == "map50"
    assert entries[0]["test_accuracy"] is not None
