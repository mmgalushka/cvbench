"""CLI tests using click's CliRunner — no Docker, no real training."""
from pathlib import Path

from click.testing import CliRunner

from cvbench.cli.runs import runs
from cvbench.cli.serve import serve
from cvbench.core.config import build_config, save_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_exp(parent: Path, name: str, **run_kwargs) -> Path:
    exp_dir = parent / name
    exp_dir.mkdir(parents=True)
    cfg = build_config("data")
    cfg.run.name = name
    cfg.run.date = "2026-01-01"
    cfg.run.status = "done"
    for k, v in run_kwargs.items():
        setattr(cfg.run, k, v)
    save_config(cfg, str(exp_dir))
    return exp_dir


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------

def test_runs_list_empty(tmp_path):
    runner = CliRunner()
    result = runner.invoke(runs, ["list", str(tmp_path)])
    assert result.exit_code == 0
    assert "No experiments found" in result.output


def test_runs_list_shows_entries(tmp_path):
    _write_exp(tmp_path, "exp_a", val_accuracy=0.91, epochs_run=10)
    runner = CliRunner()
    result = runner.invoke(runs, ["list", str(tmp_path)])
    assert result.exit_code == 0
    assert "exp_a" in result.output


def test_runs_list_sort_by_val_accuracy(tmp_path):
    _write_exp(tmp_path, "exp_low", val_accuracy=0.7)
    _write_exp(tmp_path, "exp_high", val_accuracy=0.95)
    runner = CliRunner()
    result = runner.invoke(runs, ["list", str(tmp_path), "--sort", "val_accuracy"])
    assert result.exit_code == 0
    assert result.output.index("exp_high") < result.output.index("exp_low")


# ---------------------------------------------------------------------------
# runs best
# ---------------------------------------------------------------------------

def test_runs_best(tmp_path):
    _write_exp(tmp_path, "bad_loss", val_loss=0.9)
    _write_exp(tmp_path, "good_loss", val_loss=0.1)
    runner = CliRunner()
    result = runner.invoke(runs, ["best", str(tmp_path)])
    assert result.exit_code == 0
    assert "good_loss" in result.output


def test_runs_best_no_metric(tmp_path):
    _write_exp(tmp_path, "no_metric")  # val_accuracy stays None
    runner = CliRunner()
    result = runner.invoke(runs, ["best", str(tmp_path)])
    assert result.exit_code == 0
    assert "No experiments" in result.output


# ---------------------------------------------------------------------------
# runs show
# ---------------------------------------------------------------------------

def test_runs_show(tmp_path):
    exp_a = _write_exp(tmp_path, "exp_a", val_accuracy=0.9)
    runner = CliRunner()
    result = runner.invoke(runs, ["show", str(exp_a)])
    assert result.exit_code == 0
    assert "exp_a" in result.output
    assert "val_accuracy" in result.output
    assert "None" in result.output  # Exports section, no export/ dir yet


def test_runs_show_with_exports_and_eval(tmp_path):
    import json

    exp_a = _write_exp(tmp_path, "exp_a")
    (exp_a / "export" / "tflite").mkdir(parents=True)
    report = {
        "overall": {"label": "Overall Accuracy", "value": 0.87},
        "per_class": {
            "cat": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10},
            "dog": {"precision": 0.7, "recall": 0.6, "f1": 0.65, "support": 5},
        },
        "confusion_matrix": {"classes": ["cat", "dog"], "matrix": [[8, 2], [1, 4]]},
    }
    (exp_a / "eval_report.json").write_text(json.dumps(report))

    runner = CliRunner()
    result = runner.invoke(runs, ["show", str(exp_a)])
    assert result.exit_code == 0
    assert "tflite" in result.output
    assert "Overall Accuracy" in result.output
    assert "cat" in result.output
    assert "Confusion matrix" in result.output


def test_runs_show_classification_report_dispatches_to_dedicated_renderer(tmp_path):
    """A real classification report (with a "task" key) prints Top-3 accuracy."""
    import json

    exp_a = _write_exp(tmp_path, "exp_a")
    report = {
        "task": "classification",
        "n_images": 100,
        "overall_accuracy": 0.9123,
        "top3_accuracy": 0.98,
        "per_class": {
            "cat": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 50},
        },
        "confusion_matrix": {"classes": ["cat"], "matrix": [[50]]},
    }
    (exp_a / "eval_report.json").write_text(json.dumps(report))

    runner = CliRunner()
    result = runner.invoke(runs, ["show", str(exp_a)])
    assert result.exit_code == 0
    assert "Top-3 accuracy" in result.output
    assert "98.0%" in result.output


def test_runs_show_detection_report_dispatches_to_dedicated_renderer(tmp_path):
    """A real detection report (with a "task" key) prints localization + per-class AP."""
    import json

    exp_a = _write_exp(tmp_path, "exp_a")
    report = {
        "task": "detection",
        "n_images": 20,
        "detection": {
            "map50": 0.75,
            "conf_threshold": 0.25,
            "iou_threshold": 0.5,
            "counts": {"tp": 30, "fp": 5, "fn": 3},
            "localization": {
                "mean_iou": 0.82, "ap50": 0.75, "ap75": 0.6,
                "recall_sweep": {"0.5": 0.9, "0.75": 0.7, "0.9": 0.4},
            },
            "confusion_matrix": {"classes": ["cat", "background"], "matrix": [[15, 1], [1, 0]]},
        },
        "per_class": {
            "cat": {"ap": 0.7, "ap75": 0.55, "precision": 0.85, "recall": 0.8, "f1": 0.82, "support": 15},
        },
    }
    (exp_a / "eval_report.json").write_text(json.dumps(report))

    runner = CliRunner()
    result = runner.invoke(runs, ["show", str(exp_a)])
    assert result.exit_code == 0
    assert "Localization" in result.output
    assert "AP@50" in result.output and "AP@75" in result.output
    assert "0.5500" in result.output  # cat's per-class AP@75


def test_runs_show_no_config(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(runs, ["show", str(empty_dir)])
    assert result.exit_code != 0
    assert "config.yaml" in result.output


def test_runs_show_not_found():
    runner = CliRunner()
    result = runner.invoke(runs, ["show", "no_such_run_xyz"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# runs compare
# ---------------------------------------------------------------------------

def test_runs_compare(tmp_path):
    exp_a = _write_exp(tmp_path, "exp_a", val_accuracy=0.8)
    exp_b = _write_exp(tmp_path, "exp_b", val_accuracy=0.9)
    runner = CliRunner()
    result = runner.invoke(runs, ["compare", str(exp_a), str(exp_b)])
    assert result.exit_code == 0
    assert "exp_a" in result.output
    assert "exp_b" in result.output


def test_runs_compare_no_config(tmp_path):
    exp_a = _write_exp(tmp_path, "exp_a")
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(runs, ["compare", str(exp_a), str(empty_dir)])
    assert result.exit_code != 0
    assert "config.yaml" in result.output


# ---------------------------------------------------------------------------
# serve --help (no uvicorn / TensorFlow needed)
# ---------------------------------------------------------------------------

def test_serve_help():
    result = CliRunner().invoke(serve, ["--help"])
    assert result.exit_code == 0
    for token in ("--host", "--port", "CVBENCH_URL"):
        assert token in result.output
