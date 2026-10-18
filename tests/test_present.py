"""The shared presentation layer: terminal and WebUI must show the same fields, labels and values."""
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from cvbench.cli.data import data
from cvbench.cli.generate import generate
from cvbench.cli.runs import runs
from cvbench.core import present
from cvbench.datasets.catalog import dataset_info

pytest.importorskip("fastapi", reason="requires the 'web' extra")

from cvbench.web.api import datasets as datasets_api  # noqa: E402
from cvbench.web.api import runs as runs_api  # noqa: E402

# ── formatting ───────────────────────────────────────────────────────────────

def test_one_unit_convention():
    assert present.fmt_pct(0.9345) == "93.5%"
    assert present.fmt_pct(None) == "—"
    assert present.fmt_loss(0.12345) == "0.1235"
    assert present.fmt_metric("val_loss", 0.5) == "0.5000"
    assert present.fmt_metric("map50", 0.5) == "50.0%"
    assert present.fmt_epochs(12, 20) == "12 / 20"
    assert present.fmt_epochs(12) == "12"
    assert present.fmt_epochs(None, 20) == "—"


def test_run_cell_labels_non_accuracy_test_metric():
    plain = present.present_run({"name": "r", "primary_value": 0.9})
    det = present.present_run({"name": "r", "primary_value": 0.5, "primary_metric": "map50"})
    assert present.run_cell(plain, "test") == "90.0%"
    assert present.run_cell(det, "test") == "50.0% mAP@50"


def test_narrow_terminals_drop_backbone_and_date_only():
    keys = [k for k, _, _ in present.run_columns(80)]
    assert "backbone" not in keys and "date" not in keys and "name" in keys and "test" in keys
    assert present.run_columns(200) == present.RUN_COLUMNS


# ── runs: same row in the terminal and the API ───────────────────────────────

def test_runs_list_and_api_show_the_same_values(exps, monkeypatch, write_exp):
    write_exp(exps, "exp_a", val_accuracy=0.9345, val_loss=0.2, primary_value=0.88, epochs_run=12)
    monkeypatch.setenv("COLUMNS", "200")

    (api_run,) = runs_api.list_runs()
    out = CliRunner().invoke(runs, ["list", str(exps)]).output

    assert api_run["display"]["val_accuracy"] == "93.5%"
    for key, _, _ in present.RUN_COLUMNS:
        assert present.run_cell(api_run["display"], key) in out
    for _, label, _ in present.RUN_COLUMNS:
        assert label.split()[0] in out  # headers wrap onto two lines in the terminal


def test_run_detail_headline_matches_runs_show(exps, write_exp):
    write_exp(exps, "exp_a", val_accuracy=0.9, primary_value=0.8, epochs_run=3)
    detail = runs_api.get_run("exp_a")
    out = CliRunner().invoke(runs, ["show", str(exps / "exp_a")]).output

    assert detail["display"]["fields"]
    for label, value in detail["display"]["fields"]:
        assert f"{label:<22}: {value}" in out


def test_eval_report_advisories_and_display_come_from_one_place(exps, write_exp):
    run = write_exp(exps, "exp_a", val_accuracy=0.9)
    report = {
        "task": "classification", "n_images": 10, "overall_accuracy": 0.9345, "top3_accuracy": None,
        "per_class": {"cat": {"precision": 0.5, "recall": 0.25, "f1": 0.3, "support": 4}},
    }
    (run / "eval_report.json").write_text(json.dumps(report))

    detail = runs_api.get_run("exp_a")

    assert ["Overall accuracy", "93.5%  (val 90.0%)"] in detail["eval_display"]["summary"]
    assert detail["eval_display"]["per_class"]["rows"] == [["cat", "50.0%", "25.0%", "30.0%", "4"]]
    assert detail["advisories"] == present.run_advisories(run, report)


def test_present_endpoint_serves_the_same_columns():
    labels = runs_api.get_present()
    assert [c["label"] for c in labels["run_columns"]] == [label for _, label, _ in present.RUN_COLUMNS]
    assert labels["empty"]["runs"] == present.EMPTY_RUNS


# ── datasets: one scan ───────────────────────────────────────────────────────

@pytest.fixture
def cls_root(tmp_path) -> Path:
    out = tmp_path / "data" / "synth_cls"
    result = CliRunner().invoke(
        generate, [str(out), "--train", "4", "--val", "2", "--test", "0", "--image-size", "32"]
    )
    assert result.exit_code == 0, result.output
    return out.resolve()


def test_dataset_info_counts(cls_root):
    info = dataset_info(cls_root)
    assert info["task"] == "classification" and info["format"] == "classification"
    assert info["num_classes"] == len(info["classes"]) > 0
    assert set(info["splits"]) == {"train", "val"}
    assert info["splits"]["train"]["images"] == 4 * info["num_classes"]


def test_data_list_and_api_show_the_same_counts(cls_root):
    entry = datasets_api._dataset_entry(cls_root)
    out = CliRunner().invoke(data, ["list", str(cls_root.parent)]).output

    assert entry["display"] == present.present_dataset(dataset_info(cls_root))
    assert entry["display"]["train"] == str(entry["splits"]["train"]["images"])
    for key, _, _ in present.DATASET_COLUMNS:
        assert entry["display"][key] in out


def test_empty_dataset_text_is_shared(tmp_path):
    out = CliRunner().invoke(data, ["list", str(tmp_path)]).output
    assert present.EMPTY_DATASETS.format(dir=tmp_path) in out.replace("\n", "")
