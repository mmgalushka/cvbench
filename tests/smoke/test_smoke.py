"""One-epoch pipeline check on real public datasets.

For every dataset in datasets.yaml: explore -> train (1 epoch) -> evaluate -> predict ->
export (TFLite, ONNX). Accuracy after one epoch is meaningless, so a stage passes when the
command exits cleanly and leaves the files it promises.

    ./helper.sh test -m smoke               # all datasets
    pytest -m smoke tests/smoke -k voc      # one dataset
"""
from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tests.smoke.fetch import load_manifest, prepare

BIN = Path(sys.executable).parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

pytestmark = pytest.mark.smoke


def run(stage: str, tool: str, *args: object) -> str:
    """Run a CVBench CLI command; fail naming STAGE and showing the command's output."""
    proc = subprocess.run(
        [str(BIN / tool), *map(str, args)], capture_output=True, text=True, timeout=1800
    )
    output = proc.stdout + proc.stderr
    cmd = " ".join([tool, *map(str, args)])
    assert proc.returncode == 0, f"[{stage}] `{cmd}` exited {proc.returncode}:\n{output[-2000:]}"
    return output


def expect(stage: str, path: Path) -> None:
    assert path.exists(), f"[{stage}] expected {path} to exist"


def first_test_image(ds: Path, task: str) -> Path:
    root = ds / "images" / "test" if task == "detection" else ds / "test"
    return next(p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES)


@pytest.mark.parametrize("entry", load_manifest(), ids=lambda e: e["name"])
def test_pipeline(entry, tmp_path):
    ds = prepare(entry)
    run_dir = tmp_path / "run"

    run("explore", "data", "explore", ds)

    run("train", "train", ds, "--epochs", 1, "--output", run_dir)
    expect("train", run_dir / "best.keras")
    with open(run_dir / "training_log.csv") as f:
        row = next(csv.DictReader(f))
    losses = [float(v) for k, v in row.items() if "loss" in k]
    assert losses and all(math.isfinite(v) for v in losses), f"[train] non-finite loss: {row}"

    run("evaluate", "evaluate", run_dir)
    expect("evaluate", run_dir / "eval_report.json")

    out = run("predict", "predict", run_dir, first_test_image(ds, entry["task"]))
    assert out.strip(), "[predict] no output"

    run("export-tflite", "runs", "export", run_dir, "--format", "tflite")
    expect("export-tflite", run_dir / "export" / "tflite" / "model.tflite")

    run("export-onnx", "runs", "export", run_dir, "--format", "onnx")
    expect("export-onnx", run_dir / "export" / "onnx" / "model.onnx")
