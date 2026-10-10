from click.testing import CliRunner

from cvbench.cli.runs import runs
from cvbench.services.export import (
    _build_calibration_set,
    _print_hailo_instructions,
    _print_onnx_snippet,
    _print_plan_instructions,
)


def test_plan_instructions_point_at_plan_folder(capsys):
    _print_plan_instructions("r", onnx_exists=True)
    out = capsys.readouterr().out
    assert "experiments/r/export/plan/model.onnx" in out
    assert "export/onnx" not in out
    assert "predict r --format plan" in out


def test_hailo_instructions_use_hw_arch(capsys):
    from pathlib import Path

    _print_hailo_instructions("r", Path("."), hw_arch="hailo8")
    out = capsys.readouterr().out
    assert "--hw-arch hailo8 " in out
    assert "hailo8l" not in out


def test_onnx_snippet_documents_input_contract(capsys):
    _print_onnx_snippet("model.onnx", 96)
    out = capsys.readouterr().out
    assert "onnxruntime" in out and "(1, 96, 96, 3)" in out and "0–255" in out


def test_export_help_documents_contract():
    result = CliRunner().invoke(runs, ["export", "--help"])
    assert result.exit_code == 0
    text = " ".join(result.output.split())
    assert "--hw-arch" in text
    assert "cvbench[export]" in text
    assert "dynamic-range" in text
    assert "Ignored for plan and hailo" in text


def test_calibration_set_warns_when_smaller_than_requested(tmp_path, capsys):
    import numpy as np
    from PIL import Image

    train = tmp_path / "train" / "a"
    train.mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.full((8, 8, 3), i * 40, dtype=np.uint8)).save(train / f"{i}.png")

    class _Cfg:
        class data:  # noqa: N801
            train_dir = str(tmp_path / "train")
            val_dir = str(tmp_path / "val")

        class model:  # noqa: N801
            input_size = 8

    n, _ = _build_calibration_set(_Cfg, tmp_path / "c.npy", total=50, strategy="equal")
    assert n == 3
    assert "fewer than the 50 requested" in capsys.readouterr().out
