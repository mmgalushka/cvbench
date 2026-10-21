from click.testing import CliRunner

from cvbench.cli.runs import runs
from cvbench.core.errors import ConfigError, NotFoundError
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


def test_plan_instructions_note_missing_onnx(capsys):
    _print_plan_instructions("r", onnx_exists=False)
    out = capsys.readouterr().out
    assert "runs export r --format onnx" in out


def test_run_export_rejects_unknown_hw_arch():
    import pytest

    from cvbench.services.export import run_export

    with pytest.raises(ConfigError, match="Unknown --hw-arch 'hailo99'"):
        run_export("r", format="hailo", hw_arch="hailo99")


def test_reused_calibration_set_warns_when_smaller(tmp_path, capsys):
    import numpy as np

    from cvbench.services.export import _prepare_hailo_package

    export_dir = tmp_path / "export" / "hailo"
    export_dir.mkdir(parents=True)
    (export_dir / "model.tflite").write_bytes(b"x")
    np.save(str(export_dir / "calib_set.npy"), np.zeros((4, 8, 8, 3), dtype=np.float32))

    _, meta = _prepare_hailo_package(tmp_path, cfg=None, calib_total=100)

    out = capsys.readouterr().out
    assert meta["calib_set_images"] == 4
    assert "existing calib_set.npy has 4 images" in out


def _stub_run_export(monkeypatch, tmp_path):
    """Point run_export at a fake run dir with converters stubbed out."""
    import json
    from types import SimpleNamespace

    import cvbench.services.export as export

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.keras").write_bytes(b"x")
    cfg = SimpleNamespace(
        data=SimpleNamespace(classes=["a", "b"]),
        model=SimpleNamespace(input_size=8, backbone="stub"),
        run=SimpleNamespace(val_accuracy=0.9, val_loss=0.1),
    )
    monkeypatch.setattr(export, "resolve_run_dir", lambda _name: str(run_dir))
    monkeypatch.setattr(export, "load_config", lambda _path: cfg)
    monkeypatch.setattr(export, "resolve_task", lambda _cfg: SimpleNamespace(load_model=lambda _path: object()))
    monkeypatch.setattr(export, "_export_tflite", lambda _m, p, quantize: p.write_bytes(b"t"))
    monkeypatch.setattr(export, "_export_onnx", lambda _m, p, _size: p.write_bytes(b"o"))
    return export, run_dir, json


def test_run_export_tflite_int8_prints_dynamic_range_note(monkeypatch, tmp_path, capsys):
    export, run_dir, json = _stub_run_export(monkeypatch, tmp_path)

    out_dir = export.run_export("r", format="tflite", quantize="int8")

    assert out_dir == run_dir / "export" / "tflite_int8"
    assert (out_dir / "model_int8.tflite").exists()
    info = json.loads((out_dir / "export_info.json").read_text())
    assert info["quantize"] == "int8" and "jetson_trtexec" not in info
    assert "dynamic-range quantization" in capsys.readouterr().out


def test_run_export_onnx_writes_info_and_snippet(monkeypatch, tmp_path, capsys):
    export, _, json = _stub_run_export(monkeypatch, tmp_path)

    out_dir = export.run_export("r", format="onnx", output_dir=str(tmp_path / "out"))

    assert out_dir == tmp_path / "out" / "onnx"
    info = json.loads((out_dir / "export_info.json").read_text())
    assert info["quantize"] is None and "trtexec" in info["jetson_trtexec"]
    assert "onnxruntime" in capsys.readouterr().out


def test_run_export_requires_checkpoint(monkeypatch, tmp_path):
    import pytest

    export, run_dir, _ = _stub_run_export(monkeypatch, tmp_path)
    (run_dir / "best.keras").unlink()

    for fmt in ("tflite", "plan", "hailo"):
        with pytest.raises(NotFoundError, match="best.keras"):
            export.run_export("r", format=fmt)


def test_run_export_hailo_records_hw_arch(monkeypatch, tmp_path, capsys):
    export, run_dir, json = _stub_run_export(monkeypatch, tmp_path)
    monkeypatch.setattr(
        export,
        "_prepare_hailo_package",
        lambda rd, cfg, calib_total, calib_strategy: (rd / "export" / "hailo", {"calib_set_images": 4}),
    )
    (run_dir / "export" / "hailo").mkdir(parents=True)

    out_dir = export.run_export("r", format="hailo", hw_arch="hailo8")

    info = json.loads((out_dir / "export_info.json").read_text())
    assert info["hw_arch"] == "hailo8" and info["calib_set_images"] == 4
    assert "--hw-arch hailo8 " in capsys.readouterr().out


def test_run_export_plan_writes_package_info(monkeypatch, tmp_path, capsys):
    export, run_dir, json = _stub_run_export(monkeypatch, tmp_path)
    monkeypatch.setattr(export, "_prepare_plan_package", lambda rd, cfg: rd / "export" / "plan")
    (run_dir / "export" / "plan").mkdir(parents=True)

    out_dir = export.run_export("r", format="plan")

    info = json.loads((out_dir / "export_info.json").read_text())
    assert info["format"] == "plan" and info["input_shape"] == [1, 8, 8, 3]
    assert "predict run --format plan" in capsys.readouterr().out
