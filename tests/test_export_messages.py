from click.testing import CliRunner

from cvbench.cli.runs import runs
from cvbench.core.errors import ConfigError, NotFoundError
from cvbench.services.export import (
    build_calibration_set as _build_calibration_set,
)
from cvbench.services.export import (
    print_hailo_instructions as _print_hailo_instructions,
)
from cvbench.services.export import (
    print_onnx_snippet as _print_onnx_snippet,
)
from cvbench.services.export import (
    print_plan_instructions as _print_plan_instructions,
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

    from cvbench.services.export import prepare_hailo_package as _prepare_hailo_package

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
    from cvbench.services.export import base, onnx, tflite

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "best.keras").write_bytes(b"x")
    cfg = SimpleNamespace(
        data=SimpleNamespace(classes=["a", "b"]),
        model=SimpleNamespace(input_size=8, backbone="stub"),
        run=SimpleNamespace(metrics={"val_accuracy": 0.9, "val_loss": 0.1}),
    )
    monkeypatch.setattr(export, "resolve_run_dir", lambda _name: str(run_dir))
    monkeypatch.setattr(export, "load_config", lambda _path: cfg)
    monkeypatch.setattr(base, "resolve_task", lambda _cfg: SimpleNamespace(load_model=lambda _path: object()))
    monkeypatch.setattr(tflite, "convert_tflite", lambda _m, p, quantize: p.write_bytes(b"t"))
    monkeypatch.setattr(onnx, "convert_onnx", lambda _m, p, _size: p.write_bytes(b"o"))
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
    from cvbench.services.export import hailo

    export, run_dir, json = _stub_run_export(monkeypatch, tmp_path)
    monkeypatch.setattr(
        hailo,
        "prepare_hailo_package",
        lambda rd, cfg, calib_total, calib_strategy: (rd / "export" / "hailo", {"calib_set_images": 4}),
    )
    (run_dir / "export" / "hailo").mkdir(parents=True)

    out_dir = export.run_export("r", format="hailo", hw_arch="hailo8")

    info = json.loads((out_dir / "export_info.json").read_text())
    assert info["hw_arch"] == "hailo8" and info["calib_set_images"] == 4
    assert "--hw-arch hailo8 " in capsys.readouterr().out


def test_run_export_plan_writes_package_info(monkeypatch, tmp_path, capsys):
    export, run_dir, json = _stub_run_export(monkeypatch, tmp_path)

    out_dir = export.run_export("r", format="plan")

    info = json.loads((out_dir / "export_info.json").read_text())
    assert info["format"] == "plan" and info["input_shape"] == [1, 8, 8, 3]
    assert "predict run --format plan" in capsys.readouterr().out


def test_run_export_plan_reuses_existing_onnx(monkeypatch, tmp_path, capsys):
    export, run_dir, json = _stub_run_export(monkeypatch, tmp_path)
    plan_dir = run_dir / "export" / "plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "model.onnx").write_bytes(b"keep")

    export.run_export("r", format="plan")

    assert (plan_dir / "model.onnx").read_bytes() == b"keep"
    assert "Reusing   model.onnx" in capsys.readouterr().out
    info = json.loads((plan_dir / "export_info.json").read_text())
    assert info["jetson_trtexec"].startswith("trtexec") and info["quantize"] is None


def test_run_export_ignores_hw_arch_for_non_hailo_formats(monkeypatch, tmp_path):
    export, _, _ = _stub_run_export(monkeypatch, tmp_path)

    out_dir = export.run_export("r", format="tflite", hw_arch="bogus")

    assert (out_dir / "model.tflite").exists()


def test_run_export_rejects_unknown_format():
    import pytest

    from cvbench.services.export import run_export

    with pytest.raises(ConfigError, match="Unknown export format 'coreml'"):
        run_export("r", format="coreml")


def test_registry_has_one_exporter_per_name():
    from cvbench.services.export import EXPORTERS

    assert set(EXPORTERS) == {"tflite", "onnx", "plan", "hailo"}
    assert all(name == e.name for name, e in EXPORTERS.items())


def test_export_info_keys_are_stable_across_targets(monkeypatch, tmp_path):
    export, run_dir, json = _stub_run_export(monkeypatch, tmp_path)
    common = [
        "source_checkpoint", "format", "quantize", "input_shape", "input_dtype",
        "normalization", "classes", "backbone", "val_accuracy", "val_loss", "exported_at",
    ]

    tflite_dir = export.run_export("r", format="tflite")
    onnx_dir = export.run_export("r", format="onnx")

    assert list(json.loads((tflite_dir / "export_info.json").read_text())) == common
    assert list(json.loads((onnx_dir / "export_info.json").read_text())) == [*common, "jetson_trtexec"]


def _calib_cfg(tmp_path, per_class):
    import numpy as np
    from PIL import Image

    for cls, n in per_class.items():
        d = tmp_path / "train" / cls if cls else tmp_path / "train"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.fromarray(np.full((8, 8, 3), (i * 23) % 255, dtype=np.uint8)).save(d / f"{i}.png")

    class _Cfg:
        class data:  # noqa: N801
            train_dir = str(tmp_path / "train")
            val_dir = str(tmp_path / "val")

        class model:  # noqa: N801
            input_size = 8

    return _Cfg


def test_calibration_strategies_respect_quota_and_shape(tmp_path):
    import numpy as np
    import pytest

    cfg = _calib_cfg(tmp_path, {"a": 6, "b": 2})
    for strategy, expected in (("equal", 4), ("proportional", 4), ("stratified", 4)):
        out = tmp_path / f"{strategy}.npy"
        n, n_classes = _build_calibration_set(cfg, out, total=4, strategy=strategy)
        assert (n, n_classes) == (expected, 2)
        assert np.load(out).shape == (expected, 8, 8, 3)

    out = tmp_path / "diverse.npy"
    n, n_clusters = _build_calibration_set(cfg, out, total=4, strategy="diverse")
    assert n_clusters >= 1 and np.load(out).shape[0] == n

    with pytest.raises(Exception, match="No images found"):
        _build_calibration_set(_calib_cfg(tmp_path / "empty", {"x": 0}), tmp_path / "e.npy")


def test_calibration_treats_flat_dataset_as_single_bucket(tmp_path):
    cfg = _calib_cfg(tmp_path, {"": 3})
    n, n_classes = _build_calibration_set(cfg, tmp_path / "flat.npy", total=10, strategy="equal")
    assert (n, n_classes) == (3, 1)
