"""WebUI single-image inference — task-aware result envelope."""
from pathlib import Path

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.tf

from cvbench.cli.generate import generate


@pytest.fixture
def yolo_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        generate,
        ["data/yolo", "--format", "yolo", "--train", "6", "--val", "3", "--test", "4",
         "--image-size", "64", "--max-objects", "2", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    return tmp_path


def test_predict_image_detection_returns_boxes(yolo_project):
    from cvbench.datasets.layout import list_images
    from cvbench.services.prediction import predict_image
    from cvbench.services.training import run_training

    exp_dir = run_training(
        "data/yolo", backbone="efficientnet_b0", epochs=1, batch_size=2, input_size=64,
    )

    img = list_images(yolo_project / "data/yolo/images/test")[0]
    result = predict_image(Path(exp_dir).name, img.read_bytes())

    assert result["task"] == "detection"
    assert isinstance(result["detections"], list)
    for d in result["detections"]:
        assert set(d) >= {"class_index", "class_name", "confidence", "x", "y", "w", "h"}
        assert 0.0 <= d["x"] <= 1.0 and 0.0 <= d["y"] <= 1.0



def test_cli_predict_decodes_detection_run(yolo_project):
    from cvbench.cli.predict import predict
    from cvbench.datasets.layout import list_images
    from cvbench.services.export import run_export
    from cvbench.services.training import run_training

    exp_dir = run_training(
        "data/yolo", backbone="efficientnet_b0", epochs=1, batch_size=2, input_size=64,
    )
    run = Path(exp_dir).name
    run_export(run, format="tflite")
    img = str(list_images(yolo_project / "data/yolo/images/test")[0])

    runner = CliRunner()
    # conf 0 keeps every anchor candidate, so the output is non-empty and the
    # decode path (including TFLite multi-output ordering) is exercised.
    keras_out = runner.invoke(predict, [run, img, "--conf", "0"])
    assert keras_out.exit_code == 0, keras_out.output
    assert "detection" in keras_out.output and "x=" in keras_out.output

    tflite_out = runner.invoke(predict, [run, img, "--format", "tflite", "--conf", "0"])
    assert tflite_out.exit_code == 0, tflite_out.output
    assert "x=" in tflite_out.output

    all_out = runner.invoke(predict, [run, img, "--format", "all", "--conf", "0"])
    assert all_out.exit_code == 0, all_out.output
    assert "box" in all_out.output


def test_cli_predict_conf_rejected_for_classification(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from cvbench.cli.predict import predict
    from cvbench.services import prediction

    monkeypatch.setattr(prediction, "resolve_run_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(
        prediction, "load_config", lambda path: SimpleNamespace(task="classification")
    )
    img = tmp_path / "a.jpg"
    img.write_bytes(b"")

    result = CliRunner().invoke(predict, ["cls_run", str(img), "--conf", "0.3"])

    assert result.exit_code == 1
    assert "--conf only applies to detection runs" in result.output
