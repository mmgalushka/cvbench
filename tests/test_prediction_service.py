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
