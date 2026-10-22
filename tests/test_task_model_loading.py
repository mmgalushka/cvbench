"""One model-loading path: every service loads checkpoints through ``Task.load_model``."""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.tf

from cvbench.core.config import CVBenchConfig, save_config  # noqa: E402
from cvbench.tasks import get_task, resolve_task  # noqa: E402

SIZE = 64


@pytest.fixture
def detection_run(tmp_path) -> Path:
    """A run dir holding a config.yaml and a tiny compiled detection model as best.keras."""
    from cvbench.detection.model import build_model

    cfg = CVBenchConfig(task="detection")
    cfg.model.backbone = "resnet_18"
    cfg.model.weights = "none"
    cfg.model.input_size = SIZE
    cfg.model.num_classes = 2
    cfg.detection.strides = [16, 32]
    cfg.detection.anchors = [
        [[0.15, 0.15], [0.25, 0.25], [0.35, 0.35]],
        [[0.45, 0.45], [0.55, 0.55], [0.7, 0.7]],
    ]
    run_dir = tmp_path / "det_run"
    run_dir.mkdir()
    save_config(cfg, str(run_dir))
    build_model(cfg).save(str(run_dir / "best.keras"))
    return run_dir


def test_custom_objects_are_declared_per_task():
    from cvbench.detection.losses import YoloLoss

    assert get_task("classification").custom_objects() == {}
    assert get_task("detection").custom_objects() == {"YoloLoss": YoloLoss}


def test_task_load_model_restores_a_compiled_detection_model(detection_run):
    from cvbench.core.config import load_config

    model = resolve_task(load_config(str(detection_run))).load_model(str(detection_run / "best.keras"))

    assert [tuple(s) for s in model.output_shape] == [(None, 4, 4, 21), (None, 2, 2, 21)]


def test_prediction_service_loads_a_detection_checkpoint(detection_run):
    from cvbench.core.config import load_config
    from cvbench.services import prediction

    img = detection_run / "img.png"
    from PIL import Image
    Image.new("RGB", (SIZE, SIZE), (10, 20, 30)).save(img)
    task = resolve_task(load_config(str(detection_run)))

    outputs = prediction._infer_keras(detection_run / "best.keras", [str(img)], SIZE, task)

    assert len(outputs) == 1
    assert [o.shape for o in outputs[0]] == [(4, 4, 21), (2, 2, 21)]
    assert all(o.dtype == np.float32 for o in outputs[0])


def test_export_service_loads_a_detection_checkpoint(detection_run, monkeypatch):
    from cvbench.services import export

    seen = []
    monkeypatch.setattr(export, "resolve_run_dir", lambda _name: str(detection_run))

    def fake_export(model, path, quantize):
        seen.append(model)
        path.write_bytes(b"t")

    monkeypatch.setattr(export, "_export_tflite", fake_export)

    export.run_export("det_run", format="tflite")

    assert len(seen) == 1
    assert [tuple(s) for s in seen[0].output_shape] == [(None, 4, 4, 21), (None, 2, 2, 21)]
