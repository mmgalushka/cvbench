"""Tests for classification/evaluator.py — the eval_report.json envelope shape."""

import numpy as np
import pytest

pytestmark = pytest.mark.tf


@pytest.fixture
def image_dir(tmp_path):
    """A minimal 2-class image dataset: 3 images per class."""
    from PIL import Image

    classes = ["cat", "dog"]
    for split in ["train", "test"]:
        for cls in classes:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(3):
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                img.save(d / f"{i:03d}.jpg")
    return tmp_path


def _tiny_model(num_classes: int):
    import keras

    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def test_evaluate_report_envelope_shape(tmp_path, image_dir):
    from cvbench.classification.data import build_dataset
    from cvbench.classification.evaluator import evaluate

    class_names = ["cat", "dog"]
    from cvbench.core.config import CVBenchConfig
    cfg = CVBenchConfig()
    cfg.model.input_size = 32
    cfg.data.batch_size = 2

    test_ds = build_dataset(str(image_dir / "test"), class_names, cfg, training=False)
    model = _tiny_model(len(class_names))

    report = evaluate(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        run_dir=str(tmp_path),
        test_dir=str(image_dir / "test"),
    )

    # Shared envelope keys
    assert report["task"] == "classification"
    assert report["split"] == "test"
    assert report["n_images"] == 6
    assert set(report["overall"]) == {"metric", "value", "label"}
    assert report["overall"]["metric"] == "accuracy"
    assert isinstance(report["per_class"], dict)
    assert isinstance(report["samples"], list)

    # Legacy top-level mirrors, for readers written before the envelope existed
    assert report["overall_accuracy"] == report["overall"]["value"]
    assert "confusion_matrix" in report
    assert report["classification"]["confusion_matrix"] == report["confusion_matrix"]

    # Actually written to disk
    written = (tmp_path / "eval_report.json").read_text()
    assert '"overall"' in written
    assert '"overall_accuracy"' in written
