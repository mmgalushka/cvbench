"""Tests that require TensorFlow — mark with @pytest.mark.tf."""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.tf


@pytest.fixture
def image_dir(tmp_path):
    """Create a minimal image dataset: 2 classes, 4 images each."""
    from PIL import Image

    classes = ["cat", "dog"]
    for split in ["train", "val"]:
        for cls in classes:
            d = tmp_path / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(d / f"{i:03d}.jpg")
    return tmp_path


def test_get_class_names(image_dir):
    from core.data import get_class_names
    names = get_class_names(str(image_dir / "train"))
    assert names == ["cat", "dog"]


def test_build_dataset_shape(image_dir):
    from core.config import CVBenchConfig
    from core.data import build_dataset, get_class_names

    cfg = CVBenchConfig()
    cfg.model.input_size = 64
    cfg.data.batch_size = 2
    cfg.data.train_dir = str(image_dir / "train")

    class_names = get_class_names(cfg.data.train_dir)
    ds = build_dataset(str(image_dir / "val"), class_names, cfg, training=False)

    images, labels = next(iter(ds))
    assert images.shape == (2, 64, 64, 3)
    assert labels.shape == (2, 2)  # 2 classes, one-hot


def test_build_datasets_returns_class_names(image_dir):
    from core.config import CVBenchConfig
    from core.data import build_datasets

    cfg = CVBenchConfig()
    cfg.model.input_size = 64
    cfg.data.batch_size = 2
    cfg.data.train_dir = str(image_dir / "train")
    cfg.data.val_dir = str(image_dir / "val")

    _, _, class_names = build_datasets(cfg)
    assert class_names == ["cat", "dog"]
