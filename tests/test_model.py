"""Tests that require TensorFlow/Keras — mark with @pytest.mark.tf."""
import pytest

pytestmark = pytest.mark.tf


@pytest.fixture
def minimal_cfg():
    from core.config import CVBenchConfig
    cfg = CVBenchConfig()
    cfg.model.backbone = "efficientnet_b0"
    cfg.model.input_size = 224
    cfg.model.num_classes = 3
    cfg.model.dropout = 0.2
    cfg.model.fine_tune_from_layer = 0
    cfg.training.learning_rate = 1e-4
    cfg.augmentation.placement = "inside_model"
    return cfg


def test_build_model_output_shape(minimal_cfg):
    import keras
    from core.model import build_model

    model = build_model(minimal_cfg)
    assert model.output_shape == (None, 3)


def test_build_model_with_aug_inside(minimal_cfg):
    import keras
    from core.model import build_model

    aug = keras.Sequential([keras.layers.RandomFlip("horizontal")], name="test_aug")
    model = build_model(minimal_cfg, aug_layer=aug)
    layer_names = [l.name for l in model.layers]
    assert "test_aug" in layer_names


def test_build_model_aug_outside_not_in_graph(minimal_cfg):
    import keras
    from core.model import build_model

    minimal_cfg.augmentation.placement = "outside_model"
    aug = keras.Sequential([keras.layers.RandomFlip("horizontal")], name="outside_aug")
    model = build_model(minimal_cfg, aug_layer=aug)
    layer_names = [l.name for l in model.layers]
    assert "outside_aug" not in layer_names


def test_build_model_unknown_backbone(minimal_cfg):
    from core.model import build_model
    minimal_cfg.model.backbone = "resnet_99"
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model(minimal_cfg)
