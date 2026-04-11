"""Tests that require TensorFlow/Keras — mark with @pytest.mark.tf."""
import pytest

pytestmark = pytest.mark.tf


@pytest.fixture
def minimal_cfg():
    from cvbench.core.config import CVBenchConfig
    cfg = CVBenchConfig()
    cfg.model.backbone = "efficientnet_b0"
    cfg.model.input_size = 224
    cfg.model.num_classes = 3
    cfg.model.dropout = 0.2
    cfg.model.fine_tune_from_layer = 0
    cfg.training.learning_rate = 1e-4
    return cfg


def test_build_model_output_shape(minimal_cfg):
    from cvbench.core.model import build_model

    model = build_model(minimal_cfg)
    assert model.output_shape == (None, 3)


def test_build_model_unknown_backbone(minimal_cfg):
    from cvbench.core.model import build_model
    minimal_cfg.model.backbone = "resnet_99"
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model(minimal_cfg)
