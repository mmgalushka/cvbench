"""Tests that require TensorFlow/Keras — mark with @pytest.mark.tf."""
import pytest

from cvbench.core.errors import ConfigError

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
    from cvbench.classification.model import build_model

    model = build_model(minimal_cfg)
    assert model.output_shape == (None, 3)


def test_build_model_unknown_backbone(minimal_cfg):
    from cvbench.classification.model import build_model
    minimal_cfg.model.backbone = "resnet_99"
    with pytest.raises(ConfigError, match="Unknown backbone"):
        build_model(minimal_cfg)


def _tiny_stack():
    import keras

    inputs = keras.Input(shape=(8, 8, 3))
    x = keras.layers.Conv2D(2, 3, name="c1")(inputs)
    x = keras.layers.Conv2D(2, 3, name="c2")(x)
    return keras.Model(inputs, x, name="backbone")


@pytest.mark.parametrize(
    "fine_tune_from_layer, trainable_names",
    [(0, []), (-1, ["c1", "c2"]), (2, ["c2"])],
    ids=["frozen", "unfrozen", "from-layer-2"],
)
def test_apply_freeze_semantics(fine_tune_from_layer, trainable_names):
    from cvbench.core.backbone import apply_freeze

    backbone = _tiny_stack()
    apply_freeze(backbone, fine_tune_from_layer)

    assert [lyr.name for lyr in backbone.layers if lyr.trainable and lyr.name.startswith("c")] == trainable_names
    assert backbone.trainable is (fine_tune_from_layer != 0)


def test_trainer_reapplies_freeze_to_a_reloaded_checkpoint():
    import keras

    from cvbench.core.trainer import _apply_fine_tune_flags

    backbone = _tiny_stack()
    backbone.trainable = False
    inputs = keras.Input(shape=(8, 8, 3))
    model = keras.Model(inputs, keras.layers.Dense(2)(keras.layers.Flatten()(backbone(inputs))))

    _apply_fine_tune_flags(model, -1)

    assert model.get_layer("backbone").trainable is True
