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


# ---- build_stem ---------------------------------------------------------------

def _stem_cfg(minimal_cfg, backbone="resnet_18"):
    minimal_cfg.model.backbone = backbone
    minimal_cfg.model.weights = "none"
    minimal_cfg.model.input_size = 64
    return minimal_cfg


def test_stem_without_strides_taps_the_deepest_feature(minimal_cfg):
    from cvbench.core.backbone import build_stem

    inputs, backbone, feat = build_stem(_stem_cfg(minimal_cfg))
    assert backbone.name == "backbone"
    assert inputs.shape == (None, 64, 64, 3)
    assert feat.shape == (None, 2, 2, 512)  # stride 32, resnet_18 channels
    assert backbone.trainable is False  # fine_tune_from_layer == 0


def test_stem_with_strides_taps_pyramid_levels(minimal_cfg):
    from cvbench.core.backbone import build_stem

    cfg = _stem_cfg(minimal_cfg)
    _, backbone, feats = build_stem(cfg, strides=[16, 32])
    assert backbone.name == "backbone"
    assert [f.shape[1] for f in feats] == [4, 2]
    assert backbone.trainable is False

    _, _, single = build_stem(cfg, strides=16)
    assert single.shape[1] == 4


def test_stem_rejects_unsupported_stride(minimal_cfg):
    from cvbench.core.backbone import build_stem

    with pytest.raises(ConfigError, match="Unsupported stride 7"):
        build_stem(_stem_cfg(minimal_cfg), strides=7)


# scale/offset of the keras-hub ImageConverter shipped with every registered preset
# (read from each preset's image_converter.json): (x / 255 - mean) / std per channel.
_PRESET_SCALE = [0.017124753831663668, 0.01750700280112045, 0.017429193899782133]
_PRESET_OFFSET = [-2.1179039301310043, -2.0357142857142856, -1.8044444444444445]


@pytest.mark.parametrize("strides", [None, [16, 32]], ids=["deepest", "pyramid"])
def test_stem_output_matches_the_preset_image_converter(minimal_cfg, strides):
    import keras
    import numpy as np

    from cvbench.core.backbone import build_stem

    inputs, _, feat = build_stem(_stem_cfg(minimal_cfg), strides=strides)
    model = keras.Model(inputs, feat)
    rescaling = next(lyr for lyr in model.layers if isinstance(lyr, keras.layers.Rescaling))
    normalise = keras.Model(inputs, rescaling.output)

    pixels = np.zeros((1, 64, 64, 3), dtype="float32")
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = 255.0, 128.0, 0.0
    expected = pixels * np.array(_PRESET_SCALE, dtype="float32") + np.array(_PRESET_OFFSET, dtype="float32")

    np.testing.assert_allclose(normalise.predict(pixels, verbose=0), expected, rtol=1e-5, atol=1e-5)


def _custom_factory(cfg, load_weights):
    import keras

    inp = keras.Input(shape=(64, 64, 3))
    out = keras.layers.Conv2D(4, 3, strides=2, padding="same", name="conv")(inp)
    model = keras.Model(inp, out, name="my_custom_net")
    model.pyramid_outputs = {"P5": out}
    return model


def test_stem_wraps_a_factory_backbone_under_the_backbone_name(minimal_cfg, monkeypatch):
    from cvbench.core import backbone as backbone_mod

    monkeypatch.setattr(backbone_mod, "BACKBONES", dict(backbone_mod.BACKBONES))
    backbone_mod.register_backbone(backbone_mod.BackboneSpec("custom_net", factory=f"{__name__}:_custom_factory"))

    cfg = _stem_cfg(minimal_cfg, backbone="custom_net")
    _, deepest, feat = backbone_mod.build_stem(cfg)
    assert deepest.name == "backbone" and feat.shape == (None, 32, 32, 4)
    _, pyramid, feat32 = backbone_mod.build_stem(cfg, strides=32)
    assert pyramid.name == "backbone" and feat32.shape == (None, 32, 32, 4)


def test_build_optimizer_selects_by_type(minimal_cfg):
    import keras

    from cvbench.core.optim import build_optimizer

    assert isinstance(build_optimizer(minimal_cfg), keras.optimizers.Adam)
    minimal_cfg.training.optimizer.type = "sgd"
    minimal_cfg.training.optimizer.momentum = 0.9
    opt = build_optimizer(minimal_cfg)
    assert isinstance(opt, keras.optimizers.SGD) and opt.momentum == 0.9
