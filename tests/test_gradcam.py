"""Grad-CAM service: output contract, determinism, flat vs nested backbones, failure modes."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.tf

from cvbench.core.errors import ConfigError
from cvbench.services import gradcam  # noqa: E402

SIZE = 32


def _flat_model(n_classes=3, activation="relu"):
    import keras

    inputs = keras.Input(shape=(SIZE, SIZE, 3))
    x = keras.layers.Rescaling(1 / 255.0)(inputs)  # keeps the softmax out of saturation
    x = keras.layers.Conv2D(4, 3, activation=activation)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, keras.layers.Dense(n_classes, activation="softmax")(x))


def _nested_model(n_classes=3):
    """A Conv2D inside a functional sub-model, the shape keras-hub backbones give."""
    import keras

    inner_in = keras.Input(shape=(SIZE, SIZE, 3))
    inner_out = keras.layers.Conv2D(4, 3, activation="relu")(inner_in)
    backbone = keras.Model(inner_in, inner_out, name="backbone")

    inputs = keras.Input(shape=(SIZE, SIZE, 3))
    x = keras.layers.Rescaling(1 / 255.0)(inputs)
    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, keras.layers.Dense(n_classes, activation="softmax")(x))


def _dense_only_model():
    import keras

    inputs = keras.Input(shape=(SIZE, SIZE, 3))
    x = keras.layers.GlobalAveragePooling2D()(inputs)
    return keras.Model(inputs, keras.layers.Dense(3, activation="softmax")(x))


def _save(model, tmp_path) -> str:
    path = tmp_path / "model.keras"
    model.save(str(path))
    return str(path)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "img.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (48, 40, 3), dtype=np.uint8)).save(path)
    return str(path)


def _decode(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))))


@pytest.mark.parametrize("build", [_flat_model, _nested_model], ids=["flat", "nested"])
def test_compute_gradcam_returns_a_model_sized_rgb_png(build, tmp_path, image_path):
    checkpoint = _save(build(), tmp_path)

    out = _decode(gradcam.compute_gradcam(checkpoint, image_path, class_index=1))

    assert out.shape == (SIZE, SIZE, 3) and out.dtype == np.uint8


def test_compute_gradcam_is_deterministic(tmp_path, image_path):
    checkpoint = _save(_nested_model(), tmp_path)

    first = gradcam.compute_gradcam(checkpoint, image_path, 0)
    assert gradcam.compute_gradcam(checkpoint, image_path, 0) == first


def test_gradcam_depends_on_the_target_class(tmp_path, image_path):
    model = _flat_model(activation=None)
    model.layers[-1].set_weights([np.eye(4, 3, dtype="float32"), np.zeros(3, "float32")])  # class c <- channel c
    checkpoint = _save(model, tmp_path)

    by_class = {gradcam.compute_gradcam(checkpoint, image_path, c) for c in range(3)}

    assert len(by_class) > 1


def test_compute_gradcam_from_bytes_matches_the_path_variant_and_cleans_up(tmp_path, image_path, monkeypatch):
    checkpoint = _save(_flat_model(), tmp_path)
    with open(image_path, "rb") as f:
        raw = f.read()
    seen = []
    real = gradcam.compute_gradcam
    monkeypatch.setattr(gradcam, "compute_gradcam", lambda c, p, i: seen.append(p) or real(c, p, i))

    via_bytes = gradcam.compute_gradcam_from_bytes(checkpoint, raw, 2)

    assert via_bytes == real(checkpoint, image_path, 2)
    import os
    assert len(seen) == 1 and not os.path.exists(seen[0])


def test_compute_gradcam_from_bytes_cleans_up_when_the_model_fails(tmp_path, monkeypatch):
    seen = []

    def failing(_checkpoint, path, _index):
        seen.append(path)
        raise RuntimeError("boom")

    monkeypatch.setattr(gradcam, "compute_gradcam", failing)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")

    with pytest.raises(RuntimeError, match="boom"):
        gradcam.compute_gradcam_from_bytes("ignored.keras", buf.getvalue(), 0)

    import os
    assert not os.path.exists(seen[0])


def test_compute_gradcam_requires_a_conv_layer(tmp_path, image_path):
    checkpoint = _save(_dense_only_model(), tmp_path)

    with pytest.raises(ConfigError, match="No Conv2D layer"):
        gradcam.compute_gradcam(checkpoint, image_path, 0)


def test_find_backbone_and_last_conv():
    backbone, conv = gradcam._find_backbone_and_last_conv(_flat_model())
    assert backbone is None and conv is not None

    backbone, conv = gradcam._find_backbone_and_last_conv(_nested_model())
    assert backbone is not None and backbone.name == "backbone" and conv in backbone.layers

    assert gradcam._find_backbone_and_last_conv(_dense_only_model()) == (None, None)


def test_colorize_resizes_and_maps_to_the_jet_ramp():
    cam = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)

    heat = gradcam._colorize(cam, 16)

    assert heat.shape == (16, 16, 3) and heat.dtype == np.uint8
    low, high = heat[0, 0], heat[0, -1]
    assert low[2] > low[0]    # cold end is blue
    assert high[0] > high[2]  # hot end is red


def test_blend_is_clipped_and_respects_alpha():
    original = np.full((4, 4, 3), 200, dtype=np.uint8)
    heatmap = np.full((4, 4, 3), 255, dtype=np.uint8)

    assert np.all(gradcam._blend(original, heatmap, alpha=0.0) == 200)
    assert np.all(gradcam._blend(original, heatmap, alpha=1.0) == 255)
    mixed = gradcam._blend(original, heatmap, alpha=0.5)
    assert mixed.dtype == np.uint8 and np.all((mixed > 200) & (mixed < 255))


def test_to_base64_png_round_trips():
    arr = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    assert np.array_equal(_decode(gradcam._to_base64_png(arr)), arr)
