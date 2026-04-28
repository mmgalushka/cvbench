"""Grad-CAM explanation service.

Computes a Grad-CAM heatmap overlay for a single image given a Keras model
checkpoint and a target class index.  Returns a base64-encoded PNG that can
be embedded directly in an <img src="data:image/png;base64,..."> tag.

The last Conv2D layer is located automatically, so this works with any
standard CNN backbone (EfficientNet, MobileNet, ResNet, etc.).
"""

from __future__ import annotations

import base64
import io

import numpy as np


def compute_gradcam_from_bytes(checkpoint: str, image_bytes: bytes, class_index: int) -> str:
    """Like compute_gradcam but accepts raw image bytes instead of a file path."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name
    try:
        return compute_gradcam(checkpoint, tmp_path, class_index)
    finally:
        os.unlink(tmp_path)


def compute_gradcam(checkpoint: str, image_path: str, class_index: int) -> str:
    """Return a base64 PNG of the Grad-CAM heatmap blended onto the original image.

    Args:
        checkpoint:  Path to the .keras model file.
        image_path:  Path to the image file.
        class_index: Target class index (the predicted or true class to explain).

    Returns:
        Base64-encoded PNG string (no data-URI prefix).
    """
    import tensorflow as tf
    import keras
    from PIL import Image as PILImage

    model = keras.saving.load_model(checkpoint)
    input_size = model.input_shape[1]

    img = PILImage.open(image_path).convert("RGB")
    img = img.resize((input_size, input_size))
    img_rgb = np.array(img)
    arr = tf.cast(img_rgb.astype("float32")[None], tf.float32)  # (1,H,W,3) RGB

    with tf.GradientTape() as tape:
        conv_outputs, predictions = _run_with_conv_output(model, arr, tape=tape)
        if conv_outputs is None:
            raise ValueError("No Conv2D layer found in model — Grad-CAM requires a CNN backbone.")
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)          # (1, h, w, filters)
    pooled = tf.reduce_mean(grads, axis=(1, 2))[0]     # (filters,)
    cam = tf.reduce_sum(conv_outputs[0] * pooled, axis=-1).numpy()  # (h, w)

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    heatmap = _colorize(cam, input_size)               # (H, W, 3) uint8 RGB
    original = img_rgb.astype(np.uint8)                # (H, W, 3) RGB
    blended = _blend(original, heatmap, alpha=0.45)    # (H, W, 3) RGB

    return _to_base64_png(blended)


# ── helpers ──────────────────────────────────────────────────────────────────

def _run_with_conv_output(model, arr, tape=None):
    """Run the model and return (last_conv_output, predictions).

    Handles nested backbones by splitting model.layers into three ordered groups
    and running them sequentially: pre_layers → backbone grad-model → post_layers.
    This avoids a Keras graph-connectivity error when last_conv.output belongs to
    the backbone's sub-graph rather than the top-level model's graph.

    Returns (None, None) when no Conv2D is found.
    """
    import keras

    backbone, last_conv = _find_backbone_and_last_conv(model)
    if last_conv is None:
        return None, None

    if backbone is not None:
        # Partition layers: everything before backbone, backbone itself, everything after
        pre_layers, post_layers = [], []
        found = False
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            if layer is backbone:
                found = True
                continue
            (post_layers if found else pre_layers).append(layer)

        # Grad model scoped entirely to the backbone sub-graph
        conv_model = keras.Model(
            inputs=backbone.inputs,
            outputs=[last_conv.output, backbone.outputs[0]],
        )

        # Pre-backbone pass (rescaling, etc.)
        x = arr
        for layer in pre_layers:
            x = layer(x, training=False)

        # Backbone pass — intercept conv activations here
        conv_outputs, x = conv_model(x, training=False)
        if tape is not None:
            tape.watch(conv_outputs)

        # Post-backbone head (GlobalAveragePooling, Dense, Dropout, …)
        for layer in post_layers:
            x = layer(x, training=False)

        return conv_outputs, x
    else:
        # Flat model — last_conv.output is directly reachable from model.inputs
        conv_model = keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.outputs[0]],
        )
        conv_outputs, predictions = conv_model(arr, training=False)
        if tape is not None:
            tape.watch(conv_outputs)
        return conv_outputs, predictions


def _find_backbone_and_last_conv(model):
    """Return (backbone_layer_or_None, last_Conv2D_layer_or_None)."""
    import keras

    backbone = None
    last_conv = None

    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = layer
        elif hasattr(layer, "layers") and hasattr(layer, "inputs"):
            # Nested functional sub-model — treat as backbone
            backbone = layer
            for sub in layer.layers:
                if isinstance(sub, keras.layers.Conv2D):
                    last_conv = sub

    return backbone, last_conv


def _colorize(cam: np.ndarray, size: int) -> np.ndarray:
    """Resize CAM to (size, size) and apply jet colormap → uint8 RGB."""
    from PIL import Image

    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_img = Image.fromarray(cam_uint8, mode="L").resize(
        (size, size), Image.BILINEAR
    )
    cam_arr = np.array(cam_img, dtype=np.float32) / 255.0  # [0,1]

    # Jet colormap: blue→cyan→green→yellow→red
    r = np.clip(1.5 - np.abs(cam_arr * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(cam_arr * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(cam_arr * 4 - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _blend(original: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend heatmap over original image."""
    return np.clip(
        original.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha,
        0, 255,
    ).astype(np.uint8)


def _to_base64_png(arr: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
