from pathlib import Path

import click
import keras
import numpy as np


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Path to .keras model file.")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="Image file or folder of images.")
def predict(checkpoint, input_path):
    """Run inference on an image or folder of images."""
    import tensorflow as tf

    model = keras.saving.load_model(checkpoint)
    size = model.input_shape[1]

    paths = _collect_images(input_path)
    if not paths:
        raise click.ClickException(f"No images found at: {input_path}")

    w = 55
    print("━" * w)
    print(" CVBench — predict")
    print("━" * w)

    for img_path in paths:
        img = tf.keras.utils.load_img(img_path, target_size=(size, size))
        arr = tf.keras.utils.img_to_array(img)[None]  # (1, H, W, 3)
        probs = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        print(f" {Path(img_path).name:<35} class {top_idx}  ({confidence * 100:.1f}%)")

    print("━" * w)


def _collect_images(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in _IMAGE_EXTS else []
    return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in _IMAGE_EXTS)
