"""Synthetic geometric shapes dataset generator.

Produces a 4-class grayscale image dataset (circle, square, triangle, star)
ready to drop into the cvbench data/ folder and train immediately.
"""

import math
import random
import shutil
from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle", "star"]


# ---------------------------------------------------------------------------
# Shape drawing
# ---------------------------------------------------------------------------

def _gray(rng: random.Random) -> int:
    return rng.randint(30, 220)


def _noisy_background(size: int, rng: random.Random) -> Image.Image:
    base = np.full((size, size), _gray(rng), dtype=np.int16)
    noise = np.array([rng.randint(-25, 25) for _ in range(size * size)], dtype=np.int16)
    arr = np.clip(base + noise.reshape(size, size), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def _bbox(cx, cy, r):
    return (cx - r, cy - r, cx + r, cy + r)


def draw_circle(draw: ImageDraw.Draw, size: int, rng: random.Random):
    margin = size * 0.15
    r = rng.uniform(size * 0.15, size * 0.35)
    cx = rng.uniform(margin + r, size - margin - r)
    cy = rng.uniform(margin + r, size - margin - r)
    draw.ellipse(_bbox(cx, cy, r), fill=_gray(rng))


def draw_square(draw: ImageDraw.Draw, size: int, rng: random.Random):
    margin = size * 0.1
    half = rng.uniform(size * 0.12, size * 0.32)
    cx = rng.uniform(margin + half, size - margin - half)
    cy = rng.uniform(margin + half, size - margin - half)
    angle = rng.uniform(-15, 15)
    a = math.radians(angle)
    cos_a, sin_a = math.cos(a), math.sin(a)
    pts = [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]
    ]
    draw.polygon(pts, fill=_gray(rng))


def draw_triangle(draw: ImageDraw.Draw, size: int, rng: random.Random):
    margin = size * 0.1
    r = rng.uniform(size * 0.15, size * 0.35)
    cx = rng.uniform(margin + r, size - margin - r)
    cy = rng.uniform(margin + r, size - margin - r)
    offset = rng.uniform(0, 2 * math.pi)
    pts = [
        (cx + r * math.cos(offset + i * 2 * math.pi / 3),
         cy + r * math.sin(offset + i * 2 * math.pi / 3))
        for i in range(3)
    ]
    draw.polygon(pts, fill=_gray(rng))


def draw_star(draw: ImageDraw.Draw, size: int, rng: random.Random):
    margin = size * 0.1
    r_outer = rng.uniform(size * 0.18, size * 0.35)
    r_inner = r_outer * rng.uniform(0.35, 0.50)
    cx = rng.uniform(margin + r_outer, size - margin - r_outer)
    cy = rng.uniform(margin + r_outer, size - margin - r_outer)
    offset = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(5):
        a_out = offset + i * 2 * math.pi / 5
        a_in = a_out + math.pi / 5
        pts.append((cx + r_outer * math.cos(a_out), cy + r_outer * math.sin(a_out)))
        pts.append((cx + r_inner * math.cos(a_in),  cy + r_inner * math.sin(a_in)))
    draw.polygon(pts, fill=_gray(rng))


_DRAW_FN = {
    "circle":   draw_circle,
    "square":   draw_square,
    "triangle": draw_triangle,
    "star":     draw_star,
}


def generate_image(cls: str, size: int, rng: random.Random) -> Image.Image:
    img = _noisy_background(size, rng)
    _DRAW_FN[cls](ImageDraw.Draw(img), size, rng)
    return img


def generate_split(split: str, output_root: Path, n: int, size: int, rng: random.Random):
    for cls in CLASSES:
        cls_dir = output_root / split / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            generate_image(cls, size, rng).save(cls_dir / f"{i:04d}.jpg", quality=90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("output", default="data/synthetic")
@click.option("--image-size", default=64, show_default=True,
              help="Width and height of generated images.")
@click.option("--train", "n_train", default=200, show_default=True,
              help="Images per class in the train split.")
@click.option("--val", "n_val", default=50, show_default=True,
              help="Images per class in the val split.")
@click.option("--test", "n_test", default=50, show_default=True,
              help="Images per class in the test split.")
@click.option("--seed", default=42, show_default=True,
              help="Random seed for reproducibility.")
@click.option("--overwrite", is_flag=True, default=False,
              help="Delete and recreate output directory if it exists.")
def generate(output, image_size, n_train, n_val, n_test, seed, overwrite):
    """Generate a synthetic geometric shapes dataset for pipeline testing."""
    out = Path(output)

    if out.exists():
        if overwrite:
            shutil.rmtree(out)
        else:
            raise click.ClickException(
                f"Output directory '{out}' already exists. Use --overwrite to replace it."
            )

    rng = random.Random(seed)
    total = (n_train + n_val + n_test) * len(CLASSES)

    w = 55
    print("━" * w)
    print(" CVBench — generate synthetic dataset")
    print("━" * w)
    print(f" Classes    : {', '.join(CLASSES)}")
    print(f" Image size : {image_size}×{image_size}  grayscale")
    print(f" Train      : {n_train} per class  ({n_train * len(CLASSES)} total)")
    print(f" Val        : {n_val}  per class  ({n_val  * len(CLASSES)} total)")
    print(f" Test       : {n_test}  per class  ({n_test * len(CLASSES)} total)")
    print(f" Output     : {out}/")
    print("━" * w)

    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        print(f" Generating {split} ({n * len(CLASSES)} images)...", end=" ", flush=True)
        generate_split(split, out, n, image_size, rng)
        print("done")

    print("━" * w)
    print(f" {total} images written to {out}/")
    print("━" * w)
