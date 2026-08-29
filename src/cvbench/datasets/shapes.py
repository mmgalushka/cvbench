"""Geometry for the synthetic shapes dataset — one shape, one bounding box.

Pure math/PIL, no dataset-layout or CLI concerns.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle", "star"]

# Radius (or half-side, for squares) as a fraction of the image size.
_RADIUS_RANGE = {
    "circle":   (0.15, 0.35),
    "square":   (0.12, 0.32),
    "triangle": (0.15, 0.35),
    "star":     (0.18, 0.35),
}

# Keep-out border as a fraction of the image size.
_MARGIN = {
    "circle":   0.15,
    "square":   0.10,
    "triangle": 0.10,
    "star":     0.10,
}

# Detection layout: how hard we try to place a non-overlapping shape.
PLACEMENT_ATTEMPTS = 30
MAX_IOU = 0.15
MAX_COVERAGE = 0.35


def _gray(rng: random.Random) -> int:
    return rng.randint(30, 220)


def noisy_background(size: int, rng: random.Random) -> Image.Image:
    base = np.full((size, size), _gray(rng), dtype=np.int16)
    noise = np.array([rng.randint(-25, 25) for _ in range(size * size)], dtype=np.int16)
    arr = np.clip(base + noise.reshape(size, size), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def _bbox(cx, cy, r):
    return (cx - r, cy - r, cx + r, cy + r)


def circle_shape(cx, cy, r, rng: random.Random):
    return ("ellipse", _bbox(cx, cy, r))


def square_shape(cx, cy, half, rng: random.Random):
    a = math.radians(rng.uniform(-15, 15))
    cos_a, sin_a = math.cos(a), math.sin(a)
    pts = [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]
    ]
    return ("polygon", pts)


def triangle_shape(cx, cy, r, rng: random.Random):
    offset = rng.uniform(0, 2 * math.pi)
    pts = [
        (cx + r * math.cos(offset + i * 2 * math.pi / 3),
         cy + r * math.sin(offset + i * 2 * math.pi / 3))
        for i in range(3)
    ]
    return ("polygon", pts)


def star_shape(cx, cy, r_outer, rng: random.Random):
    r_inner = r_outer * rng.uniform(0.35, 0.50)
    offset = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(5):
        a_out = offset + i * 2 * math.pi / 5
        a_in = a_out + math.pi / 5
        pts.append((cx + r_outer * math.cos(a_out), cy + r_outer * math.sin(a_out)))
        pts.append((cx + r_inner * math.cos(a_in),  cy + r_inner * math.sin(a_in)))
    return ("polygon", pts)


_SHAPE_FN = {
    "circle":   circle_shape,
    "square":   square_shape,
    "triangle": triangle_shape,
    "star":     star_shape,
}


def random_shape(cls: str, size: int, rng: random.Random):
    """Pick a random position and radius for CLS inside a SIZE×SIZE image."""
    lo, hi = _RADIUS_RANGE[cls]
    r = rng.uniform(size * lo, size * hi)
    margin = size * _MARGIN[cls]
    cx = rng.uniform(margin + r, size - margin - r)
    cy = rng.uniform(margin + r, size - margin - r)
    return _SHAPE_FN[cls](cx, cy, r, rng)


def shape_bbox(shape, size: int | None = None):
    """Axis-aligned bounding box of SHAPE, optionally clipped to the image."""
    kind, payload = shape
    if kind == "ellipse":
        x1, y1, x2, y2 = payload
    else:
        xs = [p[0] for p in payload]
        ys = [p[1] for p in payload]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if size is not None:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(size), x2), min(float(size), y2)
    return (x1, y1, x2, y2)


def render(draw: ImageDraw.ImageDraw, shape, fill: int):
    kind, payload = shape
    if kind == "ellipse":
        draw.ellipse(payload, fill=fill)
    else:
        draw.polygon(payload, fill=fill)


def overlaps(a, b) -> bool:
    """True if two boxes overlap enough that one shape would occlude the other.

    Rejects both large mutual overlap (IoU) and a small box sitting inside a
    large one, which IoU alone would happily accept.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return False
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 1.0
    smallest = min(area_a, area_b)
    covered = inter / smallest if smallest > 0 else 1.0
    return iou > MAX_IOU or covered > MAX_COVERAGE


def gray(rng: random.Random) -> int:
    """Public alias of ``_gray`` — used by ``synth.py`` to render shape fill."""
    return _gray(rng)
