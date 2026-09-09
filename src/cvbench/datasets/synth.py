"""Synthetic geometric shapes dataset — image generation and split writers.

Produces a 4-class grayscale image dataset (circle, square, triangle, star) in
either layout:

* ``classification`` — one shape per image, foldered by class
  (``<out>/<split>/<class>/*.jpg``).
* ``yolo`` — several shapes per image with YOLO txt annotations
  (``<out>/images/<split>/*.jpg`` + ``<out>/labels/<split>/*.txt`` + ``data.yaml``).
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import ImageDraw

from cvbench.datasets import layout
from cvbench.datasets.shapes import (
    CLASSES,
    PLACEMENT_ATTEMPTS,
    gray,
    noisy_background,
    overlaps,
    random_shape,
    render,
    shape_bbox,
)

FORMATS = ["classification", "yolo"]


def generate_image(cls: str, size: int, rng: random.Random):
    """Single centred-ish shape on a noisy background (classification format)."""
    img = noisy_background(size, rng)
    render(ImageDraw.Draw(img), random_shape(cls, size, rng), gray(rng))
    return img


def generate_detection_image(size: int, rng: random.Random, max_objects: int):
    """Several shapes on a noisy background, with their bounding boxes.

    Returns ``(image, boxes)`` where each box is ``(class_name, (x1, y1, x2, y2))``
    in pixel coordinates.
    """
    img = noisy_background(size, rng)
    draw = ImageDraw.Draw(img)
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []

    for _ in range(rng.randint(1, max_objects)):
        for _attempt in range(PLACEMENT_ATTEMPTS):
            cls = rng.choice(CLASSES)
            shape = random_shape(cls, size, rng)
            bbox = shape_bbox(shape, size)
            if any(overlaps(bbox, other) for _, other in boxes):
                continue
            render(draw, shape, gray(rng))
            boxes.append((cls, bbox))
            break

    return img, boxes


def to_yolo_line(cls: str, bbox, size: int) -> str:
    """Format one YOLO annotation line: ``class_id xc yc w h`` (normalized)."""
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2 / size
    yc = (y1 + y2) / 2 / size
    w = (x2 - x1) / size
    h = (y2 - y1) / size
    return f"{CLASSES.index(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def generate_split(split: str, output_root: Path, n: int, size: int, rng: random.Random):
    """Write one classification split: ``<root>/<split>/<class>/*.jpg``."""
    for cls in CLASSES:
        cls_dir = output_root / split / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            generate_image(cls, size, rng).save(cls_dir / f"{i:04d}.jpg", quality=90)


def generate_yolo_split(split: str, output_root: Path, n: int, size: int,
                        rng: random.Random, max_objects: int):
    """Write one detection split: ``images/<split>/*.jpg`` + ``labels/<split>/*.txt``."""
    img_dir = output_root / "images" / split
    lbl_dir = output_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        img, boxes = generate_detection_image(size, rng, max_objects)
        stem = f"{i:04d}"
        img.save(img_dir / f"{stem}.jpg", quality=90)
        lines = [to_yolo_line(cls, bbox, size) for cls, bbox in boxes]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def write_data_yaml(output_root: Path, splits: list[str]):
    """Write the Ultralytics-style ``data.yaml`` for the synthetic shapes classes."""
    layout.write_data_yaml(output_root, splits, CLASSES)
