"""Synthetic geometric shapes dataset generator.

Produces a 4-class grayscale image dataset (circle, square, triangle, star)
ready to drop into the cvbench data/ folder and train immediately.

Two output formats are supported:

* ``classification`` — one shape per image, foldered by class
  (``<out>/<split>/<class>/*.jpg``).
* ``yolo`` — several shapes per image with YOLO txt annotations
  (``<out>/images/<split>/*.jpg`` + ``<out>/labels/<split>/*.txt`` + ``data.yaml``).
"""

import math
import random
import shutil
from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle", "star"]

FORMATS = ["classification", "yolo"]

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
_PLACEMENT_ATTEMPTS = 30
_MAX_IOU = 0.15
_MAX_COVERAGE = 0.35


# ---------------------------------------------------------------------------
# Shape geometry
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


def _render(draw: ImageDraw.ImageDraw, shape, fill: int):
    kind, payload = shape
    if kind == "ellipse":
        draw.ellipse(payload, fill=fill)
    else:
        draw.polygon(payload, fill=fill)


def _overlaps(a, b) -> bool:
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
    return iou > _MAX_IOU or covered > _MAX_COVERAGE


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_image(cls: str, size: int, rng: random.Random) -> Image.Image:
    """Single centred-ish shape on a noisy background (classification format)."""
    img = _noisy_background(size, rng)
    _render(ImageDraw.Draw(img), random_shape(cls, size, rng), _gray(rng))
    return img


def generate_detection_image(size: int, rng: random.Random, max_objects: int):
    """Several shapes on a noisy background, with their bounding boxes.

    Returns ``(image, boxes)`` where each box is ``(class_name, (x1, y1, x2, y2))``
    in pixel coordinates.
    """
    img = _noisy_background(size, rng)
    draw = ImageDraw.Draw(img)
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []

    for _ in range(rng.randint(1, max_objects)):
        for _attempt in range(_PLACEMENT_ATTEMPTS):
            cls = rng.choice(CLASSES)
            shape = random_shape(cls, size, rng)
            bbox = shape_bbox(shape, size)
            if any(_overlaps(bbox, other) for _, other in boxes):
                continue
            _render(draw, shape, _gray(rng))
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


# ---------------------------------------------------------------------------
# Split writers
# ---------------------------------------------------------------------------

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
    """Write the Ultralytics-style ``data.yaml`` describing the dataset."""
    lines = [f"path: {output_root}"]
    for split in splits:
        key = "val" if split == "val" else split
        lines.append(f"{key}: images/{split}")
    lines.append("names:")
    lines.extend(f"  {i}: {cls}" for i, cls in enumerate(CLASSES))
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("output", default="data/synthetic")
@click.option("--format", "fmt", type=click.Choice(FORMATS), default="classification",
              show_default=True,
              help="Dataset layout: class folders, or YOLO images/labels + data.yaml.")
@click.option("--image-size", default=64, show_default=True,
              help="Width and height of generated images.")
@click.option("--train", "n_train", default=200, show_default=True,
              help="Images per class in the train split (per split for --format yolo).")
@click.option("--val", "n_val", default=50, show_default=True,
              help="Images per class in the val split (per split for --format yolo).")
@click.option("--test", "n_test", default=50, show_default=True,
              help="Images per class in the test split (per split for --format yolo).")
@click.option("--max-objects", default=3, show_default=True,
              help="Maximum shapes per image (--format yolo only).")
@click.option("--seed", default=42, show_default=True,
              help="Random seed for reproducibility.")
@click.option("--overwrite", is_flag=True, default=False,
              help="Delete and recreate output directory if it exists.")
def generate(output, fmt, image_size, n_train, n_val, n_test, max_objects, seed, overwrite):
    """Generate a synthetic geometric shapes dataset for pipeline testing."""
    out = Path(output)

    if max_objects < 1:
        raise click.ClickException("--max-objects must be at least 1.")

    if out.exists():
        if overwrite:
            shutil.rmtree(out)
        else:
            raise click.ClickException(
                f"Output directory '{out}' already exists. Use --overwrite to replace it."
            )

    rng = random.Random(seed)
    is_yolo = fmt == "yolo"
    per_split = 1 if is_yolo else len(CLASSES)
    total = (n_train + n_val + n_test) * per_split

    w = 55
    print("━" * w)
    print(" CVBench — generate synthetic dataset")
    print("━" * w)
    print(f" Format     : {fmt}")
    print(f" Classes    : {', '.join(CLASSES)}")
    print(f" Image size : {image_size}×{image_size}  grayscale")
    if is_yolo:
        print(f" Objects    : 1–{max_objects} per image")
        print(f" Train      : {n_train} images")
        print(f" Val        : {n_val} images")
        print(f" Test       : {n_test} images")
    else:
        print(f" Train      : {n_train} per class  ({n_train * len(CLASSES)} total)")
        print(f" Val        : {n_val}  per class  ({n_val  * len(CLASSES)} total)")
        print(f" Test       : {n_test}  per class  ({n_test * len(CLASSES)} total)")
    print(f" Output     : {out}/")
    print("━" * w)

    written_splits = []
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        if n <= 0:
            continue
        count = n * per_split
        print(f" Generating {split} ({count} images)...", end=" ", flush=True)
        if is_yolo:
            generate_yolo_split(split, out, n, image_size, rng, max_objects)
        else:
            generate_split(split, out, n, image_size, rng)
        written_splits.append(split)
        print("done")

    if is_yolo:
        write_data_yaml(out, written_splits)
        print(f" Wrote {out / 'data.yaml'}")

    print("━" * w)
    print(f" {total} images written to {out}/")
    print("━" * w)
