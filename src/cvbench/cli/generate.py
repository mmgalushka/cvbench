"""``cvbench data generate`` — synthetic geometric shapes dataset generator.

Produces a 4-class grayscale image dataset (circle, square, triangle, star)
ready to drop into the cvbench data/ folder and train immediately. The actual
geometry and file-writing logic live in ``cvbench.datasets`` — this module is
Click plumbing only.

Two output formats are supported:

* ``classification`` — one shape per image, foldered by class
  (``<out>/<split>/<class>/*.jpg``).
* ``yolo`` — several shapes per image with YOLO txt annotations
  (``<out>/images/<split>/*.jpg`` + ``<out>/labels/<split>/*.txt`` + ``data.yaml``).
"""

import random
import shutil
from pathlib import Path

import click

from cvbench.datasets.shapes import CLASSES
from cvbench.datasets.synth import FORMATS, generate_split, generate_yolo_split, write_data_yaml


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
