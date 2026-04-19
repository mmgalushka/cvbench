"""Data management commands: generate synthetic datasets and explore data quality."""

from pathlib import Path

import click
import numpy as np

from cvbench.cli.generate import generate
from cvbench.core.data import get_class_distribution, print_class_distribution

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@click.group()
def data():
    """Manage and inspect datasets."""


data.add_command(generate, name="generate")


def _mean_brightness(path: Path) -> float:
    from PIL import Image
    return float(np.array(Image.open(path).convert("L")).mean())


@data.command("explore")
@click.argument("data_dir")
@click.option("--split", default="train", show_default=True,
              help="Dataset split to analyse (train / val / test).")
def explore(data_dir, split):
    """Analyse per-class brightness and class distribution to detect potential bias.

    DATA_DIR is the root dataset directory (containing train/, val/, test/
    subdirectories) or a split directory directly.
    """
    from cvbench.core import _fmt

    root = Path(data_dir)
    split_dir = root / split if (root / split).is_dir() else root

    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise click.ClickException(f"No class subdirectories found in '{split_dir}'")

    stats = []
    for cls_dir in class_dirs:
        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        if not images:
            continue
        brightnesses = [_mean_brightness(f) for f in images]
        arr = np.array(brightnesses)
        stats.append({
            "class": cls_dir.name,
            "count": len(images),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        })

    if not stats:
        raise click.ClickException("No images found.")

    means = [s["mean"] for s in stats]
    dataset_mean = float(np.mean(means))
    std_of_means = float(np.std(means))
    max_cls = max(len(s["class"]) for s in stats)

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — data explore')}  {_fmt.dim('|')}  {_fmt.dim(str(split_dir))}")
    print(_fmt.rule())
    print(f" {_fmt.bold('Brightness distribution per class')}  {_fmt.dim('[0–255 scale]')}")
    print()
    print(_fmt.dim(f"   {'Class':<{max_cls}}  {'Images':>7}  {'Mean':>6}  {'Std':>6}  {'Min':>5}  {'Max':>5}"))

    biased = [s for s in stats if std_of_means > 0 and abs(s["mean"] - dataset_mean) > std_of_means]

    for s in stats:
        flag = f"  {_fmt.yellow('⚠️')}" if s in biased else ""
        mean_str = _fmt.bold(f"{s['mean']:>6.1f}")
        print(
            f"   {s['class']:<{max_cls}}  {s['count']:>7}  "
            f"{mean_str}  {s['std']:>6.1f}  "
            f"{s['min']:>5.1f}  {s['max']:>5.1f}{flag}"
        )

    print()
    print(f" Dataset mean brightness : {_fmt.bold(f'{dataset_mean:.1f}')}")

    if biased:
        print()
        for s in biased:
            dev = s["mean"] - dataset_mean
            direction = "brighter" if dev > 0 else "darker"
            print(_fmt.yellow(f" ⚠️  '{s['class']}' is {abs(dev):.1f} units {direction} than the dataset mean"))
        print()
        print(f"   {_fmt.dim('Suggestion: consider brightness augmentation or per-image normalization.')}")
    else:
        print(_fmt.green(" ✓ No significant brightness bias detected."))

    print()
    dist = get_class_distribution(str(split_dir))
    print_class_distribution(dist)
    counts = list(dist.values())
    std_of_counts = float(np.std(counts))
    mean_of_counts = float(np.mean(counts))
    imbalanced = std_of_counts > 0 and any(abs(c - mean_of_counts) > std_of_counts for c in counts)
    if not imbalanced:
        print(_fmt.green(" ✓ No significant class imbalance detected."))
    print(_fmt.rule())
