"""Data management commands: generate synthetic datasets and explore data quality."""

import hashlib
import random
import secrets
import shutil
from pathlib import Path

import click
import numpy as np

from cvbench.cli.generate import generate
from cvbench.core.data import get_class_distribution, print_class_distribution

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_TOKEN_LEN = 16
_MAX_RETRIES = 10


def _fresh_token(used: set) -> str:
    for _ in range(10_000):
        token = secrets.token_hex(8)  # 8 bytes = 16 hex chars
        if token not in used:
            return token
    raise RuntimeError("Could not generate a unique token after 10,000 attempts.")


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


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


@data.command("upsample")
@click.argument("src_dir")
@click.argument("dst_dir")
@click.option("--augmentation", "aug_file", required=True,
              help="Augmentation YAML spec file.")
@click.option("--target", required=True, type=int,
              help="Target number of images in the output folder.")
def upsample(src_dir, dst_dir, aug_file, target):
    """Upsample a class folder to TARGET images using augmentation.

    SRC_DIR  source class folder (e.g. data/train/dog)\n
    DST_DIR  destination class folder (e.g. data_aug/train/dog)\n

    All originals are copied first with fresh random hex filenames,
    then augmented variants are generated until TARGET is reached.
    DST_DIR must be empty or non-existent.
    """
    from PIL import Image
    from cvbench.core import _fmt
    from cvbench.core.config import load_aug_file
    from cvbench.augmentations.pipeline import build_aug_pipeline

    src = Path(src_dir)
    dst = Path(dst_dir)

    if not src.is_dir():
        raise click.ClickException(f"Source directory not found: '{src}'")

    images = sorted(f for f in src.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    if not images:
        raise click.ClickException(f"No images found in '{src}'")

    n_src = len(images)

    if n_src >= target:
        raise click.ClickException(
            f"'{src.name}' already has {n_src} images — "
            f"use 'data downsample' to reduce to {target}."
        )

    if dst.exists():
        existing = [f for f in dst.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        if existing:
            raise click.ClickException(
                f"Destination '{dst}' already contains {len(existing)} image(s). "
                "Provide an empty or non-existent directory."
            )
    else:
        dst.mkdir(parents=True)

    aug_cfg = load_aug_file(aug_file)
    pipeline = build_aug_pipeline(aug_cfg.transforms)

    used_tokens: set[str] = set()
    used_hashes: set[str] = set()

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — data upsample')}")
    print(_fmt.rule())
    print(f"  Source  : {_fmt.dim(str(src))}  ({n_src} images)")
    print(f"  Target  : {target} images  (+{target - n_src} to generate)")
    print()

    # --- copy originals ---
    print(f" {_fmt.bold('Copying originals...')}")
    for img_path in images:
        token = _fresh_token(used_tokens)
        used_tokens.add(token)
        dst_path = dst / f"{token}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dst_path)
        arr = np.array(Image.open(img_path).convert("RGB"))
        used_hashes.add(_md5(arr))
    print(f"  {_fmt.green('✓')} Copied {n_src} original(s)")
    print()

    # --- generate augmented samples ---
    n_to_generate = target - n_src
    print(f" {_fmt.bold(f'Generating {n_to_generate} augmented sample(s)...')}")

    generated = 0
    total_skipped = 0

    with click.progressbar(length=n_to_generate, width=40) as bar:
        while generated < n_to_generate:
            src_path = random.choice(images)
            arr = np.array(Image.open(src_path).convert("RGB")).astype(np.float32)
            suffix = src_path.suffix.lower()

            aug_uint8 = None
            for _ in range(_MAX_RETRIES):
                candidate = np.clip(pipeline(arr), 0, 255).astype(np.uint8)
                if _md5(candidate) not in used_hashes:
                    aug_uint8 = candidate
                    break

            if aug_uint8 is None:
                total_skipped += 1
                continue

            h = _md5(aug_uint8)
            used_hashes.add(h)
            token = _fresh_token(used_tokens)
            used_tokens.add(token)
            Image.fromarray(aug_uint8).save(dst / f"{token}{suffix}")
            generated += 1
            bar.update(1)

    print()
    print(f"  {_fmt.green('✓')} Generated {generated} augmented image(s)")
    if total_skipped:
        print(f"  {_fmt.yellow('⚠')}  Skipped {total_skipped} duplicate(s) after {_MAX_RETRIES} retries each")
    print()
    print(f"  Output  : {_fmt.bold(str(dst))}  ({_fmt.green(str(len(list(dst.iterdir()))))} images total)")
    print(_fmt.rule())
