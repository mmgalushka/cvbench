"""Data management commands: generate synthetic datasets and explore data quality."""

import random
import secrets
import shutil
from pathlib import Path

import click
import numpy as np

from cvbench.cli import _help
from cvbench.cli.augmentations import augmentations
from cvbench.cli.generate import generate
from cvbench.datasets import clean as clean_mod
from cvbench.datasets import dedup as dedup_mod
from cvbench.datasets import flatten as flatten_mod
from cvbench.datasets import hashify as hashify_mod
from cvbench.datasets import split as split_mod
from cvbench.datasets.stats import get_class_distribution, print_class_distribution

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_TOKEN_LEN = 16
_MAX_RETRIES = 10


def _fresh_token(used: set) -> str:
    for _ in range(10_000):
        token = secrets.token_hex(8)  # 8 bytes = 16 hex chars
        if token not in used:
            return token
    raise RuntimeError("Could not generate a unique token after 10,000 attempts.")


@_help.group(
    examples=[
        ("data generate", "make a synthetic dataset to experiment with"),
        ("data split data/pool data/ready", "turn a flat folder into train/val/test"),
        ("data explore data/ready", "check class balance and brightness bias"),
    ],
)
def data():
    """Generate, inspect and reshape datasets."""


data.add_command(generate, name="generate")
data.add_command(augmentations, name="aug")


def _mean_brightness(path: Path) -> float:
    from PIL import Image
    return float(np.array(Image.open(path).convert("L")).mean())


@data.command(
    "explore",
    short_help="Report per-class brightness and class balance.",
    examples=[
        ("data explore data/ready", "Analyse the train split"),
        ("data explore data/ready --split test", "Analyse a different split"),
    ],
    see_also=[("data upsample <src> <dst> --augmentation aug.yaml --target 500",
               "grow an under-represented class")],
)
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


@data.command(
    "upsample",
    short_help="Grow a class folder to TARGET images via augmentation.",
    examples=[
        ("data upsample data/train/dog data_aug/train/dog --augmentation aug.yaml --target 1500",
         "Copy originals, then add augmented variants until the folder holds 1500 images"),
    ],
    see_also=[("data aug generate", "make an augmentation spec first")],
)
@click.argument("src_dir")
@click.argument("dst_dir")
@click.option("--augmentation", "aug_file", required=True,
              help="Augmentation YAML spec file, or the name of a saved 'data aug' config.")
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
    from cvbench.core.augmentations_store import resolve_aug_file
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

    aug_cfg = load_aug_file(resolve_aug_file(aug_file))
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
        used_hashes.add(hashify_mod.hash_array(arr))
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
                if hashify_mod.hash_array(candidate) not in used_hashes:
                    aug_uint8 = candidate
                    break

            if aug_uint8 is None:
                total_skipped += 1
                continue

            h = hashify_mod.hash_array(aug_uint8)
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


@data.command(
    "clean",
    short_help="Copy a dataset, dropping OS/editor junk files.",
    examples=[
        ("data clean data/raw data/clean", "Drop .DS_Store, Thumbs.db, ._* and editor temp files"),
        ("data clean data/raw data/clean --dry-run", "List the junk without writing anything"),
    ],
)
@click.argument("src")
@click.argument("dst")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be removed without writing DST.")
def clean(src, dst, dry_run):
    """Copy SRC to DST, dropping OS/editor junk.

    SRC  dataset directory to clean (classification or YOLO layout)\n
    DST  destination for the cleaned copy; must be empty or non-existent.

    Drops Finder/Explorer metadata (.DS_Store, Thumbs.db, __MACOSX/, ...),
    AppleDouble shadow files (._*), and editor swap/temp files. Directories
    left empty by junk removal are simply not created at DST. SRC is never
    modified.
    """
    from cvbench.core import _fmt

    src_dir = Path(src)
    dst_dir = Path(dst)

    if not src_dir.is_dir():
        raise click.ClickException(f"Source directory not found: '{src_dir}'")

    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise click.ClickException(
            f"Destination '{dst_dir}' already contains files. "
            "Provide an empty or non-existent directory."
        )

    plan = clean_mod.clean_dataset(src_dir, dst_dir, dry_run)

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — data clean')}")
    print(_fmt.rule())
    print(f"  Source  : {_fmt.dim(str(src_dir))}")
    print(f"  Dest    : {_fmt.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    print()

    n_junk = len(plan.junk_files) + len(plan.junk_dirs)
    if n_junk:
        print(f" {_fmt.bold('Junk found:')}")
        for rel in plan.junk_dirs:
            print(f"   {_fmt.yellow('⚠')}  {rel}/  {_fmt.dim('(directory)')}")
        for rel in plan.junk_files:
            print(f"   {_fmt.yellow('⚠')}  {rel}")
    else:
        print(f" {_fmt.green('✓')} No junk found.")

    print()
    verb = "Would keep" if dry_run else "Kept"
    suffix = f"  {_fmt.dim(f'({n_junk} junk item(s) skipped)')}" if n_junk else ""
    print(f"  {_fmt.green('✓')} {verb} {len(plan.keep)} file(s){suffix}")
    print(_fmt.rule())


@data.command(
    "hashify",
    short_help="Copy a dataset, renaming images to content hashes.",
    examples=[
        ("data hashify data/raw data/hashed",
         "Give every image a deterministic content-based filename"),
        ("data hashify data/raw data/hashed --dry-run", "Preview the renames"),
    ],
    see_also=[("data dedup data/hashed data/final", "then drop genuine duplicates")],
)
@click.argument("src")
@click.argument("dst")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be renamed without writing DST.")
def hashify(src, dst, dry_run):
    """Copy SRC to DST, renaming every image to a content-hash filename.

    SRC  dataset directory to hashify (classification or YOLO layout)\n
    DST  destination for the renamed copy; must be empty or non-existent.

    Names are derived from pixel content, so the same image gets the same
    16-hex-char filename every time (deterministic, idempotent) regardless
    of its original basename. Never deletes: two images that land on the
    same destination name both survive, the second with a numeric suffix.
    Use 'data dedup' to remove genuine duplicates. YOLO label files are
    renamed to match their image's new name. SRC is never modified.
    """
    from cvbench.core import _fmt

    src_dir = Path(src)
    dst_dir = Path(dst)

    if not src_dir.is_dir():
        raise click.ClickException(f"Source directory not found: '{src_dir}'")

    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise click.ClickException(
            f"Destination '{dst_dir}' already contains files. "
            "Provide an empty or non-existent directory."
        )

    plan = hashify_mod.hashify_dataset(src_dir, dst_dir, dry_run)

    collisions = sum(1 for a in plan.actions if "-" in Path(a.dst_image).stem)

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — data hashify')}")
    print(_fmt.rule())
    print(f"  Source  : {_fmt.dim(str(src_dir))}")
    print(f"  Dest    : {_fmt.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    print()

    verb = "Would rename" if dry_run else "Renamed"
    print(f"  {_fmt.green('✓')} {verb} {len(plan.actions)} image(s)")
    if collisions:
        print(f"  {_fmt.yellow('⚠')}  {collisions} filename collision(s) resolved with a numeric suffix")
    print(_fmt.rule())


@data.command(
    "dedup",
    short_help="Copy a dataset, dropping exact-duplicate images.",
    examples=[
        ("data dedup data/raw data/deduped", "Keep one copy of each image"),
        ("data dedup data/split data/deduped --across-splits",
         "Also warn when the same image appears in more than one split"),
    ],
)
@click.argument("src")
@click.argument("dst")
@click.option("--across-splits", is_flag=True, default=False,
              help="Warn when a duplicate group spans more than one split (train/val/test).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be removed without writing DST.")
def dedup(src, dst, across_splits, dry_run):
    """Copy SRC to DST, dropping exact-duplicate images.

    SRC  dataset directory to dedup (classification or YOLO layout)\n
    DST  destination for the deduplicated copy; must be empty or non-existent.

    Duplicates are found by full image-content hash. Within each duplicate
    group only the lexicographically-first path is kept. For YOLO, dropping
    an image also drops its paired label file. SRC is never modified. Use
    'data hashify' first if you also want canonical filenames.
    """
    from cvbench.core import _fmt

    src_dir = Path(src)
    dst_dir = Path(dst)

    if not src_dir.is_dir():
        raise click.ClickException(f"Source directory not found: '{src_dir}'")

    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise click.ClickException(
            f"Destination '{dst_dir}' already contains files. "
            "Provide an empty or non-existent directory."
        )

    plan = dedup_mod.dedup_dataset(src_dir, dst_dir, dry_run)

    n_dupe_files = sum(len(v) - 1 for v in plan.duplicate_groups.values())

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — data dedup')}")
    print(_fmt.rule())
    print(f"  Source  : {_fmt.dim(str(src_dir))}")
    print(f"  Dest    : {_fmt.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    print()

    if plan.duplicate_groups:
        print(f" {_fmt.bold(f'{len(plan.duplicate_groups)} duplicate group(s) found:')}")
        for h, paths in plan.duplicate_groups.items():
            kept, dupes = paths[0], paths[1:]
            print(f"   {_fmt.dim(h[:8])}  {_fmt.green(str(kept))} (kept)")
            for p in dupes:
                print(f"   {' ' * 8}  {_fmt.yellow(str(p))} (dropped)")
    else:
        print(f" {_fmt.green('✓')} No duplicates found.")

    if across_splits:
        print()
        if plan.cross_split_leaks:
            print(f" {_fmt.yellow(f'⚠️  {len(plan.cross_split_leaks)} duplicate group(s) leak across splits:')}")
            for h, paths in plan.cross_split_leaks.items():
                print(f"   {_fmt.dim(h[:8])}  {', '.join(str(p) for p in paths)}")
        else:
            print(f" {_fmt.green('✓')} No cross-split leakage detected.")

    print()
    verb = "Would keep" if dry_run else "Kept"
    suffix = f"  {_fmt.dim(f'({n_dupe_files} duplicate(s) dropped)')}" if n_dupe_files else ""
    print(f"  {_fmt.green('✓')} {verb} {len(plan.keep)} image(s){suffix}")
    print(_fmt.rule())


@data.command(
    "split",
    short_help="Split a flat dataset into train/val/test, stratified by class.",
    examples=[
        ("data split data/pool data/ready", "Default 80/10/10 stratified split"),
        ("data split data/pool data/ready --train 0.7 --val 0.15 --test 0.15",
         "Custom ratios — the three must sum to 1.0"),
        ("data split data/pool data/ready --dry-run", "Preview the per-class counts"),
    ],
    see_also=[("train data/ready --epochs 20", "train on the result")],
)
@click.argument("src")
@click.argument("dst")
@click.option("--train", "train_ratio", default=0.8, show_default=True, type=float,
              help="Fraction of images assigned to the train split.")
@click.option("--val", "val_ratio", default=0.1, show_default=True, type=float,
              help="Fraction of images assigned to the val split.")
@click.option("--test", "test_ratio", default=0.1, show_default=True, type=float,
              help="Fraction of images assigned to the test split.")
@click.option("--seed", default=42, show_default=True, type=int,
              help="Random seed for the stratified shuffle.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show the planned split without writing DST.")
def split(src, dst, train_ratio, val_ratio, test_ratio, seed, dry_run):
    """Copy SRC to DST, split into train/val/test, stratified by class.

    SRC  dataset directory to split — a flat pool (classification:
    <class>/*; YOLO: images/* + labels/*) or an already-split dataset,
    which is pooled back together before re-partitioning.\n
    DST  destination for the split dataset; must be empty or non-existent.

    Classification stratifies by class folder. YOLO images can carry boxes
    of more than one class, so the stratification key is each image's
    primary (most frequent, ties broken by lowest id) box class; images
    with no boxes are split the same proportional, seeded way as every
    other group. SRC is never modified.
    """
    from cvbench.core import _fmt

    src_dir = Path(src)
    dst_dir = Path(dst)

    if not src_dir.is_dir():
        raise click.ClickException(f"Source directory not found: '{src_dir}'")

    ratios = (train_ratio, val_ratio, test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise click.ClickException(
            f"--train/--val/--test must sum to 1.0 (got {sum(ratios):.4f})."
        )

    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise click.ClickException(
            f"Destination '{dst_dir}' already contains files. "
            "Provide an empty or non-existent directory."
        )

    try:
        plan = split_mod.split_dataset(src_dir, dst_dir, ratios, seed, dry_run)
    except ValueError as e:
        raise click.ClickException(str(e))

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — data split')}")
    print(_fmt.rule())
    print(f"  Source  : {_fmt.dim(str(src_dir))}  ({_fmt.dim('yolo' if plan.is_yolo else 'classification')})")
    print(f"  Dest    : {_fmt.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    print(f"  Ratios  : train={train_ratio}  val={val_ratio}  test={test_ratio}  seed={seed}")
    print()

    classes = sorted({c for counts in plan.counts.values() for c in counts})
    max_cls = max((len(c) for c in classes), default=5)
    print(_fmt.dim(f"   {'Class':<{max_cls}}  {'Train':>7}  {'Val':>7}  {'Test':>7}"))
    for cls in classes:
        row = [plan.counts.get(s, {}).get(cls, 0) for s in ("train", "val", "test")]
        print(f"   {cls:<{max_cls}}  {row[0]:>7}  {row[1]:>7}  {row[2]:>7}")

    print()
    verb = "Would write" if dry_run else "Wrote"
    total = len(plan.actions)
    print(f"  {_fmt.green('✓')} {verb} {total} image(s) across {len(plan.splits_written)} split(s)")
    print(_fmt.rule())


@data.command(
    "flatten",
    short_help="Pool an already-split dataset back into one flat folder.",
    examples=[
        ("data flatten data/ready data/pool", "Undo a split so you can re-partition"),
    ],
    see_also=[("data split data/pool data/ready", "re-split with new ratios or seed")],
)
@click.argument("src")
@click.argument("dst")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show the flatten plan without writing DST.")
def flatten(src, dst, dry_run):
    """Copy SRC to DST, pooling train/val/test back into one flat dataset.

    SRC  an already-split dataset (classification or YOLO layout)\n
    DST  destination for the flattened copy; must be empty or non-existent.

    The exact inverse of 'data split': every image from every split is
    copied into one flat pool (classification: <class>/*; YOLO: images/*
    + labels/*), with no train/val/test structure. Use 'data split'
    afterward to re-partition. SRC is never modified.
    """
    from cvbench.core import _fmt

    src_dir = Path(src)
    dst_dir = Path(dst)

    if not src_dir.is_dir():
        raise click.ClickException(f"Source directory not found: '{src_dir}'")

    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise click.ClickException(
            f"Destination '{dst_dir}' already contains files. "
            "Provide an empty or non-existent directory."
        )

    try:
        plan = flatten_mod.flatten_dataset(src_dir, dst_dir, dry_run)
    except ValueError as e:
        raise click.ClickException(str(e))

    print(_fmt.rule())
    print(f" {_fmt.bold('CVBench — data flatten')}")
    print(_fmt.rule())
    print(f"  Source  : {_fmt.dim(str(src_dir))}  ({_fmt.dim('yolo' if plan.is_yolo else 'classification')})")
    print(f"  Dest    : {_fmt.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    print()

    classes = sorted(plan.counts)
    max_cls = max((len(c) for c in classes), default=5)
    print(_fmt.dim(f"   {'Class':<{max_cls}}  {'Count':>7}"))
    for cls in classes:
        print(f"   {cls:<{max_cls}}  {plan.counts[cls]:>7}")

    print()
    verb = "Would write" if dry_run else "Wrote"
    print(f"  {_fmt.green('✓')} {verb} {len(plan.actions)} image(s)")
    print(_fmt.rule())


