"""Data management commands: generate synthetic datasets and explore data quality."""

import random
import secrets
import shutil
from pathlib import Path
from typing import Any

import click
import numpy as np

from cvbench.cli import _help
from cvbench.cli.generate import generate
from cvbench.core import present
from cvbench.core.data_store import DATA_DIR, DATA_REG
from cvbench.datasets import convert as convert_mod
from cvbench.datasets import flatten as flatten_mod
from cvbench.datasets import hashify as hashify_mod
from cvbench.datasets import layout as layout_mod
from cvbench.datasets import prep as prep_mod
from cvbench.datasets import split as split_mod
from cvbench.datasets.catalog import scan_datasets
from cvbench.datasets.stats import print_class_distribution

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
        ("data convert data/VOC2007 data/voc", "turn a Pascal VOC or COCO dataset into YOLO"),
        ("data explore data/ready", "check class balance and brightness"),
    ],
)
def data():
    """Generate, inspect and reshape datasets."""


data.add_command(generate, name="generate")


@data.command(
    "list",
    short_help="List datasets (default: data/).",
    examples=[
        ("data list", "every dataset under data/"),
    ],
    see_also=[("data explore <dataset>", "check class balance and brightness")],
)
@click.argument("data_dir", default=DATA_DIR)
def list_data(data_dir):
    """List datasets in DATA_DIR (default: data/)."""
    from cvbench.core import console

    DATA_REG.base_dir = data_dir
    names = DATA_REG.list_entries()
    if not names:
        print(f" {present.EMPTY_DATASETS.format(dir=data_dir)}")
        return

    infos = scan_datasets(data_dir, names)
    console.table(
        [(label, align) if align == "right" else label for _, label, align in present.DATASET_COLUMNS],
        [
            [present.present_dataset(info)[key] for key, _, _ in present.DATASET_COLUMNS]
            for info in infos
        ],
    )


def _mean_brightness(path: Path) -> float:
    from PIL import Image
    return float(np.array(Image.open(path).convert("L")).mean())


def _print_group(h, kept, dropped):
    """One line per duplicate group: green = kept, red = dropped."""
    from cvbench.core import console

    parts = [f"{console.green('✓')} {console.green(str(p))}" for p in kept]
    parts += [f"{console.red('✗')} {console.red(str(p))}" for p in dropped]
    print(f"   {console.dim(h[:8])}  {'  '.join(parts)}")


def _print_integrity(root, issues):
    """Print the integrity sections of ``data explore`` (unreadable, incompatible, leaks, duplicates, conflicts)."""
    from cvbench.core import console

    def group_line(h, paths):
        print(f"   {console.dim(h[:8])}  {', '.join(console.yellow(str(p)) for p in paths)}")

    corrupt, leaks, conflicts = issues.corrupt, issues.leaks, issues.label_conflicts
    duplicates = issues.duplicates
    print(f" {console.bold('Data integrity')}  {console.dim('|')}  {console.dim(str(root))}")
    if corrupt:
        print(f" {console.bold(f'{len(corrupt)} unreadable image(s):')}")
        for rel, reason in corrupt:
            print(f"   {console.yellow(str(rel))}  {console.dim(reason)}")
    else:
        console.success("No unreadable images found.")
    if issues.incompatible:
        n = len(issues.incompatible)
        print(f" {console.bold(f'{n} image(s) PIL reads but TensorFlow rejects (training would fail):')}")
        for rel, reason in issues.incompatible:
            print(f"   {console.yellow(str(rel))}  {console.dim(reason)}")
        console.warning("Drop them or re-encode them as PNG with 'data prep --on-incompatible drop|repair'.")
    else:
        console.success("No TensorFlow-incompatible images found.")
    if leaks:
        print(f" {console.bold(f'{len(leaks)} image(s) appear in more than one split:')}")
        for h, paths in leaks.items():
            group_line(h, paths)
    else:
        console.success("No cross-split duplicates found.")
    if duplicates:
        print(f" {console.bold(f'{len(duplicates)} within-split duplicate group(s):')}")
        for h, paths in duplicates.items():
            group_line(h, paths)
        console.warning("Next: run 'data prep <src> <dst>' to remove the duplicates.")
    else:
        console.success("No within-split duplicates found.")
    if conflicts:
        print(f" {console.bold(f'{len(conflicts)} image(s) with conflicting labels:')}")
        for h, paths in conflicts.items():
            group_line(h, paths)
    else:
        console.success("No conflicting labels found.")
    if issues:
        console.warning("Data problems found; run 'data prep <src> <dst>' to remove them.")
    print(console.rule())


# Mean brightness (0–255) gap for one class between two splits that signals a real shift.
_SPLIT_SHIFT_THRESHOLD = 25.0


def _explore_split(split_dir: Path, bad: set[Path]) -> dict[str, float] | None:
    """Print the brightness and class-balance report for SPLIT_DIR.

    Returns the per-class mean brightness, or None if the split holds no images.
    """
    from cvbench.core import console

    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise click.ClickException(f"No class subdirectories found in '{split_dir}'")

    all_images = {
        cls_dir: layout_mod.list_images(cls_dir, exts=layout_mod.READABLE_IMAGE_EXTS)
        for cls_dir in class_dirs
    }
    readable = {cls_dir: [f for f in images if f not in bad] for cls_dir, images in all_images.items()}
    stats: list[dict[str, Any]] = []
    with console.progress(sum(map(len, readable.values())), "Measuring brightness") as advance:
        brightness: dict[Path, list[float]] = {}
        for cls_dir, images in readable.items():
            brightness[cls_dir] = []
            for f in images:
                brightness[cls_dir].append(_mean_brightness(f))
                advance(1)
    for cls_dir, images in readable.items():
        if not images:
            continue
        arr = np.array(brightness[cls_dir])
        stats.append({
            "class": cls_dir.name,
            "count": len(all_images[cls_dir]),
            "unreadable": len(all_images[cls_dir]) - len(images),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        })

    if not stats:
        return None

    means = [s["mean"] for s in stats]
    dataset_mean = float(np.mean(means))
    std_of_means = float(np.std(means))
    max_cls = max(len(s["class"]) for s in stats)

    print(console.rule())
    print(f" {console.bold('CVBench — data explore')}  {console.dim('|')}  {console.dim(str(split_dir))}")
    print(console.rule())
    print(f" {console.bold('Brightness distribution per class')}  {console.dim('[0–255 scale]')}")
    print()
    print(console.dim(f"   {'Class':<{max_cls}}  {'Images':>7}  {'Mean':>6}  {'Std':>6}  {'Min':>5}  {'Max':>5}"))

    biased = [s for s in stats if std_of_means > 0 and abs(s["mean"] - dataset_mean) > std_of_means]

    for s in stats:
        n_bad = s["unreadable"]
        note = f"  {console.dim(f'({n_bad} unreadable, excluded from brightness)')}" if n_bad else ""
        mean_str = console.bold(f"{s['mean']:>6.1f}")
        print(
            f"   {s['class']:<{max_cls}}  {s['count']:>7}  "
            f"{mean_str}  {s['std']:>6.1f}  "
            f"{s['min']:>5.1f}  {s['max']:>5.1f}{note}"
        )

    print()
    print(f" Dataset mean brightness : {console.bold(f'{dataset_mean:.1f}')}")

    if biased:
        names = ", ".join(f"'{s['class']}'" for s in biased)
        print(console.dim(
            f" Note: {names} differ{'s' if len(biased) == 1 else ''} noticeably in brightness from the "
            "dataset mean. That is often natural class signal, not a problem."
        ))

    print()
    dist = dict(sorted(((c.name, len(i)) for c, i in all_images.items()), key=lambda x: -x[1]))
    if not print_class_distribution(dist):
        console.success("No significant class imbalance detected.")
    return {s["class"]: s["mean"] for s in stats}


def _print_split_shift(names: list[str], per_split: list[dict[str, float] | None]) -> None:
    """Warn about classes whose mean brightness differs strongly between splits."""
    from cvbench.core import console

    shifted = []
    for cls in sorted({c for m in per_split if m for c in m}):
        vals = [(n, m[cls]) for n, m in zip(names, per_split, strict=True) if m and cls in m]
        if len(vals) > 1:
            lo, hi = min(vals, key=lambda v: v[1]), max(vals, key=lambda v: v[1])
            if hi[1] - lo[1] > _SPLIT_SHIFT_THRESHOLD:
                shifted.append((cls, lo, hi))
    if not shifted:
        return
    print(console.rule())
    for cls, lo, hi in shifted:
        console.warning(
            f"'{cls}' brightness differs between splits: {lo[0]} {lo[1]:.1f} vs {hi[0]} {hi[1]:.1f} "
            "(possible distribution shift)"
        )
    print()


@data.command(
    "explore",
    short_help="Report per-class brightness, class balance and data-integrity problems.",
    examples=[
        ("data explore data/ready", "Analyse every split"),
        ("data explore data/ready --split test", "Analyse a single split"),
    ],
    see_also=[("data prep <src> <dst>", "drop duplicates, leaks and unreadable images"),
              ("data upsample <src> <dst> --augmentation aug.yaml --target 500",
               "grow an under-represented class")],
)
@click.argument("data_dir")
@click.option("--split", default=None,
              help="Analyse only this split (train / val / test); default: every split present.")
def explore(data_dir, split):
    """Analyse per-class brightness and class distribution.

    DATA_DIR is the root dataset directory (containing train/, val/, test/
    subdirectories) or a split directory directly.

    Unreadable images (empty or undecodable), images PIL reads but TensorFlow
    rejects (e.g. a truncated BMP named .jpg; training would crash on them),
    images that appear in more than one split (data leakage), identical images
    within a split and identical images with conflicting labels are listed once
    for the whole dataset, and the command exits with status 1 if any are
    found; run ``data prep`` to drop or repair them.
    """
    from cvbench.core import console

    root = Path(data_dir)
    if not root.is_dir():
        raise click.ClickException(f"Dataset directory not found: '{root}'")
    n_images = len(layout_mod.list_images(root, recursive=True, exts=layout_mod.READABLE_IMAGE_EXTS))
    with console.progress(n_images, "Checking integrity") as advance:
        issues = prep_mod.find_integrity_issues(root, progress=advance)
    bad = {root / rel for rel, _ in issues.corrupt}

    if layout_mod.is_yolo_dataset(root):
        _print_integrity(root, issues)
        if issues:
            raise SystemExit(1)
        return

    names = [split] if split else [n for n in layout_mod.SPLIT_NAMES if (root / n).is_dir()]
    split_dirs = [root / n for n in names if (root / n).is_dir()] or [root]
    reported = [_explore_split(d, bad) for d in split_dirs]
    if not any(reported):
        raise click.ClickException("No images found.")
    _print_split_shift([d.name for d in split_dirs], reported)
    print(console.rule())
    _print_integrity(root, issues)
    if issues:
        raise SystemExit(1)


@data.command(
    "aug",
    short_help="Write an augmentation config file to edit and reuse.",
    examples=[
        ("data aug --preset light", "Write workspace/augmentation.yaml from the 'light' preset"),
        ("data aug", "Write the fully-commented reference sheet, then uncomment what you want"),
        ("data aug --preset heavy -o workspace/heavy.yaml", "Write to a specific path"),
    ],
    see_also=[
        ("train <dataset> --augmentation workspace/augmentation.yaml", "apply it live during training"),
        ("data upsample", "materialise augmented images for one class folder"),
    ],
)
@click.option("--preset", type=click.Choice(["light", "standard", "heavy", "reference"]),
              default="reference", show_default=True,
              help="Starting point. light/standard/heavy are ready-to-use configs; "
                   "reference lists every transform, commented out, for you to edit.")
@click.option("-o", "--output", "output", default="workspace/augmentation.yaml",
              show_default=True, type=click.Path(dir_okay=False),
              help="Where to write the config file.")
@click.option("--force", is_flag=True, default=False, help="Overwrite the output file if it exists.")
def aug(preset, output, force):
    """Write an augmentation config file, then edit it in any editor.

    The file is plain YAML: pass it to `train --augmentation FILE` (applied
    live during training) or `data upsample --augmentation FILE` (written to
    disk). Use --preset reference for a sheet of every available transform
    with all parameters and range/one_of syntax explained.
    """
    from cvbench.cli._aug_template import _preset_transforms, _reference_yaml, _render_config_yaml
    from cvbench.core import console

    path = Path(output)
    if path.exists() and not force:
        raise click.ClickException(f"'{path}' already exists — use --force to overwrite.")

    content = _reference_yaml() if preset == "reference" else _render_config_yaml(_preset_transforms(preset))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    console.success(f"Saved → {path}")
    print("  Open it in an editor to fine-tune any value.")
    print(f"  Usage:  train data/<dataset> --augmentation {path}")


@data.command(
    "upsample",
    short_help="Grow a class folder to TARGET images via augmentation.",
    examples=[
        ("data upsample data/train/dog data_aug/train/dog --augmentation aug.yaml --target 1500",
         "Copy originals, then add augmented variants until the folder holds 1500 images"),
    ],
    see_also=[("data aug", "make an augmentation config first")],
)
@click.argument("src_dir")
@click.argument("dst_dir")
@click.option("--augmentation", "aug_file", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Augmentation YAML file (see 'data aug').")
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

    from cvbench.augmentations.pipeline import build_aug_pipeline
    from cvbench.core import console
    from cvbench.core.config import load_aug_file

    src = Path(src_dir)
    dst = Path(dst_dir)

    if not src.is_dir():
        raise click.ClickException(f"Source directory not found: '{src}'")

    images = layout_mod.list_images(src, exts=layout_mod.READABLE_IMAGE_EXTS)
    if not images:
        raise click.ClickException(f"No images found in '{src}'")

    n_src = len(images)

    if n_src >= target:
        raise click.ClickException(
            f"'{src.name}' already has {n_src} images, at or above --target {target}; "
            "nothing to upsample."
        )

    if dst.exists():
        existing = layout_mod.list_images(dst, exts=layout_mod.READABLE_IMAGE_EXTS)
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

    print(console.rule())
    print(f" {console.bold('CVBench — data upsample')}")
    print(console.rule())
    print(f"  Source  : {console.dim(str(src))}  ({n_src} images)")
    print(f"  Target  : {target} images  (+{target - n_src} to generate)")
    print()

    # --- copy originals ---
    with console.progress(n_src, "Copying originals") as advance:
        for img_path in images:
            token = _fresh_token(used_tokens)
            used_tokens.add(token)
            dst_path = dst / f"{token}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dst_path)
            arr = np.array(Image.open(img_path).convert("RGB"))
            used_hashes.add(hashify_mod.hash_array(arr))
            advance(1)
    console.success(f"Copied {n_src} original(s)")
    print()

    # --- generate augmented samples ---
    n_to_generate = target - n_src

    generated = 0
    total_skipped = 0

    with console.progress(n_to_generate, f"Generating {n_to_generate} augmented sample(s)") as advance:
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
            advance(1)

    console.success(f"Generated {generated} augmented image(s)")
    if total_skipped:
        print(f"  {console.yellow('⚠')}  Skipped {total_skipped} duplicate(s) after {_MAX_RETRIES} retries each")
    print()
    print(f"  Output  : {console.bold(str(dst))}  ({console.green(str(len(list(dst.iterdir()))))} images total)")
    print(console.rule())


@data.command(
    "prep",
    short_help="Copy a dataset, dropping corrupt images, split leaks and duplicates.",
    examples=[
        ("data prep data/raw data/prepped",
         "Hash-rename images, drop corrupt files, split leaks and duplicates"),
        ("data prep data/raw data/prepped --no-hash --no-duplicates",
         "Keep original filenames but still drop within-split duplicates"),
        ("data prep data/raw data/prepped --on-incompatible repair",
         "Re-encode images TensorFlow can't decode as PNG instead of dropping them"),
        ("data prep data/raw data/prepped --dry-run", "Preview the plan without writing anything"),
    ],
)
@click.argument("src")
@click.argument("dst")
@click.option("--no-hash", is_flag=True, default=False,
              help="Keep original filenames instead of renaming to content hashes.")
@click.option("--no-duplicates", is_flag=True, default=False,
              help="Drop within-split duplicates (always on unless --no-hash is given).")
@click.option("--on-incompatible", type=click.Choice(["drop", "repair"]), default="drop",
              show_default=True,
              help="What to do with images PIL reads but TensorFlow rejects: drop them, "
                   "or re-encode just those as lossless PNG.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show the plan without writing DST.")
def prep(src, dst, no_hash, no_duplicates, on_incompatible, dry_run):
    """Copy SRC to DST, dropping unusable images and leaked duplicates.

    SRC  dataset directory to prep (classification or YOLO layout)\n
    DST  destination for the prepped copy; must be empty or non-existent.

    Always: corrupt images (empty or undecodable) are dropped and listed;
    images PIL reads but TensorFlow's decoder rejects (which would crash
    training) are listed and, per --on-incompatible, dropped (default) or
    re-encoded as lossless PNG (only those files; every other file is copied
    byte-for-byte). Run ``data explore`` to see which images are affected.
    An image present in more than one split is kept only in the
    highest-priority one (train > val > test), and for YOLO its label file
    goes with it. The same image under different classes (or, for YOLO, with
    differing label files) cannot be labelled reliably, so every copy is dropped.

    By default images are renamed to the full 32-hex-char MD5 digest of
    their pixel content and within-split duplicates are dropped (the
    lexicographically-first path is kept). With --no-hash the original
    filenames are kept and duplicates are only warned about, unless
    --no-duplicates is also given. SRC is never modified.
    """
    from cvbench.core import console

    src_dir = Path(src)
    dst_dir = Path(dst)

    if not src_dir.is_dir():
        raise click.ClickException(f"Source directory not found: '{src_dir}'")

    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise click.ClickException(
            f"Destination '{dst_dir}' already contains files. "
            "Provide an empty or non-existent directory."
        )

    n_images = len(layout_mod.list_images(src_dir, recursive=True, exts=layout_mod.READABLE_IMAGE_EXTS))
    with console.progress(n_images, "Scanning") as advance:
        plan = prep_mod.build_plan(
            src_dir, hash_names=not no_hash, remove_duplicates=no_duplicates, progress=advance,
            on_incompatible=on_incompatible,
        )
    if not dry_run:
        with console.progress(len(plan.actions), "Copying") as advance:
            prep_mod.apply_plan(plan, src_dir, dst_dir, progress=advance)

    print(console.rule())
    print(f" {console.bold('CVBench — data prep')}")
    print(console.rule())
    print(f"  Source  : {console.dim(str(src_dir))}")
    print(f"  Dest    : {console.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    print()

    if plan.corrupt:
        print(f" {console.bold(f'{len(plan.corrupt)} corrupt image(s) dropped:')}")
        for rel, reason in plan.corrupt:
            print(f"   {console.red('✗')} {console.red(str(rel))}  {console.dim(reason)}")
    else:
        console.success("No corrupt images found.")

    if plan.incompatible:
        repaired = plan.on_incompatible == "repair"
        verb = "re-encoded as PNG" if repaired else "dropped"
        print(f" {console.bold(f'{len(plan.incompatible)} TensorFlow-incompatible image(s) {verb}:')}")
        for rel, reason in plan.incompatible:
            mark, color = (console.yellow("↻"), console.yellow) if repaired else (console.red("✗"), console.red)
            print(f"   {mark} {color(str(rel))}  {console.dim(reason)}")
        if not repaired:
            console.warning("Pass --on-incompatible repair to keep them (re-encoded as PNG).")
    else:
        console.success("No TensorFlow-incompatible images found.")

    if plan.cross_split_leaks:
        n_leak_files = sum(len(v) - 1 for v in plan.cross_split_leaks.values())
        print(f" {console.bold(f'{n_leak_files} cross-split duplicate(s) removed:')}")
        for h, paths in plan.cross_split_leaks.items():
            _print_group(h, paths[:1], paths[1:])
    else:
        console.success("No cross-split duplicates found.")

    if plan.label_conflicts:
        n_conflict_files = sum(len(v) for v in plan.label_conflicts.values())
        print(f" {console.bold(f'{n_conflict_files} file(s) with conflicting labels dropped:')}")
        for h, paths in plan.label_conflicts.items():
            _print_group(h, [], paths)
    else:
        console.success("No conflicting labels found.")

    if plan.duplicate_groups:
        print(f" {console.bold(f'{len(plan.duplicate_groups)} within-split duplicate group(s):')}")
        for h, paths in plan.duplicate_groups.items():
            _print_group(h, paths if not plan.dedup else paths[:1], paths[1:] if plan.dedup else [])
        if not plan.dedup:
            console.warning("Duplicates were kept; pass --no-duplicates to remove them.")
    else:
        console.success("No within-split duplicates found.")

    print()
    splits = [s for s in (*layout_mod.SPLIT_NAMES, None)
              if plan.counts_before[s] or plan.counts_after[s]]
    print(console.dim(f"   {'Split':<8}  {'Before':>7}  {'After':>7}"))
    for s in splits:
        print(f"   {s or '(none)':<8}  {plan.counts_before[s]:>7}  {plan.counts_after[s]:>7}")

    print()
    verb = "Would write" if dry_run else "Wrote"
    console.success(f"{verb} {len(plan.actions)} image(s)")
    print(console.rule())


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

    SRC  flat dataset directory to split (classification: <class>/*;
    YOLO: images/* + labels/*). An already-split SRC is rejected — run
    'data flatten' first to pool it back together.\n
    DST  destination for the split dataset; must be empty or non-existent.

    Classification stratifies by class folder. YOLO images can carry boxes
    of more than one class, so the stratification key is each image's
    primary (most frequent, ties broken by lowest id) box class; images
    with no boxes are split the same proportional, seeded way as every
    other group. SRC is never modified.
    """
    from cvbench.core import console

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

    plan = split_mod.plan_split(src_dir, ratios, seed)
    if not dry_run:
        with console.progress(len(plan.actions), "Copying") as advance:
            split_mod.apply_plan(plan, dst_dir, progress=advance)

    print(console.rule())
    print(f" {console.bold('CVBench — data split')}")
    print(console.rule())
    print(f"  Source  : {console.dim(str(src_dir))}  ({console.dim('yolo' if plan.is_yolo else 'classification')})")
    print(f"  Dest    : {console.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    print(f"  Ratios  : train={train_ratio}  val={val_ratio}  test={test_ratio}  seed={seed}")
    print()

    classes = sorted({c for counts in plan.counts.values() for c in counts})
    max_cls = max((len(c) for c in classes), default=5)
    print(console.dim(f"   {'Class':<{max_cls}}  {'Train':>7}  {'Val':>7}  {'Test':>7}"))
    for cls in classes:
        row = [plan.counts.get(s, {}).get(cls, 0) for s in ("train", "val", "test")]
        print(f"   {cls:<{max_cls}}  {row[0]:>7}  {row[1]:>7}  {row[2]:>7}")

    print()
    verb = "Would write" if dry_run else "Wrote"
    total = len(plan.actions)
    console.success(f"{verb} {total} image(s) across {len(plan.splits_written)} split(s)")
    print(console.rule())


@data.command(
    "convert",
    short_help="Convert a Pascal VOC or COCO dataset into YOLO.",
    examples=[
        ("data convert data/VOC2007 data/voc", "Convert, keeping the source's own splits when it has them"),
        ("data convert data/coco data/pets --classes cat,dog", "Keep only some classes (their order sets the ids)"),
        ("data convert data/VOC2007 data/voc --resplit --train 0.7 --val 0.15 --test 0.15",
         "Ignore the source's splits and re-split with custom ratios"),
        ("data convert data/VOC2007 data/voc --dry-run", "Preview the classes and per-split counts"),
    ],
    see_also=[("train data/voc --epochs 20", "train on the result")],
)
@click.argument("src")
@click.argument("dst")
@click.option("--classes", default=None, metavar="A,B,C",
              help="Comma-separated classes to keep, in class-id order. Default: every class, sorted.")
@click.option("--train", "train_ratio", default=0.8, show_default=True, type=float,
              help="Fraction of images assigned to train when the data is split here.")
@click.option("--val", "val_ratio", default=0.1, show_default=True, type=float,
              help="Fraction of images assigned to val when the data is split here.")
@click.option("--test", "test_ratio", default=0.1, show_default=True, type=float,
              help="Fraction of images assigned to test when the data is split here.")
@click.option("--seed", default=42, show_default=True, type=int,
              help="Random seed for the split and for --limit.")
@click.option("--limit", default=None, type=click.IntRange(min=1),
              help="Keep a seeded random sample of this many images.")
@click.option("--resplit", is_flag=True, default=False,
              help="Ignore the source's own split lists and split with --train/--val/--test.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show the planned conversion without writing DST.")
def convert(src, dst, classes, train_ratio, val_ratio, test_ratio, seed, limit, resplit, dry_run):
    """Convert the Pascal VOC or COCO dataset in SRC to a YOLO dataset in DST.

    SRC  a Pascal VOC dataset (Annotations/ + JPEGImages/) or a COCO one
    (instances JSON + images)\n
    DST  destination for the YOLO dataset; must be empty or non-existent.

    Conversion is explicit so you choose the classes and splits after looking
    at the data. The source's own split lists (VOC ImageSets, COCO train/val/
    test files, a 'valid' folder counts as val) are used when present;
    otherwise the images get a seeded, class-stratified split. Dropped objects
    (VOC 'difficult', COCO crowds, unselected classes) and images are listed.
    SRC is never modified.
    """
    from cvbench.core import console

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

    wanted = [c.strip() for c in classes.split(",") if c.strip()] if classes else None
    plan = convert_mod.plan_convert(
        src_dir, classes=wanted, ratios=ratios, seed=seed, limit=limit, resplit=resplit,
    )
    if "val" not in plan.images:
        console.warning(
            "No val split in the result; detection training needs one. "
            "Re-run with --resplit, or add a val list to the source."
        )
    if not dry_run:
        with console.progress(len(plan.actions), "Copying") as advance:
            convert_mod.apply_plan(plan, dst_dir, progress=advance)

    print(console.rule())
    print(f" {console.bold('CVBench — data convert')}")
    print(console.rule())
    print(f"  Source  : {console.dim(str(src_dir))}  ({console.dim(convert_mod.FORMAT_LABELS[plan.source_format])})")
    print(f"  Dest    : {console.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    how = "the source's own split lists" if plan.used_source_splits else f"seeded split, seed={seed}"
    print(f"  Splits  : {how}")
    print(f"  Classes : {', '.join(f'{i}={c}' for i, c in enumerate(plan.class_names))}")
    print()

    max_cls = max((len(c) for c in plan.class_names), default=5)
    print(console.dim(f"   {'Boxes':<{max_cls}}  {'Train':>7}  {'Val':>7}  {'Test':>7}"))
    for cls in plan.class_names:
        row = [plan.counts.get(s, {}).get(cls, 0) for s in ("train", "val", "test")]
        print(f"   {cls:<{max_cls}}  {row[0]:>7}  {row[1]:>7}  {row[2]:>7}")
    row = [plan.images.get(s, 0) for s in ("train", "val", "test")]
    print(console.dim(f"   {'Images':<{max_cls}}  {row[0]:>7}  {row[1]:>7}  {row[2]:>7}"))

    if plan.dropped or plan.kept_notes:
        print()
        for reason, n in sorted(plan.dropped.items()):
            print(f"  Dropped : {n} {reason}")
        for reason, n in sorted(plan.kept_notes.items()):
            print(f"  Note    : {n} {reason}")

    print()
    verb = "Would write" if dry_run else "Wrote"
    console.success(f"{verb} {len(plan.actions)} image(s) across {len(plan.splits_written)} split(s)")
    print(console.rule())


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
    from cvbench.core import console

    src_dir = Path(src)
    dst_dir = Path(dst)

    if not src_dir.is_dir():
        raise click.ClickException(f"Source directory not found: '{src_dir}'")

    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise click.ClickException(
            f"Destination '{dst_dir}' already contains files. "
            "Provide an empty or non-existent directory."
        )

    plan = flatten_mod.plan_flatten(src_dir)
    if not dry_run:
        with console.progress(len(plan.actions), "Copying") as advance:
            flatten_mod.apply_plan(plan, dst_dir, progress=advance)

    print(console.rule())
    print(f" {console.bold('CVBench — data flatten')}")
    print(console.rule())
    print(f"  Source  : {console.dim(str(src_dir))}  ({console.dim('yolo' if plan.is_yolo else 'classification')})")
    print(f"  Dest    : {console.dim(str(dst_dir))}{'  (dry run)' if dry_run else ''}")
    print()

    classes = sorted(plan.counts)
    max_cls = max((len(c) for c in classes), default=5)
    print(console.dim(f"   {'Class':<{max_cls}}  {'Count':>7}"))
    for cls in classes:
        print(f"   {cls:<{max_cls}}  {plan.counts[cls]:>7}")

    print()
    verb = "Would write" if dry_run else "Wrote"
    console.success(f"{verb} {len(plan.actions)} image(s)")
    print(console.rule())


