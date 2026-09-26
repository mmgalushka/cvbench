"""``data prep`` — copy a dataset, giving every image a canonical content-hash
filename and dropping exact duplicates, in a single pass.

Each image's filename is the full 32-hex-char MD5 digest of its decoded
pixel content (:func:`cvbench.datasets.hashify.hash_image_file`). At that
length, two *different* images landing on the same digest is
cryptographically negligible, so — unlike a truncated hash — a name
collision here reliably means "same content": within each duplicate group
only the lexicographically-first original path is kept, and it alone is
copied under its hash name. For YOLO, dropping an image also drops its
paired label file.

Optionally flags "cross-split leakage" — a duplicate group whose members
span more than one split (train/val/test) — since that's data leakage
between splits, independent of whether ``--dry-run``/removal is requested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cvbench.datasets import hashify, layout


@dataclass
class PrepAction:
    src_image: Path            # absolute
    dst_image: Path            # relative to dst, hash-named
    src_label: Path | None = None   # absolute, YOLO only
    dst_label: Path | None = None   # relative to dst, YOLO only


@dataclass
class PrepPlan:
    actions: list[PrepAction] = field(default_factory=list)
    extra_files: list[Path] = field(default_factory=list)  # relative to src/dst, copied verbatim
    duplicate_groups: dict[str, list[Path]] = field(default_factory=dict)  # hash -> relative paths, len > 1
    cross_split_leaks: dict[str, list[Path]] = field(default_factory=dict)  # subset of duplicate_groups


def _split_of(rel: Path) -> str | None:
    """Best-effort split name ('train'/'val'/'test') for a relative image path."""
    parts = rel.parts
    if parts and parts[0] == layout.IMAGES_DIRNAME and len(parts) > 1 and parts[1] in layout.SPLIT_NAMES:
        return parts[1]
    if parts and parts[0] in layout.SPLIT_NAMES:
        return parts[0]
    return None


def _label_rel(image_rel: Path) -> Path | None:
    """The YOLO label path relative to a dataset root for an image relative path."""
    parts = image_rel.parts
    if parts and parts[0] == layout.IMAGES_DIRNAME:
        return Path(layout.LABELS_DIRNAME, *parts[1:]).with_suffix(".txt")
    return None


def build_plan(src: Path) -> PrepPlan:
    """Compute the hash-rename + dedup plan for SRC. Read-only."""
    groups: dict[str, list[Path]] = {}
    for img in layout.list_images(src):
        rel = img.relative_to(src)
        groups.setdefault(hashify.hash_image_file(img), []).append(rel)

    plan = PrepPlan()
    is_yolo = layout.is_yolo_dataset(src)

    for h, paths in groups.items():
        paths_sorted = sorted(paths)
        kept_rel = paths_sorted[0]

        if len(paths_sorted) > 1:
            plan.duplicate_groups[h] = paths_sorted
            splits = {_split_of(p) for p in paths_sorted}
            if len(splits - {None}) > 1:
                plan.cross_split_leaks[h] = paths_sorted

        dst_rel = kept_rel.parent / f"{h}{kept_rel.suffix.lower()}"

        src_label = dst_label = None
        if is_yolo:
            label_rel = _label_rel(kept_rel)
            if label_rel is not None and (src / label_rel).is_file():
                src_label = src / label_rel
                dst_label = _label_rel(dst_rel)

        plan.actions.append(PrepAction(src / kept_rel, dst_rel, src_label, dst_label))

    plan.actions.sort(key=lambda a: a.dst_image)

    if is_yolo and (src / "data.yaml").is_file():
        plan.extra_files.append(Path("data.yaml"))

    return plan


def apply_plan(plan: PrepPlan, src: Path, dst: Path) -> None:
    """Materialize PLAN at DST."""
    import shutil

    for action in plan.actions:
        dst_image = dst / action.dst_image
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(action.src_image, dst_image)
        if action.src_label and action.dst_label:
            dst_label = dst / action.dst_label
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(action.src_label, dst_label)

    for rel in plan.extra_files:
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dst_path)


def prep_dataset(src: Path, dst: Path, dry_run: bool) -> PrepPlan:
    """Build the prep plan for SRC and, unless DRY_RUN, write it to DST."""
    plan = build_plan(src)
    if not dry_run:
        apply_plan(plan, src, dst)
    return plan
