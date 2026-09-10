"""``data dedup`` — copy a dataset, dropping exact-duplicate images.

Duplicates are grouped by full-content MD5 (via
:mod:`cvbench.datasets.hashify`). Within each group, the
lexicographically-first relative path is kept; the rest are dropped. For a
YOLO dataset, dropping an image also drops its paired label file; the kept
image's label is carried over unchanged.

Optionally flags "cross-split leakage" — a duplicate group whose members
span more than one split (train/val/test) — since that's data leakage
between splits, independent of whether ``--dry-run``/removal is requested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cvbench.datasets import hashify, layout


@dataclass
class DedupPlan:
    keep: list[Path] = field(default_factory=list)                     # relative to src
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


def build_plan(src: Path) -> DedupPlan:
    """Find duplicate/cross-split-leak groups under SRC. Read-only."""
    groups: dict[str, list[Path]] = {}
    for img in layout.list_images(src):
        rel = img.relative_to(src)
        groups.setdefault(hashify.hash_image_file(img), []).append(rel)

    plan = DedupPlan()
    for h, paths in groups.items():
        paths_sorted = sorted(paths)
        plan.keep.append(paths_sorted[0])
        if len(paths_sorted) > 1:
            plan.duplicate_groups[h] = paths_sorted
            splits = {_split_of(p) for p in paths_sorted}
            if len(splits - {None}) > 1:
                plan.cross_split_leaks[h] = paths_sorted

    plan.keep.sort()
    return plan


def _label_rel(image_rel: Path) -> Path | None:
    """The YOLO label path relative to a dataset root for an image relative path."""
    parts = image_rel.parts
    if parts and parts[0] == layout.IMAGES_DIRNAME:
        return Path(layout.LABELS_DIRNAME, *parts[1:]).with_suffix(".txt")
    return None


def apply_plan(plan: DedupPlan, src: Path, dst: Path) -> None:
    """Materialize PLAN at DST — copy only the kept images (+ their labels)."""
    import shutil

    is_yolo = layout.is_yolo_dataset(src)
    for rel in plan.keep:
        dst_img = dst / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dst_img)

        if is_yolo:
            label_rel = _label_rel(rel)
            if label_rel is not None and (src / label_rel).is_file():
                dst_label = dst / label_rel
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / label_rel, dst_label)

    if is_yolo and (src / "data.yaml").is_file():
        shutil.copy2(src / "data.yaml", dst / "data.yaml")


def dedup_dataset(src: Path, dst: Path, dry_run: bool) -> DedupPlan:
    """Build the dedup plan for SRC and, unless DRY_RUN, write it to DST."""
    plan = build_plan(src)
    if not dry_run:
        apply_plan(plan, src, dst)
    return plan
