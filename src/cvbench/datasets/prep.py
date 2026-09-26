"""``data prep`` — copy a dataset, dropping unusable images and split leaks.

Every image is decoded and hashed (:func:`cvbench.datasets.hashify.hash_image_file`,
the full 32-hex-char MD5 of its pixel content, so a digest match reliably
means "same image"). In a single pass ``data prep`` then:

* drops **corrupt** images (empty or undecodable) and reports why;
* removes **cross-split duplicates** — the same image in more than one of
  train/val/test is data leakage — keeping the copy in the highest-priority
  split (train > val > test). For YOLO the paired label file goes with it;
* fails on **label conflicts** — the same image filed under different
  classes (classification only);
* by default renames images to their hash and drops within-split
  duplicates (lexicographically-first path wins). ``hash_names=False``
  keeps original filenames and only drops within-split duplicates when
  ``remove_duplicates`` is set.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from cvbench.datasets import hashify, layout

SPLIT_PRIORITY = layout.SPLIT_NAMES  # earlier wins when an image leaks across splits


class LabelConflictError(ValueError):
    """The same image content appears under different classes."""


@dataclass
class PrepAction:
    src_image: Path            # absolute
    dst_image: Path            # relative to dst
    src_label: Path | None = None   # absolute, YOLO only
    dst_label: Path | None = None   # relative to dst, YOLO only


@dataclass
class PrepPlan:
    actions: list[PrepAction] = field(default_factory=list)
    extra_files: list[Path] = field(default_factory=list)  # relative to src/dst, copied verbatim
    corrupt: list[tuple[Path, str]] = field(default_factory=list)  # (relative path, reason)
    # hash -> relative paths; first is kept, the rest are dropped
    cross_split_leaks: dict[str, list[Path]] = field(default_factory=dict)
    # hash -> relative paths within one split; dropped only when ``dedup`` is True
    duplicate_groups: dict[str, list[Path]] = field(default_factory=dict)
    label_mismatches: list[tuple[Path, Path]] = field(default_factory=list)  # (dropped, kept) YOLO images
    dedup: bool = True
    counts_before: Counter = field(default_factory=Counter)  # split -> images
    counts_after: Counter = field(default_factory=Counter)


def _split_of(rel: Path) -> str | None:
    """Best-effort split name ('train'/'val'/'test') for a relative image path."""
    parts = rel.parts
    if parts and parts[0] == layout.IMAGES_DIRNAME and len(parts) > 1 and parts[1] in layout.SPLIT_NAMES:
        return parts[1]
    if parts and parts[0] in layout.SPLIT_NAMES:
        return parts[0]
    return None


def _label_rel(image_rel: Path) -> Path:
    """The YOLO label path for an image relative path (which starts with 'images/')."""
    return Path(layout.LABELS_DIRNAME, *image_rel.parts[1:]).with_suffix(".txt")


def _class_of(rel: Path) -> str | None:
    """Class folder of a classification image path, or None if it has none."""
    parts = rel.parts[:-1]
    if parts and parts[0] in layout.SPLIT_NAMES:
        parts = parts[1:]
    return parts[0] if parts else None


def _read_label(path: Path) -> str:
    return path.read_text().strip() if path.is_file() else ""


def _find_corrupt(src: Path, images: list[Path]) -> tuple[dict[str, list[Path]], list[tuple[Path, str]]]:
    """Hash every image; return (hash -> relative paths, corrupt (path, reason))."""
    groups: dict[str, list[Path]] = {}
    corrupt: list[tuple[Path, str]] = []
    for img in images:
        rel = img.relative_to(src)
        if img.stat().st_size == 0:
            corrupt.append((rel, "empty file"))
            continue
        try:
            groups.setdefault(hashify.hash_image_file(img), []).append(rel)
        except Exception as e:  # PIL raises many types for bad data
            corrupt.append((rel, f"cannot decode ({type(e).__name__})"))
    return groups, corrupt


def _check_label_conflicts(groups: dict[str, list[Path]]) -> None:
    conflicts = []
    for paths in groups.values():
        classes = {_class_of(p) for p in paths} - {None}
        if len(classes) > 1:
            conflicts.append(sorted(paths))
    if conflicts:
        lines = [f"  {', '.join(str(p) for p in paths)}" for paths in sorted(conflicts)]
        raise LabelConflictError(
            f"{len(conflicts)} image(s) appear under different classes:\n" + "\n".join(lines)
        )


def build_plan(src: Path, hash_names: bool = True, remove_duplicates: bool = False) -> PrepPlan:
    """Compute the prep plan for SRC. Read-only.

    Raises :class:`LabelConflictError` if one image sits under several classes.
    """
    is_yolo = layout.is_yolo_dataset(src)
    images = layout.list_images(src)
    groups, corrupt = _find_corrupt(src, images)

    plan = PrepPlan(corrupt=sorted(corrupt), dedup=hash_names or remove_duplicates)
    for img in images:
        plan.counts_before[_split_of(img.relative_to(src))] += 1

    if not is_yolo:
        _check_label_conflicts(groups)

    for h, paths in groups.items():
        paths_sorted = sorted(paths)

        by_split: dict[str | None, list[Path]] = {}
        for p in paths_sorted:
            by_split.setdefault(_split_of(p), []).append(p)
        known = [s for s in SPLIT_PRIORITY if s in by_split]
        if len(known) > 1:
            keep_split = known[0]
            kept_members = by_split[keep_split]
            dropped = [p for s in known[1:] for p in by_split[s]]
            plan.cross_split_leaks[h] = [kept_members[0], *dropped]
            if is_yolo:
                kept_label = _read_label(src / _label_rel(kept_members[0]))
                for p in dropped:
                    if _read_label(src / _label_rel(p)) != kept_label:
                        plan.label_mismatches.append((p, kept_members[0]))
        else:
            kept_members = paths_sorted

        if len(kept_members) > 1:
            plan.duplicate_groups[h] = kept_members
        survivors = kept_members[:1] if plan.dedup else kept_members

        for rel in survivors:
            dst_rel = rel.parent / f"{h}{rel.suffix.lower()}" if hash_names else rel
            src_label = dst_label = None
            if is_yolo and (src / _label_rel(rel)).is_file():
                src_label = src / _label_rel(rel)
                dst_label = _label_rel(dst_rel)
            plan.actions.append(PrepAction(src / rel, dst_rel, src_label, dst_label))
            plan.counts_after[_split_of(dst_rel)] += 1

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


def prep_dataset(
    src: Path, dst: Path, dry_run: bool, hash_names: bool = True, remove_duplicates: bool = False
) -> PrepPlan:
    """Build the prep plan for SRC and, unless DRY_RUN, write it to DST."""
    plan = build_plan(src, hash_names, remove_duplicates)
    if not dry_run:
        apply_plan(plan, src, dst)
    return plan
