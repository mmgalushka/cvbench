"""``data prep`` — copy a dataset, dropping unusable images and split leaks.

Every image is decoded and hashed (:func:`cvbench.datasets.hashify.hash_image_file`,
the full 32-hex-char MD5 of its pixel content, so a digest match reliably
means "same image"). In a single pass ``data prep`` then:

* drops **corrupt** images (empty or undecodable) and reports why;
* flags **incompatible** images — PIL reads them but TensorFlow's decoder,
  the one training uses, rejects them (e.g. a truncated or 16-bit BMP named
  ``.jpg``). They are dropped by default or, with ``on_incompatible="repair"``,
  re-encoded as lossless PNG (only those files, nothing else is touched);
* removes **cross-split duplicates** — the same image in more than one of
  train/val/test is data leakage — keeping the copy in the highest-priority
  split (train > val > test). For YOLO the paired label file goes with it;
* drops **label conflicts** entirely — the same image filed under different
  classes (classification) or with differing label files (YOLO). Which label
  is right cannot be told, so every copy goes from all splits;
* by default renames images to their hash and drops within-split
  duplicates (lexicographically-first path wins). ``hash_names=False``
  keeps original filenames and only drops within-split duplicates when
  ``remove_duplicates`` is set.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from cvbench.datasets import hashify, layout

SPLIT_PRIORITY = layout.SPLIT_NAMES  # earlier wins when an image leaks across splits


@dataclass
class PrepAction:
    src_image: Path            # absolute
    dst_image: Path            # relative to dst
    repair: bool = False       # re-encode as PNG instead of copying
    src_label: Path | None = None   # absolute, YOLO only
    dst_label: Path | None = None   # relative to dst, YOLO only


@dataclass
class PrepPlan:
    actions: list[PrepAction] = field(default_factory=list)
    extra_files: list[Path] = field(default_factory=list)  # relative to src/dst, copied verbatim
    corrupt: list[tuple[Path, str]] = field(default_factory=list)  # (relative path, reason)
    # (relative path, reason) PIL reads but TensorFlow rejects; dropped or repaired per ``on_incompatible``
    incompatible: list[tuple[Path, str]] = field(default_factory=list)
    on_incompatible: str = "drop"
    # hash -> relative paths; first is kept, the rest are dropped
    cross_split_leaks: dict[str, list[Path]] = field(default_factory=dict)
    # hash -> relative paths within one split; dropped only when ``dedup`` is True
    duplicate_groups: dict[str, list[Path]] = field(default_factory=dict)
    # hash -> relative paths; every copy is dropped because the labels disagree
    label_conflicts: dict[str, list[Path]] = field(default_factory=dict)
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


@contextmanager
def _silence_native_stderr() -> Iterator[None]:
    """Mute fd 2 so libjpeg/TensorFlow warnings don't tear through a progress bar."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def tf_decode_error(path: Path) -> str | None:
    """Why TensorFlow's decoder rejects PATH, or None if it decodes.

    Mirrors the training loader (``tf.image.decode_image`` to 3 channels).
    TensorFlow is imported lazily so the datasets package stays TF-free.
    """
    import tensorflow as tf

    try:
        tf.image.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError as e:
        msg = re.sub(r"\{\{.*?\}\}|\[Op:.*", "", e.message or "").strip()
        return msg or "invalid image"
    return None


def _find_corrupt(
    src: Path, images: list[Path], progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, list[Path]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    """Hash every image; return (hash -> relative paths, corrupt, incompatible).

    Corrupt and incompatible are lists of (relative path, reason). Incompatible
    images still appear in the hash groups (PIL reads them), so duplicate and
    leak detection cover them.

    PROGRESS, if given, is called with 1 after each image is handled.
    """
    groups: dict[str, list[Path]] = {}
    corrupt: list[tuple[Path, str]] = []
    incompatible: list[tuple[Path, str]] = []
    with _silence_native_stderr():
        for img in images:
            rel = img.relative_to(src)
            if img.stat().st_size == 0:
                corrupt.append((rel, "empty file"))
            else:
                try:
                    groups.setdefault(hashify.hash_image_file(img), []).append(rel)
                except Exception as e:  # PIL raises many types for bad data
                    corrupt.append((rel, f"cannot decode ({type(e).__name__})"))
                else:
                    if reason := tf_decode_error(img):
                        incompatible.append((rel, reason))
            if progress:
                progress(1)
    return groups, corrupt, incompatible


def _find_label_conflicts(
    src: Path, groups: dict[str, list[Path]], is_yolo: bool
) -> dict[str, list[Path]]:
    """Groups of identical images whose labels disagree (hash -> sorted paths)."""
    conflicts = {}
    for h, paths in groups.items():
        if is_yolo:
            disagree = len({_read_label(src / _label_rel(p)) for p in paths}) > 1
        else:
            disagree = len({_class_of(p) for p in paths} - {None}) > 1
        if disagree:
            conflicts[h] = sorted(paths)
    return conflicts


def _split_leak(paths: list[Path]) -> list[Path] | None:
    """[kept, *dropped] when PATHS span several splits, else None.

    The first path of the highest-priority split is kept; copies in lower
    splits are dropped.
    """
    by_split: dict[str | None, list[Path]] = {}
    for p in sorted(paths):
        by_split.setdefault(_split_of(p), []).append(p)
    known = [s for s in SPLIT_PRIORITY if s in by_split]
    if len(known) < 2:
        return None
    return [by_split[known[0]][0], *(p for s in known[1:] for p in by_split[s])]


@dataclass
class IntegrityIssues:
    corrupt: list[tuple[Path, str]]         # (relative path, reason)
    incompatible: list[tuple[Path, str]]    # readable by PIL, rejected by TensorFlow
    leaks: dict[str, list[Path]]            # hash -> [kept, *dropped] across splits, labels agree
    label_conflicts: dict[str, list[Path]]  # hash -> paths whose labels disagree

    def __bool__(self) -> bool:
        return bool(self.corrupt or self.incompatible or self.leaks or self.label_conflicts)


def find_integrity_issues(src: Path, progress: Callable[[int], None] | None = None) -> IntegrityIssues:
    """Read-only scan of SRC for what ``data prep`` would drop.

    PROGRESS, if given, is called with 1 per image scanned.
    """
    groups, corrupt, incompatible = _find_corrupt(src, layout.list_images(src), progress)
    conflicts = _find_label_conflicts(src, groups, layout.is_yolo_dataset(src))
    leaks = {
        h: leak
        for h, paths in groups.items()
        if h not in conflicts and (leak := _split_leak(paths))
    }
    return IntegrityIssues(sorted(corrupt), sorted(incompatible), leaks, conflicts)


def build_plan(
    src: Path, hash_names: bool = True, remove_duplicates: bool = False,
    progress: Callable[[int], None] | None = None, on_incompatible: str = "drop",
) -> PrepPlan:
    """Compute the prep plan for SRC. Read-only.

    ON_INCOMPATIBLE is "drop" or "repair" and decides what happens to images
    TensorFlow can't decode (listed in ``plan.incompatible`` either way).

    PROGRESS, if given, is called with 1 per image scanned.

    Images whose labels conflict are excluded from the plan and listed in
    ``plan.label_conflicts``.
    """
    is_yolo = layout.is_yolo_dataset(src)
    images = layout.list_images(src)
    groups, corrupt, incompatible = _find_corrupt(src, images, progress)

    plan = PrepPlan(
        corrupt=sorted(corrupt), incompatible=sorted(incompatible),
        on_incompatible=on_incompatible, dedup=hash_names or remove_duplicates,
    )
    incompatible_set = {rel for rel, _ in incompatible}
    for img in images:
        plan.counts_before[_split_of(img.relative_to(src))] += 1

    plan.label_conflicts = _find_label_conflicts(src, groups, is_yolo)

    for h, paths in groups.items():
        if h in plan.label_conflicts:
            continue
        paths_sorted = sorted(paths)

        leak = _split_leak(paths_sorted)
        if leak:
            keep_split = _split_of(leak[0])
            kept_members = [p for p in paths_sorted if _split_of(p) == keep_split]
            plan.cross_split_leaks[h] = leak
        else:
            kept_members = paths_sorted

        if len(kept_members) > 1:
            plan.duplicate_groups[h] = kept_members
        survivors = kept_members[:1] if plan.dedup else kept_members

        for rel in survivors:
            repair = rel in incompatible_set
            if repair and on_incompatible != "repair":
                continue
            suffix = ".png" if repair else rel.suffix.lower()
            dst_rel = rel.parent / f"{h}{suffix}" if hash_names else (
                rel.with_suffix(suffix) if repair else rel)
            src_label = dst_label = None
            if is_yolo and (src / _label_rel(rel)).is_file():
                src_label = src / _label_rel(rel)
                dst_label = _label_rel(dst_rel)
            plan.actions.append(PrepAction(src / rel, dst_rel, repair, src_label, dst_label))
            plan.counts_after[_split_of(dst_rel)] += 1

    plan.actions.sort(key=lambda a: a.dst_image)

    if is_yolo and (src / "data.yaml").is_file():
        plan.extra_files.append(Path("data.yaml"))

    return plan


def apply_plan(
    plan: PrepPlan, src: Path, dst: Path, progress: Callable[[int], None] | None = None,
) -> None:
    """Materialize PLAN at DST. PROGRESS, if given, is called with 1 per image copied."""
    import shutil

    for action in plan.actions:
        dst_image = dst / action.dst_image
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        if action.repair:
            from PIL import Image
            with Image.open(action.src_image) as im:
                im.convert("RGB").save(dst_image, format="PNG")
        else:
            shutil.copy2(action.src_image, dst_image)
        if action.src_label and action.dst_label:
            dst_label = dst / action.dst_label
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(action.src_label, dst_label)
        if progress:
            progress(1)

    for rel in plan.extra_files:
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dst_path)


def prep_dataset(
    src: Path, dst: Path, dry_run: bool, hash_names: bool = True, remove_duplicates: bool = False,
    on_incompatible: str = "drop",
) -> PrepPlan:
    """Build the prep plan for SRC and, unless DRY_RUN, write it to DST."""
    plan = build_plan(src, hash_names, remove_duplicates, on_incompatible=on_incompatible)
    if not dry_run:
        apply_plan(plan, src, dst)
    return plan
