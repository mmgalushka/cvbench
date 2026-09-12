"""``data split`` — partition a flat dataset into train/val/test, stratified.

Works on a flat pool only (classification: ``<class>/*``; YOLO:
``images/*`` + ``labels/*``) — an already-split SRC is rejected; run
``data flatten`` first to pool it back together, then re-split the result.
Classification stratifies by class folder. YOLO images can carry boxes of
more than one class, so the stratification key is each image's *primary*
class — its most frequent box class, ties broken by the lowest class id;
images with no boxes fall into their own group and are split the same
(proportional, seeded) way as every other group.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from cvbench.datasets import layout

T = TypeVar("T")

UNLABELED_KEY = "__unlabeled__"


def stratified_split(
    items: list[T], key_fn: Callable[[T], str], ratios: tuple[float, float, float], seed: int,
) -> dict[str, list[T]]:
    """Group ITEMS by KEY_FN, seeded-shuffle each group, and slice by RATIOS."""
    train_r, val_r, _test_r = ratios
    groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)

    result: dict[str, list[T]] = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        group = list(groups[key])
        random.Random(f"{seed}:{key}").shuffle(group)
        n = len(group)
        n_train = min(round(n * train_r), n)
        n_val = min(round(n * val_r), n - n_train)
        result["train"].extend(group[:n_train])
        result["val"].extend(group[n_train:n_train + n_val])
        result["test"].extend(group[n_train + n_val:])
    return result


@dataclass
class SplitAction:
    src_image: Path             # absolute
    dst_image: Path             # relative to dst
    src_label: Path | None = None    # absolute, YOLO only
    dst_label: Path | None = None    # relative to dst, YOLO only


@dataclass
class SplitPlan:
    actions: list[SplitAction] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)   # split -> class -> count
    is_yolo: bool = False
    class_names: list[str] | None = None
    splits_written: list[str] = field(default_factory=list)


def _classification_pool(src: Path) -> dict[str, list[Path]]:
    """class name -> image paths, read from a flat pool (SRC/<class>/*)."""
    pool: dict[str, list[Path]] = defaultdict(list)
    for cls_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        pool[cls_dir.name].extend(layout.list_images(cls_dir))
    return pool


def build_plan_classification(src: Path, ratios: tuple[float, float, float], seed: int) -> SplitPlan:
    pool = _classification_pool(src)
    keyed = [(img, cls) for cls, imgs in pool.items() for img in imgs]
    assigned = stratified_split(keyed, key_fn=lambda t: t[1], ratios=ratios, seed=seed)

    plan = SplitPlan(is_yolo=False)
    for split_name, entries in assigned.items():
        if not entries:
            continue
        used_by_class: dict[str, set[str]] = defaultdict(set)
        for img, cls in entries:
            dst_name = layout.dedupe_filename(img.name, used_by_class[cls])
            dst_rel = Path(split_name) / cls / dst_name
            plan.actions.append(SplitAction(img, dst_rel))
            plan.counts.setdefault(split_name, {})
            plan.counts[split_name][cls] = plan.counts[split_name].get(cls, 0) + 1
        plan.splits_written.append(split_name)
    return plan


def _yolo_pool(src: Path) -> list[tuple[Path, Path | None, Path]]:
    """(image, label-or-None, image-path-relative-to-images-root) from a flat pool."""
    images_root = src / layout.IMAGES_DIRNAME
    labels_root = src / layout.LABELS_DIRNAME
    pairs: list[tuple[Path, Path | None, Path]] = []
    for img in layout.list_images(images_root):
        rel = img.relative_to(images_root)
        label_path = labels_root / rel.with_suffix(".txt")
        pairs.append((img, label_path if label_path.is_file() else None, rel))
    return pairs


def _primary_class_key(label_path: Path | None) -> str:
    if label_path is None:
        return UNLABELED_KEY
    boxes = layout.read_yolo_boxes(label_path)
    if not boxes:
        return UNLABELED_KEY
    counts = Counter(cls_id for cls_id, _ in boxes)
    max_count = max(counts.values())
    best = min(cls_id for cls_id, c in counts.items() if c == max_count)  # tie -> lowest id
    return str(best)


def build_plan_yolo(src: Path, ratios: tuple[float, float, float], seed: int) -> SplitPlan:
    class_names = layout.yolo_class_names(src)
    pool = _yolo_pool(src)
    keyed = [(img, label, rel, _primary_class_key(label)) for img, label, rel in pool]
    assigned = stratified_split(keyed, key_fn=lambda t: t[3], ratios=ratios, seed=seed)

    plan = SplitPlan(is_yolo=True, class_names=class_names)
    for split_name, entries in assigned.items():
        if not entries:
            continue
        used_by_dir: dict[Path, set[str]] = defaultdict(set)
        for img, label, rel, key in entries:
            dst_name = layout.dedupe_filename(rel.name, used_by_dir[rel.parent])
            dst_image = Path(layout.IMAGES_DIRNAME) / split_name / rel.parent / dst_name
            src_label = dst_label = None
            if label is not None:
                src_label = label
                dst_label = Path(layout.LABELS_DIRNAME) / split_name / rel.parent / f"{Path(dst_name).stem}.txt"
            plan.actions.append(SplitAction(img, dst_image, src_label, dst_label))

            cls_label = class_names[int(key)] if key != UNLABELED_KEY and int(key) < len(class_names) else key
            plan.counts.setdefault(split_name, {})
            plan.counts[split_name][cls_label] = plan.counts[split_name].get(cls_label, 0) + 1
        plan.splits_written.append(split_name)
    return plan


def apply_plan(plan: SplitPlan, dst: Path) -> None:
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

    if plan.is_yolo and plan.class_names is not None:
        layout.write_data_yaml(dst, plan.splits_written, plan.class_names)


def split_dataset(src: Path, dst: Path, ratios: tuple[float, float, float], seed: int, dry_run: bool) -> SplitPlan:
    """Build the split plan for SRC and, unless DRY_RUN, write it to DST."""
    if layout.is_already_split(src):
        raise ValueError(
            f"'{src}' is already split into train/val/test — "
            "run 'data flatten' first, then 'data split' the flattened result."
        )
    plan = build_plan_yolo(src, ratios, seed) if layout.is_yolo_dataset(src) else \
        build_plan_classification(src, ratios, seed)
    if not dry_run:
        apply_plan(plan, dst)
    return plan
