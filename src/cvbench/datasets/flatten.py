"""``data flatten`` — pool an already-split dataset back into one flat pool.

The exact inverse of ``data split``: takes a dataset already partitioned
into ``train``/``val``/``test`` (classification or YOLO layout) and copies
every split's images (and YOLO labels) into one flat pool with no split
structure — classification: ``<class>/*``; YOLO: ``images/*`` +
``labels/*``. A SRC that is not already split is rejected; there's nothing
to flatten. Run ``data split`` afterward to re-partition the result.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from cvbench.datasets import layout


@dataclass
class FlattenAction:
    src_image: Path             # absolute
    dst_image: Path             # relative to dst
    src_label: Path | None = None    # absolute, YOLO only
    dst_label: Path | None = None    # relative to dst, YOLO only


@dataclass
class FlattenPlan:
    actions: list[FlattenAction] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)   # class -> count
    is_yolo: bool = False
    class_names: list[str] | None = None


def build_plan_classification(src: Path) -> FlattenPlan:
    plan = FlattenPlan(is_yolo=False)
    used_by_class: dict[str, set[str]] = defaultdict(set)

    for split_name in layout.SPLIT_NAMES:
        split_dir = src / split_name
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = cls_dir.name
            for img in layout.list_images(cls_dir):
                dst_name = layout.dedupe_filename(img.name, used_by_class[cls])
                dst_rel = Path(cls) / dst_name
                plan.actions.append(FlattenAction(img, dst_rel))
                plan.counts[cls] = plan.counts.get(cls, 0) + 1
    return plan


def build_plan_yolo(src: Path) -> FlattenPlan:
    class_names = layout.yolo_class_names(src)
    images_root = src / layout.IMAGES_DIRNAME
    labels_root = src / layout.LABELS_DIRNAME
    plan = FlattenPlan(is_yolo=True, class_names=class_names)
    used_by_dir: dict[Path, set[str]] = defaultdict(set)

    for img in layout.list_images(images_root):
        rel = img.relative_to(images_root)
        rel_no_split = Path(*rel.parts[1:]) if rel.parts[0] in layout.SPLIT_NAMES else rel

        dst_name = layout.dedupe_filename(rel_no_split.name, used_by_dir[rel_no_split.parent])
        dst_image = Path(layout.IMAGES_DIRNAME) / rel_no_split.parent / dst_name

        label_src = labels_root / rel.with_suffix(".txt")
        src_label = dst_label = None
        if label_src.is_file():
            src_label = label_src
            dst_label = Path(layout.LABELS_DIRNAME) / rel_no_split.parent / f"{Path(dst_name).stem}.txt"

        plan.actions.append(FlattenAction(img, dst_image, src_label, dst_label))

        boxes = layout.read_yolo_boxes(label_src) if label_src.is_file() else []
        if boxes:
            from collections import Counter
            counts = Counter(cid for cid, _ in boxes)
            primary_cid = max(counts, key=lambda cid: (counts[cid], -cid))
            cls_name = class_names[primary_cid] if primary_cid < len(class_names) else str(primary_cid)
        else:
            cls_name = "__unlabeled__"
        plan.counts[cls_name] = plan.counts.get(cls_name, 0) + 1

    return plan


def apply_plan(plan: FlattenPlan, dst: Path) -> None:
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
        layout.write_data_yaml(dst, [], plan.class_names)


def flatten_dataset(src: Path, dst: Path, dry_run: bool) -> FlattenPlan:
    """Build the flatten plan for SRC and, unless DRY_RUN, write it to DST."""
    if not layout.is_already_split(src):
        raise ValueError(f"'{src}' is already flat — nothing to flatten.")
    plan = build_plan_yolo(src) if layout.is_yolo_dataset(src) else build_plan_classification(src)
    if not dry_run:
        apply_plan(plan, dst)
    return plan
