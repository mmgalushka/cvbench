"""``data merge`` — combine several datasets into one.

SRC's immediate subdirectories are the datasets to combine (each already in
classification or YOLO layout, already split); DST is the single merged
output. Splits are matched by name across sources (``train`` + ``train``,
``val`` + ``val``, ...) — a split missing from one source is simply skipped
for that source. Class name<->index maps are reconciled into one union
(YOLO label files are rewritten with remapped class ids); images are
renamed to a content hash (:mod:`cvbench.datasets.hashify`) so files from
different sources never collide by name.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from cvbench.datasets import layout
from cvbench.datasets.hashify import content_hash, unique_name


def discover_sources(src: Path) -> list[Path]:
    return sorted(p for p in src.iterdir() if p.is_dir())


def _layout_of(d: Path) -> str:
    if layout.is_yolo_dataset(d):
        return "yolo"
    if any((d / s).is_dir() for s in layout.SPLIT_NAMES):
        return "classification"
    raise ValueError(f"'{d}' is not a recognized classification or YOLO dataset")


def validate_sources(sources: list[Path]) -> str:
    """Return 'yolo' or 'classification' if SOURCES are >= 2 datasets of one layout."""
    if len(sources) < 2:
        raise ValueError(
            f"src must contain at least 2 dataset subdirectories, found {len(sources)}"
        )
    kinds = {s: _layout_of(s) for s in sources}
    distinct = set(kinds.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{s.name}={k}" for s, k in kinds.items())
        raise ValueError(f"src mixes classification and YOLO datasets: {detail}")
    return distinct.pop()


@dataclass
class MergeAction:
    src_image: Path                    # absolute
    dst_image: Path                    # relative to dst
    dst_label: Path | None = None      # relative to dst, YOLO only
    label_lines: list[str] | None = None   # already remapped, YOLO only


@dataclass
class MergePlan:
    actions: list[MergeAction] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)   # split -> class -> count
    is_yolo: bool = False
    class_names: list[str] | None = None
    splits_written: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _splits_present(sources: list[Path], has_split: Callable[[Path, str], bool]) -> list[str]:
    return [s for s in layout.SPLIT_NAMES if any(has_split(src_dir, s) for src_dir in sources)]


def build_plan_classification(sources: list[Path]) -> MergePlan:
    splits_present = _splits_present(sources, lambda d, s: (d / s).is_dir())
    plan = MergePlan(is_yolo=False)
    used_by_dir: dict[tuple[str, str], set[str]] = defaultdict(set)

    for split_name in splits_present:
        wrote_any = False
        for src_dir in sources:
            split_dir = src_dir / split_name
            if not split_dir.is_dir():
                continue
            for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                cls = cls_dir.name
                for img in layout.list_images(cls_dir):
                    used = used_by_dir[(split_name, cls)]
                    new_name = unique_name(content_hash(img), img.suffix.lower(), used)
                    dst_image = Path(split_name) / cls / new_name
                    plan.actions.append(MergeAction(img, dst_image))
                    plan.counts.setdefault(split_name, {})
                    plan.counts[split_name][cls] = plan.counts[split_name].get(cls, 0) + 1
                    wrote_any = True
        if wrote_any:
            plan.splits_written.append(split_name)
    return plan


def _remap_label_lines(label_path: Path, remap: dict[int, int]) -> list[str]:
    """Rewrite a YOLO label file's class ids via REMAP, other columns untouched."""
    lines_out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            old_id = int(float(parts[0]))
        except ValueError:
            continue
        lines_out.append(" ".join([str(remap.get(old_id, old_id)), *parts[1:]]))
    return lines_out


def build_plan_yolo(sources: list[Path]) -> MergePlan:
    per_source_names = {src_dir: layout.yolo_class_names(src_dir) for src_dir in sources}

    union_names: list[str] = []
    name_to_idx: dict[str, int] = {}
    first_index: dict[str, int] = {}
    warnings: list[str] = []
    for src_dir in sources:
        for i, name in enumerate(per_source_names[src_dir]):
            if name not in name_to_idx:
                name_to_idx[name] = len(union_names)
                union_names.append(name)
                first_index[name] = i
            elif first_index[name] != i:
                warnings.append(
                    f"class '{name}' is index {i} in '{src_dir.name}' but "
                    f"{first_index[name]} in an earlier source — check for a labeling mismatch"
                )

    splits_present = _splits_present(sources, lambda d, s: (d / layout.IMAGES_DIRNAME / s).is_dir())
    plan = MergePlan(is_yolo=True, class_names=union_names, warnings=warnings)
    used_by_dir: dict[tuple[str, Path], set[str]] = defaultdict(set)

    for split_name in splits_present:
        wrote_any = False
        for src_dir in sources:
            images_split_dir = src_dir / layout.IMAGES_DIRNAME / split_name
            if not images_split_dir.is_dir():
                continue
            labels_split_dir = src_dir / layout.LABELS_DIRNAME / split_name
            remap = {i: name_to_idx[name] for i, name in enumerate(per_source_names[src_dir])}

            for img in layout.list_images(images_split_dir):
                rel = img.relative_to(images_split_dir)
                used = used_by_dir[(split_name, rel.parent)]
                new_name = unique_name(content_hash(img), img.suffix.lower(), used)
                dst_image = Path(layout.IMAGES_DIRNAME) / split_name / rel.parent / new_name

                dst_label = label_lines = None
                label_src = labels_split_dir / rel.with_suffix(".txt")
                if label_src.is_file():
                    label_lines = _remap_label_lines(label_src, remap)
                    dst_label = Path(layout.LABELS_DIRNAME) / split_name / rel.parent / f"{Path(new_name).stem}.txt"

                plan.actions.append(MergeAction(img, dst_image, dst_label, label_lines))

                boxes = layout.read_yolo_boxes(label_src) if label_src.is_file() else []
                plan.counts.setdefault(split_name, {})
                if boxes:
                    box_counts = Counter(cid for cid, _ in boxes)
                    primary_cid = max(box_counts, key=lambda cid: (box_counts[cid], -cid))
                    cls_name = union_names[remap[primary_cid]]
                else:
                    cls_name = "__unlabeled__"
                plan.counts[split_name][cls_name] = plan.counts[split_name].get(cls_name, 0) + 1
                wrote_any = True
        if wrote_any:
            plan.splits_written.append(split_name)
    return plan


def apply_plan(plan: MergePlan, dst: Path) -> None:
    """Materialize PLAN at DST."""
    import shutil

    for action in plan.actions:
        dst_image = dst / action.dst_image
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(action.src_image, dst_image)
        if action.dst_label is not None:
            dst_label = dst / action.dst_label
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            text = "\n".join(action.label_lines)
            dst_label.write_text(f"{text}\n" if text else "")

    if plan.is_yolo and plan.class_names is not None:
        layout.write_data_yaml(dst, plan.splits_written, plan.class_names)


def merge_datasets(src: Path, dst: Path, dry_run: bool) -> MergePlan:
    """Build the merge plan for SRC's dataset subdirectories and, unless DRY_RUN, write it to DST."""
    sources = discover_sources(src)
    kind = validate_sources(sources)
    plan = build_plan_yolo(sources) if kind == "yolo" else build_plan_classification(sources)
    if not dry_run:
        apply_plan(plan, dst)
    return plan
