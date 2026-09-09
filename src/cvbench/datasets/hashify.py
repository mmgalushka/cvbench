"""``data hashify`` — copy a dataset, renaming every image to a content hash.

Names are derived from the pixel content (not the original basename, which
collides massively across sources — ``image1.jpg`` from one folder is not
``image1.jpg`` from another). Deterministic and idempotent: the same image
gets the same name every time, in any dataset.

Hashify never deletes anything. Two different source images landing on the
same destination name (in practice: byte-identical images sharing a
directory) both survive — the second gets a numeric suffix. Removing the
resulting redundant copy is ``dedup``'s job, not hashify's.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cvbench.datasets import layout

HASH_NAME_LEN = 16


def hash_array(arr: np.ndarray) -> str:
    """Full MD5 hex digest of an image's pixel bytes."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def hash_image_file(path: Path) -> str:
    """Full MD5 hex digest of the image at PATH (decoded, RGB)."""
    from PIL import Image
    return hash_array(np.array(Image.open(path).convert("RGB")))


def content_hash(path: Path) -> str:
    """The 16-hex-char filename token for the image at PATH."""
    return hash_image_file(path)[:HASH_NAME_LEN]


def _unique_name(base: str, suffix: str, used: set[str]) -> str:
    name = f"{base}{suffix}"
    i = 0
    while name in used:
        i += 1
        name = f"{base}-{i}{suffix}"
    used.add(name)
    return name


@dataclass
class HashifyAction:
    src_image: Path            # absolute
    dst_image: Path            # relative to dst
    src_label: Path | None = None   # absolute, YOLO only
    dst_label: Path | None = None   # relative to dst, YOLO only


@dataclass
class HashifyPlan:
    actions: list[HashifyAction] = field(default_factory=list)
    extra_files: list[Path] = field(default_factory=list)  # relative to src/dst, copied verbatim


def build_plan(src: Path) -> HashifyPlan:
    """Compute the rename plan for SRC. Read-only."""
    plan = HashifyPlan()
    used_by_dir: dict[Path, set[str]] = {}

    if layout.is_yolo_dataset(src):
        images_root = src / layout.IMAGES_DIRNAME
        labels_root = src / layout.LABELS_DIRNAME
        for img in layout.list_images(images_root):
            rel = img.relative_to(images_root)
            used = used_by_dir.setdefault(rel.parent, set())
            new_name = _unique_name(content_hash(img), img.suffix.lower(), used)
            dst_image = Path(layout.IMAGES_DIRNAME) / rel.parent / new_name

            label_src = labels_root / rel.with_suffix(".txt")
            src_label = dst_label = None
            if label_src.is_file():
                src_label = label_src
                dst_label = Path(layout.LABELS_DIRNAME) / rel.parent / f"{Path(new_name).stem}.txt"

            plan.actions.append(HashifyAction(img, dst_image, src_label, dst_label))

        if (src / "data.yaml").is_file():
            plan.extra_files.append(Path("data.yaml"))
    else:
        for img in layout.list_images(src):
            rel = img.relative_to(src)
            used = used_by_dir.setdefault(rel.parent, set())
            new_name = _unique_name(content_hash(img), img.suffix.lower(), used)
            plan.actions.append(HashifyAction(img, rel.parent / new_name))

    return plan


def apply_plan(plan: HashifyPlan, src: Path, dst: Path) -> None:
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


def hashify_dataset(src: Path, dst: Path, dry_run: bool) -> HashifyPlan:
    """Build the rename plan for SRC and, unless DRY_RUN, write it to DST."""
    plan = build_plan(src)
    if not dry_run:
        apply_plan(plan, src, dst)
    return plan
