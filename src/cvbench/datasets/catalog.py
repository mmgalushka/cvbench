"""One dataset scan, shared by `data list` and the WebUI Datasets page.

A dataset is summarized the same way wherever it is listed — same task, split image
counts and class list — so the terminal and the browser cannot disagree about what is
on disk. Pure filesystem work, no TensorFlow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cvbench.datasets import convert, layout
from cvbench.datasets.stats import get_dataset_overview
from cvbench.tasks import detect_task_name

FORMAT_CLASSIFICATION = "classification"
FORMAT_YOLO = "yolo"
FORMAT_UNKNOWN = "unknown"

STATUS_READY = "ready"
STATUS_NEEDS_CONVERSION = "needs conversion"
STATUS_UNKNOWN = "unknown layout"


def _has_class_folders(root: Path) -> bool:
    """Whether any ``<root>/<split>/`` holds class subfolders (the classification layout)."""
    return any(
        (root / s).is_dir() and any(p.is_dir() for p in (root / s).iterdir())
        for s in layout.SPLIT_NAMES
    )


def detect_format(root: Path) -> str:
    """``"yolo"``, ``"classification"``, ``"voc"``, ``"coco"`` or ``"unknown"`` for the folder ROOT."""
    if layout.is_yolo_dataset(root):
        return FORMAT_YOLO
    foreign = convert.detect_format(root)
    if foreign:
        return foreign
    return FORMAT_CLASSIFICATION if _has_class_folders(root) else FORMAT_UNKNOWN


def dataset_info(data_dir: str | Path) -> dict[str, Any]:
    """Summary of the dataset at DATA_DIR.

    Keys: ``name``, ``path``, ``format`` ("classification" | "yolo" | "voc" | "coco" | "unknown"),
    ``status`` ("ready" | "needs conversion" | "unknown layout"), ``task`` (None unless ready),
    ``classes`` (names), ``num_classes`` and ``splits`` — ``{split: {"images": n, "dir": path}}``
    for each split that exists on disk. Only ready datasets are scanned for classes and splits.
    """
    root = Path(data_dir)
    fmt = detect_format(root)
    info: dict[str, Any] = {
        "name": root.name,
        "path": str(root),
        "format": fmt,
        "status": STATUS_READY,
        "task": None,
        "classes": [],
        "num_classes": 0,
        "splits": {},
    }
    if fmt == FORMAT_UNKNOWN:
        info["status"] = STATUS_UNKNOWN
        return info
    if fmt in convert.FORMAT_LABELS:
        info["status"] = STATUS_NEEDS_CONVERSION
        return info

    yolo = fmt == FORMAT_YOLO
    overview = get_dataset_overview(root)

    if yolo:
        classes = layout.yolo_class_names(root)
        split_root = root / layout.IMAGES_DIRNAME
    else:
        first = next((root / s for s in layout.SPLIT_NAMES if (root / s).is_dir()), None)
        classes = sorted(p.name for p in first.iterdir() if p.is_dir()) if first else []
        split_root = root

    info.update(
        task=detect_task_name(root),
        classes=classes,
        num_classes=len(classes),
        splits={
            name: {"images": counts["images"], "dir": str(split_root / name)}
            for name, counts in overview.items()
        },
    )
    return info


def scan_datasets(data_dir: str | Path, names: list[str]) -> list[dict[str, Any]]:
    """``dataset_info`` for each of NAMES (directory names) under DATA_DIR."""
    return [dataset_info(Path(data_dir) / name) for name in names]
