"""One dataset scan, shared by `data list` and the WebUI Datasets page.

A dataset is summarized the same way wherever it is listed — same task, split image
counts and class list — so the terminal and the browser cannot disagree about what is
on disk. Pure filesystem work, no TensorFlow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from cvbench.datasets import layout
from cvbench.datasets.stats import get_dataset_overview


def dataset_info(data_dir: str | Path) -> dict[str, Any]:
    """Summary of the dataset at DATA_DIR (classification or YOLO layout).

    Keys: ``name``, ``path``, ``format`` ("classification" | "yolo"), ``task``,
    ``classes`` (names), ``num_classes`` and ``splits`` — ``{split: {"images": n, "dir": path}}``
    for each split that exists on disk.
    """
    root = Path(data_dir)
    yolo = layout.is_yolo_dataset(root)
    overview = get_dataset_overview(root)

    if yolo:
        classes = layout.yolo_class_names(root)
        split_root = root / layout.IMAGES_DIRNAME
    else:
        first = next((root / s for s in layout.SPLIT_NAMES if (root / s).is_dir()), None)
        classes = sorted(p.name for p in first.iterdir() if p.is_dir()) if first else []
        split_root = root

    return {
        "name": root.name,
        "path": str(root),
        "format": "yolo" if yolo else "classification",
        "task": layout.detect_task_name(root),
        "classes": classes,
        "num_classes": len(classes),
        "splits": {
            name: {"images": counts["images"], "dir": str(split_root / name)}
            for name, counts in overview.items()
        },
    }


def scan_datasets(data_dir: str | Path, names: list[str]) -> list[dict[str, Any]]:
    """``dataset_info`` for each of NAMES (directory names) under DATA_DIR."""
    return [dataset_info(Path(data_dir) / name) for name in names]
