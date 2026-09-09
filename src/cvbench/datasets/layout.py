"""Dataset layout — sniffing, YOLO txt parsing, and generic image listing.

Two layouts are recognised:

* classification — ``<root>/<split>/<class>/*.jpg``
* YOLO detection  — ``<root>/images/<split>/<stem>.jpg`` +
  ``<root>/labels/<split>/<stem>.txt`` (``class_id xc yc w h``, normalized) +
  ``<root>/data.yaml`` (class names)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SPLIT_NAMES = ('train', 'val', 'test')

IMAGES_DIRNAME = 'images'
LABELS_DIRNAME = 'labels'
# Cap on label files scanned when class names have to be inferred without data.yaml.
NAME_SCAN_LIMIT = 500


def list_images(directory: Path) -> list[Path]:
    """Sorted list of image files anywhere under DIRECTORY."""
    return sorted(
        f for f in directory.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


def is_yolo_dataset(data_dir: Path) -> bool:
    return (data_dir / IMAGES_DIRNAME).is_dir() and (data_dir / LABELS_DIRNAME).is_dir()


def yolo_root(split_dir: Path) -> Optional[Path]:
    """Return the dataset root if SPLIT_DIR is a YOLO split image directory."""
    parent = split_dir.parent
    if parent.name == IMAGES_DIRNAME and is_yolo_dataset(parent.parent):
        return parent.parent
    return None


def yolo_label_dir(split_dir: Path, root: Path) -> Path:
    return root / LABELS_DIRNAME / split_dir.name


def yolo_class_names(root: Path) -> list[str]:
    """Class names from data.yaml, falling back to ids found in the label files."""
    cfg_path = root / 'data.yaml'
    if cfg_path.is_file():
        try:
            raw = yaml.safe_load(cfg_path.read_text()) or {}
            names = raw.get('names')
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
            if isinstance(names, list):
                return [str(n) for n in names]
        except Exception:
            pass

    max_id = -1
    scanned = 0
    for label_path in sorted((root / LABELS_DIRNAME).rglob('*.txt')):
        for cls_id, _ in read_yolo_boxes(label_path):
            max_id = max(max_id, cls_id)
        scanned += 1
        if scanned >= NAME_SCAN_LIMIT:
            break
    return [str(i) for i in range(max_id + 1)]


def read_yolo_boxes(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Parse a YOLO txt file into ``(class_id, (x, y, w, h))`` with top-left origin."""
    if not label_path.is_file():
        return []
    boxes = []
    try:
        lines = label_path.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, w, h = (float(p) for p in parts[1:5])
        except ValueError:
            continue
        x = min(max(xc - w / 2, 0.0), 1.0)
        y = min(max(yc - h / 2, 0.0), 1.0)
        boxes.append((cls_id, (x, y, min(w, 1.0 - x), min(h, 1.0 - y))))
    return boxes


def write_data_yaml(output_root: Path, splits: list[str], class_names: list[str]) -> None:
    """Write an Ultralytics-style ``data.yaml`` describing a YOLO dataset."""
    lines = [f"path: {output_root}"]
    for split in splits:
        key = "val" if split == "val" else split
        lines.append(f"{key}: images/{split}")
    lines.append("names:")
    lines.extend(f"  {i}: {cls}" for i, cls in enumerate(class_names))
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n")


def detect_task_name(data_dir: str | Path) -> str:
    """'detection' for a YOLO-layout data dir, else 'classification'."""
    return "detection" if is_yolo_dataset(Path(data_dir)) else "classification"
