"""Dataset layout — sniffing, YOLO txt parsing, and generic image listing.

Two layouts are recognised:

* classification — ``<root>/<split>/<class>/*.jpg``
* YOLO detection  — ``<root>/images/<split>/<stem>.jpg`` +
  ``<root>/labels/<split>/<stem>.txt`` (``class_id xc yc w h``, normalized) +
  ``<root>/data.yaml`` (class names)
"""
from __future__ import annotations

from pathlib import Path

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


def is_already_split(data_dir: Path) -> bool:
    """True if DATA_DIR has a train/val/test subdir (classification or YOLO)."""
    if is_yolo_dataset(data_dir):
        return any((data_dir / IMAGES_DIRNAME / s).is_dir() for s in SPLIT_NAMES)
    return any((data_dir / s).is_dir() for s in SPLIT_NAMES)


def dedupe_filename(name: str, used: set[str]) -> str:
    """NAME, or NAME with a numeric suffix if already present in USED."""
    if name not in used:
        used.add(name)
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 0
    while True:
        i += 1
        candidate = f"{stem}-{i}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def yolo_root(split_dir: Path) -> Path | None:
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
    for scanned, label_path in enumerate(sorted((root / LABELS_DIRNAME).rglob('*.txt')), start=1):
        for cls_id, _ in read_yolo_boxes(label_path):
            max_id = max(max_id, cls_id)
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
    """Write an Ultralytics-style ``data.yaml`` describing a YOLO dataset.

    An empty SPLITS means a flat (unsplit) dataset: a single ``images:``
    key pointing at ``images/`` directly, no split subdirectories.
    """
    lines = [f"path: {output_root}"]
    if splits:
        for split in splits:
            key = "val" if split == "val" else split
            lines.append(f"{key}: images/{split}")
    else:
        lines.append("images: images")
    lines.append("names:")
    lines.extend(f"  {i}: {cls}" for i, cls in enumerate(class_names))
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n")


def detect_task_name(data_dir: str | Path) -> str:
    """'detection' for a YOLO-layout data dir, else 'classification'."""
    return "detection" if is_yolo_dataset(Path(data_dir)) else "classification"


def _class_dirs(split_dir: Path) -> list[Path]:
    return sorted(p for p in split_dir.iterdir() if p.is_dir())


def _has_images(directory: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in directory.iterdir())


def _count_images(directory: Path) -> int:
    return sum(1 for f in directory.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def classification_layout_errors(root: Path) -> list[str]:
    """Human-readable problems that stop `root` from training as a classification dataset.

    Needs ``<root>/train/<class>/*.jpg`` with at least two non-empty classes; ``val/``
    is optional (``--val-split`` then carves one out of train) but must be well formed
    when present. An unsplit dataset gets a message pointing at ``data split``.
    """
    if not root.is_dir():
        return [f"'{root}' is not a directory."]
    train = root / "train"
    if not train.is_dir():
        return [
            f"No train/ folder found in '{root}'. Training needs train/ (and optionally "
            "val/ and test/) subfolders, each holding one folder per class. If your images "
            f"are not split yet, run: data split {root} <output_dir>"
        ]
    classes = _class_dirs(train)
    if not classes:
        return [f"'{train}' has no class subfolders (expected train/<class>/<image>)."]
    empty = [c.name for c in classes if not _has_images(c)]
    if empty:
        return [f"Class folder(s) in '{train}' contain no images: {', '.join(empty)}."]
    errors = []
    if len(classes) < 2:
        errors.append(f"'{train}' has only one class ({classes[0].name}); need at least two.")
    val = root / "val"
    if val.is_dir():
        if not _class_dirs(val):
            errors.append(f"'{val}' has no class subfolders (expected val/<class>/<image>). "
                          "Remove val/ to split it off train/ with --val-split.")
    else:
        small = [c.name for c in classes if _count_images(c) < 2]
        if small:
            errors.append(
                f"No val/ folder, so train/ is split in two, which needs at least 2 images per "
                f"class; too few in: {', '.join(small)}."
            )
    return errors


def detection_layout_errors(root: Path) -> list[str]:
    """Human-readable problems that stop `root` from training as a YOLO detection dataset."""
    errors = []
    if not (root / IMAGES_DIRNAME / "train").is_dir():
        errors.append(f"No images/train found under {root}")
    if not (root / IMAGES_DIRNAME / "val").is_dir():
        errors.append(
            f"No images/val found under {root} — detection requires an explicit "
            "val split (auto-splitting is not yet supported for object detection)."
        )
    return errors


def layout_errors(data_dir: str | Path, task_name: str | None = None) -> list[str]:
    """Layout problems for DATA_DIR under TASK_NAME (sniffed from the layout when omitted)."""
    root = Path(data_dir)
    task_name = task_name or detect_task_name(root)
    if task_name == "detection":
        return detection_layout_errors(root)
    return classification_layout_errors(root)
