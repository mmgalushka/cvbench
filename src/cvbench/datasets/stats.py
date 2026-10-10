"""Class-folder statistics: counting, imbalance reporting, auto class weights.

Pure dict/filesystem math — no TensorFlow.
"""
from __future__ import annotations

from pathlib import Path

from cvbench.core import _console
from cvbench.datasets import layout

IMBALANCE_THRESHOLD = 3.0  # max/min class count ratio at which imbalance is reported


def get_dataset_overview(data_dir: str | Path) -> dict[str, dict[str, int]]:
    """Per-split image/class counts for DATA_DIR (classification or YOLO layout).

    Returns ``{split: {"images": n, "classes": n}}`` for each split present.
    Splits with no matching subdirectory are omitted.
    """
    root = Path(data_dir)
    overview: dict[str, dict[str, int]] = {}

    if layout.is_yolo_dataset(root):
        n_classes = len(layout.yolo_class_names(root))
        for split_name in layout.SPLIT_NAMES:
            split_dir = root / layout.IMAGES_DIRNAME / split_name
            if split_dir.is_dir():
                overview[split_name] = {
                    "images": len(layout.list_images(split_dir)),
                    "classes": n_classes,
                }
        return overview

    for split_name in layout.SPLIT_NAMES:
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue
        n_classes = sum(1 for p in split_dir.iterdir() if p.is_dir())
        overview[split_name] = {
            "images": len(layout.list_images(split_dir)),
            "classes": n_classes,
        }
    return overview


def get_class_distribution(train_dir: str) -> dict[str, int]:
    """Count image files per class. Returns {class_name: count} sorted by count descending."""
    dist = {
        p.name: sum(1 for f in p.iterdir() if f.is_file())
        for p in Path(train_dir).iterdir()
        if p.is_dir()
    }
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def compute_auto_weights(
    class_dist: dict[str, int], class_names: list[str]
) -> dict[int, float]:
    """Inverse-frequency class weights keyed by class index for Keras model.fit()."""
    total = sum(class_dist.values())
    n = len(class_dist)
    return {
        class_names.index(cls): round(total / (n * count), 4)
        for cls, count in class_dist.items()
    }


def resolve_class_weights(
    class_weight_cfg,
    class_dist: dict[str, int],
    class_names: list[str],
) -> dict[int, float] | None:
    """Resolve class_weight config value to a {class_index: weight} dict for Keras, or None."""
    if class_weight_cfg is None:
        return None
    if class_weight_cfg == "auto":
        return compute_auto_weights(class_dist, class_names)
    if isinstance(class_weight_cfg, dict):
        return {class_names.index(cls): float(class_weight_cfg.get(cls, 1.0)) for cls in class_names}
    return None


def print_class_distribution(class_dist: dict[str, int]) -> bool:
    """Print per-class sample counts with a bar chart; return True if imbalance was flagged.

    Imbalance is flagged when the largest class is at least ``IMBALANCE_THRESHOLD``
    times the smallest.
    """
    counts = list(class_dist.values())
    max_count = max(counts)
    min_count = min(counts)
    total = sum(counts)
    ratio = max_count / min_count if min_count > 0 else float("inf")
    uniform = all(c == counts[0] for c in counts)

    bar_width = 20
    print(f" {_console.bold('Class distribution:')}")
    rows = []
    for cls, count in class_dist.items():
        pct = count / total * 100
        bar = "" if uniform else "█" * int(count / max_count * bar_width)
        rows.append((cls, count, bar, f"{pct:.1f}%"))
    _console.table(["Class", ("Images", "right"), "", ("%", "right")], rows)

    imbalanced = ratio >= IMBALANCE_THRESHOLD
    if imbalanced:
        print()
        ratio_str = "∞" if ratio == float("inf") else f"{ratio:.1f}"
        _console.warning(f"Imbalance ratio {ratio_str}:1 (threshold {IMBALANCE_THRESHOLD:.1f}:1)")
    return imbalanced


def print_imbalance_warning(class_dist: dict[str, int], class_weight_cfg) -> None:
    """Print an imbalance warning with class-weight tip when ratio >= IMBALANCE_THRESHOLD."""
    counts = list(class_dist.values())
    max_count = max(counts)
    min_count = min(counts) if min(counts) > 0 else 1
    ratio = max_count / min_count

    if ratio >= IMBALANCE_THRESHOLD:
        _console.warning(f"Imbalance ratio {ratio:.1f}:1 (threshold {IMBALANCE_THRESHOLD:.1f}:1)")
        if class_weight_cfg is None:
            print(f"   {_console.dim('Tip: rerun with --class-weight auto')}")
        elif class_weight_cfg == "auto":
            _console.success("class_weight=auto applied")
        else:
            _console.success("custom class weights applied")
