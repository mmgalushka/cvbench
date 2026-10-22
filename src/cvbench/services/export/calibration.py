"""Calibration image collection and calibration-set building for quantizing targets."""
from __future__ import annotations

import os
import random
from pathlib import Path

from cvbench.core import console
from cvbench.core.errors import DatasetError
from cvbench.datasets.layout import list_images
from cvbench.tasks import resolve_task


def collect_images_by_class(source_dir: Path) -> dict[str, list[Path]]:
    """Group images by class subdirectory; a flat dataset is a single ``_all`` bucket."""
    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    if class_dirs:
        per_class = {d.name: list_images(d, recursive=True) for d in class_dirs}
        return {k: v for k, v in per_class.items() if v}
    return {"_all": list_images(source_dir, recursive=True)}


def calibration_groups(cfg) -> dict[str, list[Path]]:
    """The images calibration samples from, grouped (by class folder unless the task says otherwise).

    A task supplies its own candidates through ``Task.calibration_groups``; the default is the
    training images (validation images when there is no train dir) grouped by class folder.
    """
    groups = resolve_task(cfg).calibration_groups(cfg)
    if groups is not None:
        return {name: files for name, files in groups.items() if files}
    source_dir = Path(
        cfg.data.train_dir if os.path.isdir(cfg.data.train_dir) else cfg.data.val_dir
    )
    return collect_images_by_class(source_dir)


def _thumbnail_features(files: list[Path]):
    """Compact feature per image: 8x8 grayscale thumbnail, flattened and normalised."""
    import numpy as np
    from PIL import Image

    feats = []
    for p in files:
        img = Image.open(str(p)).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        feats.append(np.array(img, dtype=np.float32).flatten() / 255.0)
    return np.stack(feats)


def _select_stratified(per_class, total, rng) -> list[Path]:
    from sklearn.cluster import MiniBatchKMeans

    selected: list[Path] = []
    per_class_quota = max(1, round(total / len(per_class)))
    for files in per_class.values():
        n_clusters = min(per_class_quota, len(files))
        if n_clusters < len(files):
            labels = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=42, n_init=3
            ).fit_predict(_thumbnail_features(files))
            clusters: dict[int, list[Path]] = {}
            for path, label in zip(files, labels, strict=True):
                clusters.setdefault(int(label), []).append(path)
            for bucket in clusters.values():
                selected.append(rng.choice(bucket))
        else:
            selected.extend(files)
    return selected


def _select_equal(per_class, total, rng) -> list[Path]:
    selected: list[Path] = []
    per_class_quota = max(1, round(total / len(per_class)))
    for files in per_class.values():
        selected.extend(rng.sample(files, min(per_class_quota, len(files))))
    return selected


def _select_diverse(per_class, total, rng) -> tuple[list[Path], int]:
    from sklearn.cluster import MiniBatchKMeans

    flat_files = [p for files in per_class.values() for p in files]
    n_clusters = min(32, len(flat_files))
    labels = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, n_init=3
    ).fit_predict(_thumbnail_features(flat_files))

    buckets: list[list[Path]] = [[] for _ in range(n_clusters)]
    for path, label in zip(flat_files, labels, strict=True):
        buckets[label].append(path)

    non_empty = [c for c in buckets if c]
    per_cluster_quota = max(1, round(total / len(non_empty)))
    selected: list[Path] = []
    for cluster in non_empty:
        selected.extend(rng.sample(cluster, min(per_cluster_quota, len(cluster))))
    return selected, len(non_empty)


def _select_proportional(per_class, total, rng) -> list[Path]:
    selected: list[Path] = []
    total_available = sum(len(v) for v in per_class.values())
    for files in per_class.values():
        quota = max(1, round(len(files) / total_available * total))
        selected.extend(rng.sample(files, min(quota, len(files))))
    return selected


def build_calibration_set(
    cfg, output_path: Path, total: int = 1024, strategy: str = "stratified"
) -> tuple[int, int]:
    import numpy as np
    from PIL import Image

    size = cfg.model.input_size
    rng = random.Random(42)

    per_class = calibration_groups(cfg)
    if not per_class:
        raise DatasetError(
            f"No images found in: {cfg.data.train_dir if os.path.isdir(cfg.data.train_dir) else cfg.data.val_dir}"
        )

    n_classes = len(per_class)
    if strategy == "stratified":
        selected = _select_stratified(per_class, total, rng)
    elif strategy == "equal":
        selected = _select_equal(per_class, total, rng)
    elif strategy == "diverse":
        selected, n_classes = _select_diverse(per_class, total, rng)
    else:
        selected = _select_proportional(per_class, total, rng)

    # Shuffle so the array is not ordered by class — critical for Hailo
    # calibration algorithms that process images in sequential mini-batches.
    rng.shuffle(selected)

    unit = "cluster(s)" if strategy == "diverse" else "class(es)"
    print(
        console.dim(
            f"  Building  calib_set.npy ({len(selected)} images, "
            f"{strategy} across {n_classes} {unit}, {size}×{size})..."
        )
    )

    calib_data = []
    for img_path in selected:
        img = Image.open(str(img_path)).convert("RGB").resize((size, size))
        calib_data.append(np.array(img, dtype=np.float32))

    calib_set = np.array(calib_data)
    np.save(str(output_path), calib_set)
    print(console.dim(f"  Written   calib_set.npy  shape={calib_set.shape}"))
    if len(selected) < total:
        print(
            console.yellow(
                f"  Warning   calibration set has {len(selected)} images, fewer than the"
                f" {total} requested (not enough distinct training images for"
                f" '{strategy}'); quantization may be less accurate."
            )
        )
    return len(selected), n_classes
