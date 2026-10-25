"""Built-in layout adapters: class folders (classification) and YOLO txt (detection), plus
Pascal VOC and COCO, which are recognised but not trainable (``data convert`` turns them into YOLO).

Importing this module registers them in ``cvbench.core.layout``. TensorFlow-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from cvbench.core.errors import ConfigError
from cvbench.core.layout import LayoutAdapter, register_layout
from cvbench.core.task import DatasetSpec
from cvbench.datasets import convert
from cvbench.datasets.layout import (
    classification_layout_errors,
    detection_layout_errors,
    is_yolo_dataset,
    yolo_class_names,
)

if TYPE_CHECKING:
    from cvbench.core.config import CVBenchConfig
    from cvbench.core.task import Task


def _foreign_error(root: Path, fmt: str) -> str:
    return (
        f"'{root}' is a {convert.FORMAT_LABELS[fmt]} dataset, which CVBench cannot train on "
        f"directly. Convert it to YOLO first: data convert {root} <output_dir>"
    )


class ClassFoldersLayout(LayoutAdapter):
    """``<root>/<split>/<class>/*.jpg`` — the catch-all, so it matches any non-YOLO folder."""

    name = "class_folders"

    def sniff(self, root: Path) -> bool:
        return not is_yolo_dataset(root)

    def errors(self, root: Path) -> list[str]:
        # A VOC/COCO folder lands here (it is not YOLO), where "run data split" would be wrong advice.
        foreign = convert.detect_format(root)
        if foreign:
            return [_foreign_error(root, foreign)]
        return classification_layout_errors(root)

    def resolve(self, cfg: CVBenchConfig, task: Task) -> DatasetSpec:
        class_names = sorted(p.name for p in Path(cfg.data.train_dir).iterdir() if p.is_dir())
        cfg.data.classes = class_names
        cfg.model.num_classes = len(class_names)
        return DatasetSpec(
            class_names=class_names,
            train_dir=cfg.data.train_dir,
            val_dir=cfg.data.val_dir,
            test_dir=cfg.data.test_dir,
            num_train_images=task.count_images(cfg.data.train_dir),
        )


class YoloLayout(LayoutAdapter):
    """``<root>/images/<split>`` + ``<root>/labels/<split>`` + ``data.yaml``."""

    name = "yolo"

    def sniff(self, root: Path) -> bool:
        return is_yolo_dataset(root)

    def errors(self, root: Path) -> list[str]:
        return detection_layout_errors(root)

    def resolve(self, cfg: CVBenchConfig, task: Task) -> DatasetSpec:
        root = Path(cfg.data.data_dir)
        class_names = yolo_class_names(root)
        cfg.data.classes = class_names
        cfg.model.num_classes = len(class_names)

        train_dir = root / "images" / "train"
        val_dir = root / "images" / "val"
        test_dir = root / "images" / "test"

        cfg.data.train_dir = str(train_dir)
        cfg.data.val_dir = str(val_dir) if val_dir.is_dir() else ""
        cfg.data.test_dir = str(test_dir) if test_dir.is_dir() else ""

        return DatasetSpec(
            class_names=class_names,
            train_dir=cfg.data.train_dir,
            val_dir=cfg.data.val_dir,
            test_dir=cfg.data.test_dir,
            num_train_images=task.count_images(cfg.data.train_dir),
        )


class ForeignLayout(LayoutAdapter):
    """A detection format CVBench reads only through ``data convert``: recognised, never trained on."""

    format: ClassVar[str]

    def sniff(self, root: Path) -> bool:
        return convert.detect_format(root) == self.format

    def errors(self, root: Path) -> list[str]:
        return [_foreign_error(root, self.format)]

    def resolve(self, cfg: CVBenchConfig, task: Task) -> DatasetSpec:
        raise ConfigError(self.errors(Path(cfg.data.data_dir))[0])


class VocLayout(ForeignLayout):
    """``Annotations/*.xml`` + ``JPEGImages/`` — Pascal VOC."""

    name = "voc"
    format = convert.FORMAT_VOC


class CocoLayout(ForeignLayout):
    """An instances JSON (``annotations/``, the root, or ``<split>/_annotations.coco.json``) plus images — COCO."""

    name = "coco"
    format = convert.FORMAT_COCO


register_layout(ClassFoldersLayout())
register_layout(YoloLayout())
register_layout(VocLayout())
register_layout(CocoLayout())
