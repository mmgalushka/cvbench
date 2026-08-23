"""Classification task: class-folder datasets, softmax head, confusion-matrix eval.

``ClassificationTask`` wires the modules in this package to the ``Task``
interface — pure delegation, no new logic.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

from cvbench.classification.data import build_dataset as _build_dataset
from cvbench.classification.data import build_datasets as _build_datasets
from cvbench.classification.data import get_class_names as _get_class_names
from cvbench.classification.evaluator import evaluate as _evaluate
from cvbench.classification.model import build_model as _build_model
from cvbench.core.task import DatasetSpec, Task
from cvbench.datasets.stats import get_class_distribution, print_imbalance_warning, resolve_class_weights


class ClassificationTask(Task):
    name = "classification"

    def resolve_layout(self, cfg) -> DatasetSpec:
        class_names = _get_class_names(cfg.data.train_dir)
        cfg.data.classes = class_names
        cfg.model.num_classes = len(class_names)
        return DatasetSpec(
            class_names=class_names,
            train_dir=cfg.data.train_dir,
            val_dir=cfg.data.val_dir,
            test_dir=cfg.data.test_dir,
            num_train_images=self.count_images(cfg.data.train_dir),
        )

    def count_images(self, directory: str) -> int:
        return sum(1 for _ in Path(directory).glob("*/*"))

    def build_datasets(self, cfg, spec: DatasetSpec):
        train_ds, val_ds, _class_names, num_train = _build_datasets(cfg)
        return train_ds, val_ds, num_train

    def build_eval_dataset(self, cfg, spec: DatasetSpec):
        with contextlib.redirect_stdout(io.StringIO()):
            return _build_dataset(cfg.data.test_dir, spec.class_names, cfg, training=False)

    def fit_class_weight(self, cfg, spec: DatasetSpec):
        class_dist = get_class_distribution(cfg.data.train_dir)
        print_imbalance_warning(class_dist, cfg.training.class_weight)
        return resolve_class_weights(cfg.training.class_weight, class_dist, spec.class_names)

    def build_model(self, cfg):
        return _build_model(cfg)

    def headline_metrics(self, final_metrics: dict) -> dict:
        return {
            "val_accuracy": final_metrics.get("val_accuracy"),
            "val_loss": final_metrics.get("val_loss"),
        }

    def evaluate(self, model, eval_ds, cfg, spec: DatasetSpec, run_dir, output_dir=None) -> dict:
        return _evaluate(
            model=model,
            test_ds=eval_ds,
            class_names=spec.class_names,
            run_dir=run_dir,
            test_dir=cfg.data.test_dir,
            output_dir=output_dir,
        )

    def test_score(self, report: dict) -> tuple[str, float | None]:
        return "accuracy", report["overall"]["value"]
