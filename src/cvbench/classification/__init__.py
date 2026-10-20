"""Classification task: class-folder datasets, softmax head, confusion-matrix eval.

``ClassificationTask`` wires the modules in this package to the ``Task``
interface — pure delegation, no new logic. The TensorFlow-backed modules
(``data``, ``model``, ``evaluator``) are imported inside the methods that need
them, so asking the task for its metadata, report rendering or prediction
shaping never loads TensorFlow.
"""
from __future__ import annotations

from pathlib import Path

from cvbench.classification import predict as _predict
from cvbench.classification import view as _view
from cvbench.core.task import VAL_LOSS, DatasetSpec, MetricSpec, Task
from cvbench.datasets.layout import classification_layout_errors, is_yolo_dataset
from cvbench.datasets.stats import get_class_distribution, print_imbalance_warning, resolve_class_weights

# A frozen run is flagged when its val accuracy is within this margin of chance.
_NEAR_CHANCE_MARGIN = 0.10


class ClassificationTask(Task):
    name = "classification"
    short_name = "cls"
    primary_metric = MetricSpec("accuracy", "Accuracy", "max")
    sweep_metric = "val_loss"
    selection_metric_names = ("val_loss", "val_accuracy")
    predict_option_names = frozenset({"top_k"})

    def metrics_schema(self) -> tuple[MetricSpec, ...]:
        return (VAL_LOSS, MetricSpec("val_accuracy", "Val Acc", "max"), self.primary_metric)

    @classmethod
    def matches_layout(cls, data_dir: Path) -> bool:
        return not is_yolo_dataset(data_dir)

    def layout_errors(self, data_dir: Path) -> list[str]:
        return classification_layout_errors(data_dir)

    def resolve_layout(self, cfg) -> DatasetSpec:
        from cvbench.classification.data import get_class_names as _get_class_names

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
        from cvbench.classification.data import build_datasets as _build_datasets

        transforms = self.filter_transforms(cfg.augmentation.transforms)
        train_ds, val_ds, _class_names, num_train = _build_datasets(cfg, transforms)
        return train_ds, val_ds, num_train

    def build_eval_dataset(self, cfg, spec: DatasetSpec):
        from cvbench.classification.data import build_dataset as _build_dataset

        return _build_dataset(cfg.data.test_dir, spec.class_names, cfg, training=False)

    def fit_class_weight(self, cfg, spec: DatasetSpec):
        class_dist = get_class_distribution(cfg.data.train_dir)
        print_imbalance_warning(class_dist, cfg.training.class_weight)
        return resolve_class_weights(cfg.training.class_weight, class_dist, spec.class_names)

    def build_model(self, cfg):
        from cvbench.classification.model import build_model as _build_model

        return _build_model(cfg)

    def headline_metrics(self, final_metrics: dict) -> dict:
        return {
            "val_accuracy": final_metrics.get("val_accuracy"),
            "val_loss": final_metrics.get("val_loss"),
        }

    def training_advisories(self, cfg, final_metrics: dict) -> list[str]:
        """Flag a frozen-backbone run that ended near chance accuracy."""
        val_acc = final_metrics.get("val_accuracy")
        num_classes = cfg.model.num_classes
        if val_acc is None or num_classes < 2 or cfg.model.fine_tune_from_layer != 0:
            return []
        chance = 1.0 / num_classes
        if val_acc > chance + _NEAR_CHANCE_MARGIN:
            return []
        return [
            f"val_accuracy {val_acc:.4f} is near chance ({chance:.4f} for {num_classes} classes) "
            "and the backbone is frozen, so only the head was trained. "
            "Try --fine-tune-from-layer -1 with a lower --lr (e.g. 1e-5), more --epochs, "
            "or --augmentation."
        ]

    def evaluate(
        self, model, eval_ds, cfg, spec: DatasetSpec, run_dir, output_dir=None, conf=None, listener=None,
    ) -> dict:
        from cvbench.classification.evaluator import evaluate as _evaluate

        return _evaluate(
            model=model,
            test_ds=eval_ds,
            class_names=spec.class_names,
            run_dir=run_dir,
            test_dir=cfg.data.test_dir,
            output_dir=output_dir,
            val_accuracy=cfg.run.val_accuracy,
            listener=listener,
        )

    def test_score(self, report: dict) -> tuple[str, float | None]:
        return self.primary_metric.name, report["overall"]["value"]

    def predict(self, images, outputs_list, cfg, class_names, conf=None, top_k=1) -> list[dict]:
        return _predict.build_results(images, [o[0] for o in outputs_list], class_names, top_k)

    def predict_one(self, outputs, cfg, conf=None, top_k=1) -> dict:
        return {
            "task": self.name,
            **_predict.build_classification_result(outputs[0], cfg.data.classes, top_k),
        }

    def render_predictions(self, formats_run, all_formats) -> None:
        _view.render_predictions(formats_run, all_formats)

    def eval_summary(self, report, val_accuracy=None):
        return _view.eval_summary(report, val_accuracy)

    def per_class_table(self, report):
        return _view.per_class_table(report)

    def render_report(self, report, val_accuracy=None, advisories=()) -> None:
        _view.render_report(report, val_accuracy, advisories)
