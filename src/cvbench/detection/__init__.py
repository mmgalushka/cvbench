"""Detection task: YOLO-layout datasets, 2-scale anchor-based YOLO head, mAP eval.

``DetectionTask`` wires this package's modules to the ``Task`` interface. The
TensorFlow-backed modules (``data``, ``losses``, ``model``, ``evaluator``) are
imported inside the methods that need them, so asking the task for its
metadata, report rendering or prediction shaping never loads TensorFlow.
"""
from __future__ import annotations

from pathlib import Path

from cvbench.core.config import TransformConfig
from cvbench.core.task import VAL_LOSS, DatasetSpec, MetricSpec, Task
from cvbench.datasets.layout import detection_layout_errors, is_yolo_dataset, list_images, yolo_class_names
from cvbench.detection import predict as _predict
from cvbench.detection import view as _view
from cvbench.detection.anchors import resolve_anchors

# EfficientNet's Swish/SiLU activations are known to quantize poorly to
# INT8 — detection defaults to a pure-ReLU backbone instead, unless the user
# passed --backbone explicitly (classification keeps its own default).
_DEFAULT_DETECTION_BACKBONE = "resnet_18"

# The 5 geometric transforms move pixels, so they'd desynchronize an image
# from its (untouched) boxes. The 18 custom aug_* transforms and the 3
# photometric keras_* ones (brightness/contrast/noise) are box-safe.
_GEOMETRIC_KERAS_TRANSFORMS = frozenset({
    "keras_flip", "keras_rotation", "keras_zoom", "keras_translation", "keras_crop",
})


class DetectionTask(Task):
    name = "detection"
    short_name = "det"
    primary_metric = MetricSpec("map50", "mAP@50", "max")
    sweep_metric = "map50"
    selection_metric_names = ("val_loss", "map50")
    config_section = "detection"
    hidden_config_prefixes = ("detection.anchors",)
    predict_option_names = frozenset({"conf"})

    def metrics_schema(self) -> tuple[MetricSpec, ...]:
        return (VAL_LOSS, self.primary_metric)

    @classmethod
    def matches_layout(cls, data_dir: Path) -> bool:
        return is_yolo_dataset(data_dir)

    def layout_errors(self, data_dir: Path) -> list[str]:
        return detection_layout_errors(data_dir)

    def resolve_layout(self, cfg) -> DatasetSpec:
        root = Path(cfg.data.data_dir)
        class_names = yolo_class_names(root)
        cfg.data.classes = class_names
        cfg.model.num_classes = len(class_names)

        if not cfg.model.backbone_explicit:
            cfg.model.backbone = _DEFAULT_DETECTION_BACKBONE

        train_dir = root / "images" / "train"
        val_dir = root / "images" / "val"
        test_dir = root / "images" / "test"

        cfg.data.train_dir = str(train_dir)
        cfg.data.val_dir = str(val_dir) if val_dir.is_dir() else ""
        cfg.data.test_dir = str(test_dir) if test_dir.is_dir() else ""

        resolve_anchors(cfg)

        return DatasetSpec(
            class_names=class_names,
            train_dir=cfg.data.train_dir,
            val_dir=cfg.data.val_dir,
            test_dir=cfg.data.test_dir,
            num_train_images=self.count_images(cfg.data.train_dir),
        )

    def count_images(self, directory: str) -> int:
        d = Path(directory)
        return len(list_images(d)) if d.is_dir() else 0

    def build_datasets(self, cfg, spec: DatasetSpec):
        from cvbench.detection.data import build_detection_dataset

        train_ds = build_detection_dataset(
            spec.train_dir, cfg.data.data_dir, spec.class_names, cfg, training=True,
            transforms=self.filter_transforms(cfg.augmentation.transforms),
        )
        val_ds = build_detection_dataset(
            spec.val_dir, cfg.data.data_dir, spec.class_names, cfg, training=False
        )
        return train_ds, val_ds, spec.num_train_images

    def build_eval_dataset(self, cfg, spec: DatasetSpec):
        from cvbench.detection.data import build_detection_dataset

        return build_detection_dataset(
            spec.test_dir, cfg.data.data_dir, spec.class_names, cfg, training=False
        )

    def filter_transforms(self, transforms: list) -> list:
        kept, dropped = [], []
        for t in transforms:
            if isinstance(t, TransformConfig) and t.name in _GEOMETRIC_KERAS_TRANSFORMS:
                dropped.append(t.name)
                continue
            kept.append(t)

        if dropped:
            from cvbench.core import _console
            _console.warning(
                f"Dropping geometric augmentation(s) for detection: {', '.join(dropped)}"
                " — they would move pixels without moving the boxes."
            )
        return kept

    def build_model(self, cfg):
        from cvbench.detection.model import build_model
        return build_model(cfg)

    def load_model(self, path: str):
        # Importing cvbench.detection.losses here registers YoloLoss, so
        # register_keras_serializable can resolve "YoloLoss" by name —
        # belt-and-braces, also pass it explicitly as custom_objects.
        import warnings

        import keras

        from cvbench.detection.losses import YoloLoss

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
            return keras.saving.load_model(path, custom_objects={"YoloLoss": YoloLoss})

    def headline_metrics(self, final_metrics: dict) -> dict:
        # Detection has no classification accuracy; the training headline
        # stays val_loss, with mAP computed as a post-hoc `evaluate` step.
        return {"val_accuracy": None, "val_loss": final_metrics.get("val_loss")}

    def evaluate(self, model, eval_ds, cfg, spec: DatasetSpec, run_dir, output_dir=None, conf=None) -> dict:
        from cvbench.detection.evaluator import evaluate as _evaluate
        return _evaluate(
            model=model,
            test_ds=eval_ds,
            class_names=spec.class_names,
            run_dir=run_dir,
            test_dir=spec.test_dir,
            ds_root=cfg.data.data_dir,
            anchors=cfg.detection.anchors,
            strides=cfg.detection.strides,
            conf_threshold=cfg.detection.conf_threshold if conf is None else conf,
            iou_threshold=cfg.detection.iou_threshold,
            max_detections=cfg.detection.max_detections,
            output_dir=output_dir,
        )

    def test_score(self, report: dict) -> tuple[str, float | None]:
        return self.primary_metric.name, report["overall"]["value"]

    def predict(self, images, outputs_list, cfg, class_names, conf=None, top_k=1) -> list[dict]:
        return _predict.build_detection_results(images, outputs_list, cfg, conf)

    def predict_one(self, outputs, cfg, conf=None, top_k=1) -> dict:
        result = _predict.build_detection_result(outputs, cfg, conf)
        # Boxes are normalized top-left xywh in [0, 1] — what shared.js::buildBoxLayer() consumes.
        top = max(result["detections"], key=lambda d: d["confidence"], default=None)
        return {
            "task": self.name,
            **result,
            "class_name": top["class_name"] if top else None,
            "class_index": top["class_index"] if top else None,
            "confidence": top["confidence"] if top else None,
        }

    def render_predictions(self, formats_run, all_formats) -> None:
        _view.render_predictions(formats_run, all_formats)

    def eval_summary(self, report, val_accuracy=None):
        return _view.eval_summary(report, val_accuracy)

    def per_class_table(self, report):
        return _view.per_class_table(report)

    def report_advisories(self, report) -> list[str]:
        return _view.report_advisories(report)

    def render_report(self, report, val_accuracy=None, advisories=()) -> None:
        _view.render_report(report, val_accuracy, advisories)
