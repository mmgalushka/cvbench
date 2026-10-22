"""A test-only unlabeled task: an autoencoder with two artifacts and its own fit loop.

It exists to prove the Task extension hooks without adding a model family to the product:

* ``label_spec`` is ``none`` (flat folders of images, no class names),
* ``layout`` names a ``LayoutAdapter`` registered here (``flat_images``),
* ``artifacts`` are ``("encoder", "decoder")`` — ``best.keras`` and ``best.decoder.keras``,
* ``fit`` is a hand-written GradientTape loop reporting through ``ctx.end_epoch``,
* ``export_targets`` narrows the Exporter registry to ``onnx`` and ``tflite``,
* ``calibration_groups`` supplies its own calibration candidates,
* its settings live in its own config section (``autoenc``).

Nothing outside ``tests/`` imports this module; it is registered through a faked entry point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cvbench.core.layout import LayoutAdapter, register_layout
from cvbench.core.task import DatasetSpec, LabelSpec, MetricSpec, Task, TaskView
from cvbench.datasets.layout import is_yolo_dataset, list_images


@dataclass
class AutoEncConfig:
    latent_dim: int = 8


class FlatImagesLayout(LayoutAdapter):
    """``<root>/train/*.png`` (+ optional ``val/``, ``test/``): images only, no labels."""

    name = "flat_images"

    def sniff(self, root: Path) -> bool:
        train = root / "train"
        return (
            not is_yolo_dataset(root) and train.is_dir() and bool(list_images(train))
            and not any(p.is_dir() for p in train.iterdir())
        )

    def errors(self, root: Path) -> list[str]:
        if not (root / "train").is_dir():
            return [f"No train/ folder found in '{root}'."]
        if not list_images(root / "train"):
            return [f"'{root / 'train'}' contains no images."]
        if not (root / "val").is_dir():
            return [f"No val/ folder found in '{root}'."]
        return []

    def resolve(self, cfg, task) -> DatasetSpec:
        root = Path(cfg.data.data_dir)
        cfg.data.train_dir = str(root / "train")
        cfg.data.val_dir = str(root / "val")
        cfg.data.test_dir = str(root / "test") if (root / "test").is_dir() else str(root / "val")
        cfg.data.classes = []
        cfg.model.num_classes = 0
        return DatasetSpec([], cfg.data.train_dir, cfg.data.val_dir, cfg.data.test_dir,
                           task.count_images(cfg.data.train_dir))


register_layout(FlatImagesLayout())


class AutoEncoderView(TaskView):
    def render_predictions(self, formats_run, all_formats):
        for fmt in formats_run:
            print(fmt)

    def eval_summary(self, report, val_accuracy=None):
        return [("Latent score", f"{report['overall']['value']:.3f}")]

    def per_class_table(self, report):
        return {"columns": [], "rows": []}

    def render_report(self, report, val_accuracy=None, advisories=()):
        print(report)


class AutoEncoderTask(Task):
    name = "autoencoder"
    short_name = "ae"
    default_backbone = "efficientnet_b0"  # unused: the model has no backbone
    primary_metric = MetricSpec("latent_score", "Latent Score", "max")
    sweep_metric = "val_loss"
    selection_metric_names = ("val_loss",)
    layout = "flat_images"
    layout_priority = 5
    label_spec = LabelSpec("none")
    config_section = ("autoenc", AutoEncConfig)

    def artifacts(self):
        return ("encoder", "decoder")

    def export_targets(self):
        return ("onnx", "tflite")

    def calibration_groups(self, cfg):
        return {"_all": list_images(Path(cfg.data.train_dir))}

    def count_images(self, directory):
        d = Path(directory)
        return len(list_images(d)) if d.is_dir() else 0

    def _dataset(self, cfg, directory, training):
        import numpy as np

        from cvbench.core.input_pipeline import build_image_dataset

        paths = [str(p) for p in list_images(Path(directory))]
        return build_image_dataset(
            paths, np.zeros((len(paths), 1), dtype="float32"), cfg.model.input_size,
            cfg.data.batch_size, training=training, seed=cfg.training.seed,
        )

    def build_datasets(self, cfg, spec):
        return (self._dataset(cfg, spec.train_dir, True), self._dataset(cfg, spec.val_dir, False),
                spec.num_train_images)

    def build_eval_dataset(self, cfg, spec):
        return self._dataset(cfg, spec.test_dir, False)

    def build_model(self, cfg):
        import keras

        size, latent = cfg.model.input_size, cfg.autoenc.latent_dim
        enc_in = keras.Input((size, size, 3))
        x = keras.layers.Rescaling(1 / 255)(enc_in)
        x = keras.layers.Conv2D(4, 3, strides=4, activation="relu")(x)
        return keras.Model(enc_in, keras.layers.Dense(latent)(keras.layers.Flatten()(x)), name="encoder")

    def build_models(self, cfg, primary=None):
        import keras

        size, latent = cfg.model.input_size, cfg.autoenc.latent_dim
        encoder = primary if primary is not None else self.build_model(cfg)
        dec_in = keras.Input((latent,))
        x = keras.layers.Dense(size * size * 3, activation="sigmoid")(dec_in)
        decoder = keras.Model(dec_in, keras.layers.Reshape((size, size, 3))(x), name="decoder")
        return {"encoder": encoder, "decoder": decoder}

    def prepare_resume(self, models, cfg, listener=None):
        """The loop creates its optimizer itself and the models have no backbone: nothing to adjust."""

    def fit(self, ctx):
        import keras
        import tensorflow as tf

        enc, dec = ctx.models["encoder"], ctx.models["decoder"]
        weights = enc.trainable_variables + dec.trainable_variables
        opt = keras.optimizers.Adam(ctx.cfg.training.learning_rate)
        train_it = iter(ctx.train_ds)
        history: dict[str, list[float]] = {"loss": [], "val_loss": []}

        def recon_loss(x):
            return tf.reduce_mean(tf.square(x / 255.0 - dec(enc(x))))

        epochs = 0
        for epoch in range(ctx.initial_epoch, ctx.cfg.training.epochs):
            train_losses = []
            for _ in range(ctx.steps_per_epoch):
                x, _ = next(train_it)
                with tf.GradientTape() as tape:
                    loss = recon_loss(x)
                opt.apply_gradients(zip(tape.gradient(loss, weights), weights, strict=True))
                train_losses.append(float(loss))
            val_losses = [float(recon_loss(x)) for x, _ in ctx.val_ds]
            logs = {"loss": sum(train_losses) / len(train_losses), "val_loss": sum(val_losses) / len(val_losses)}
            for k, v in logs.items():
                history[k].append(v)
            ctx.end_epoch(epoch, logs)
            epochs += 1
            if ctx.stop_requested():
                break
        from cvbench.core.fit import FitResult

        return FitResult(history=history, epochs=epochs)

    def headline_metrics(self, final_metrics):
        return {"val_loss": final_metrics.get("val_loss")}

    def evaluate(self, model, eval_ds, cfg, spec, run_dir, output_dir=None, conf=None, listener=None):
        import numpy as np

        spread = float(np.mean([np.var(model.predict(x, verbose=0)) for x, _ in eval_ds]))
        report = {
            "task": self.name,
            "overall": {"metric": self.primary_metric.name, "value": spread / (1.0 + spread)},
            "per_class": [],
        }
        (Path(run_dir) / "eval_report.json").write_text(json.dumps(report))
        return report

    def predict(self, images, outputs_list, cfg, class_names, conf=None, top_k=1):
        return [{"image": i, "latent": [float(v) for v in o[0]]} for i, o in zip(images, outputs_list, strict=True)]

    def predict_one(self, outputs, cfg, conf=None, top_k=1):
        return {"task": self.name, "latent": [float(v) for v in outputs[0]]}

    def view(self):
        return AutoEncoderView()
