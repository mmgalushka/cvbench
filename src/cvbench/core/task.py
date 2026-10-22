"""The Task interface — every training-scenario decision the orchestrators need.

``services/training.py`` and ``services/evaluation.py`` call a ``Task``
instance for every task-varying decision instead of branching on task type
themselves. Concrete implementations live in peer packages (e.g.
``cvbench.classification.ClassificationTask``), never in ``core/`` — this
module only declares the contract and knows nothing about implementations.

Adding a third task means writing a new ``Task`` subclass and registering it
in ``cvbench.tasks``; ``tests/test_tasks_contract.py`` is a conformance suite
that runs over every registered task and catches an incomplete one early.
"""
from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from cvbench.core.errors import NotFoundError

if TYPE_CHECKING:
    import keras
    import tensorflow as tf

    from cvbench.core.config import CVBenchConfig
    from cvbench.core.events import Listener
    from cvbench.core.fit import FitContext, FitResult
    from cvbench.core.layout import LayoutAdapter


@dataclass(frozen=True)
class DatasetSpec:
    """The resolved shape of a dataset, as determined by ``Task.resolve_layout``."""

    class_names: list[str]
    train_dir: str
    val_dir: str  # "" when the task auto-splits from train_dir
    test_dir: str
    num_train_images: int


@dataclass(frozen=True)
class MetricSpec:
    """One metric a task records on its runs: how to rank, label and format it."""

    name: str
    label: str
    direction: str  # "min" | "max"
    fmt: str = "pct"  # "pct" (0..1 shown as %) | "loss" (plain float)


VAL_LOSS = MetricSpec("val_loss", "Val Loss", "min", fmt="loss")

LABEL_KINDS = ("class", "boxes", "masks", "none")


@dataclass(frozen=True)
class LabelSpec:
    """What a task's samples are labelled with.

    ``class``: one class per image (a folder per class); ``boxes``: boxes with a class;
    ``masks``: an image-shaped target per sample (e.g. segmentation); ``none``: unlabeled
    (autoencoders, VAEs, GANs). Orchestrators ask this instead of assuming class names.
    """

    kind: str = "class"

    def __post_init__(self) -> None:
        if self.kind not in LABEL_KINDS:
            raise ValueError(f"Unknown label kind {self.kind!r}; expected one of {LABEL_KINDS}")

    @property
    def has_classes(self) -> bool:
        """Whether the dataset carries class names (``DatasetSpec.class_names`` is non-empty)."""
        return self.kind in ("class", "boxes")

    @property
    def joint_targets(self) -> bool:
        """Whether the target is an image-shaped array that geometric augmentation must move with the image."""
        return self.kind == "masks"


class TaskView(abc.ABC):
    """How a task shows itself in the terminal and tables: prediction and eval-report rendering.

    Stateless and TensorFlow-free. ``core/present.py`` decides *what* to show,
    ``core/render.py`` holds the terminal building blocks, and a task's view
    composes them into that task's own report body.
    """

    @abc.abstractmethod
    def render_predictions(self, formats_run: list[dict], all_formats: bool) -> None:
        """Print ``predict`` results to the terminal."""

    @abc.abstractmethod
    def eval_summary(self, report: dict, val_accuracy: float | None = None) -> list[tuple[str, str]]:
        """Ordered (label, value) headline facts of an eval report."""

    @abc.abstractmethod
    def per_class_table(self, report: dict) -> dict[str, list]:
        """Columns and formatted rows of the per-class metrics table."""

    def report_advisories(self, report: dict) -> list[str]:
        """Advisories derivable from an eval report alone."""
        return []

    @abc.abstractmethod
    def render_report(self, report: dict, val_accuracy: float | None = None, advisories=()) -> None:
        """Print the body of an eval report to the terminal."""


class Task(abc.ABC):
    """A training scenario: classification, detection, and (later) others.

    Instances are stateless — all state lives in the ``CVBenchConfig`` and
    ``DatasetSpec`` passed into every method — so ``cvbench.tasks.get_task``
    can hand out a fresh instance per call for free.

    Everything outside the model/data path (metadata, presentation,
    prediction shaping) is TensorFlow-free, so the CLI and WebUI can ask a
    task about itself without paying for a TensorFlow import.
    """

    name: ClassVar[str]
    short_name: ClassVar[str]  # run-name / table prefix, e.g. "cls"
    default_backbone: ClassVar[str]  # backbone used when --backbone is not passed
    primary_metric: ClassVar[MetricSpec]  # the run's single headline test score
    sweep_metric: ClassVar[str]  # default selection metric of a sweep
    # Metrics a sweep may rank trials on — never the test score of a task that also has a
    # validation-time metric (ranking on the test set would leak it into model selection).
    selection_metric_names: ClassVar[tuple[str, ...]]
    # (config section name, its dataclass) holding the task-only settings; the config
    # loader/saver only touch that section for runs of this task.
    config_section: ClassVar[tuple[str, type] | None] = None
    # Order in which `tasks.detect_task_name` sniffs dataset layouts: higher is tried first.
    # The catch-all task (matches any layout) must have the lowest priority.
    layout_priority: ClassVar[int] = 0
    # Name of the registered ``LayoutAdapter`` (``cvbench.core.layout``) that sniffs, validates
    # and resolves this task's datasets. When set, matches_layout/layout_errors/resolve_layout
    # delegate to it; a task may still override those methods directly.
    layout: ClassVar[str | None] = None
    label_spec: ClassVar[LabelSpec] = LabelSpec("class")
    hidden_config_prefixes: ClassVar[tuple[str, ...]] = ()  # bulky derived keys `runs show` skips
    predict_option_names: ClassVar[frozenset[str]] = frozenset()  # subset of {"conf", "top_k"}

    @property
    def run_prefix(self) -> str:
        return self.short_name

    def metrics_schema(self) -> tuple[MetricSpec, ...]:
        """Every metric a run of this task can record, primary one included."""
        return (VAL_LOSS, self.primary_metric)

    def metric(self, name: str) -> MetricSpec | None:
        return next((m for m in self.metrics_schema() if m.name == name), None)

    def predict_options(self) -> frozenset[str]:
        """Which of ``conf`` / ``top_k`` this task's predictions honour."""
        return self.predict_option_names

    def meta(self) -> dict[str, Any]:
        """JSON-able description of the task, served to the WebUI."""
        return {
            "name": self.name,
            "short_name": self.short_name,
            "primary_metric": self.primary_metric.name,
            "sweep_metric": self.sweep_metric,
            "selection_metrics": list(self.selection_metric_names),
            "metrics": [vars(m) for m in self.metrics_schema()],
            "predict_options": sorted(self.predict_options()),
        }

    # ---- dataset layout -------------------------------------------------

    @classmethod
    def matches_layout(cls, data_dir: Path) -> bool:
        """Whether DATA_DIR looks like this task's dataset layout (used to sniff the task)."""
        if cls.layout is not None:
            from cvbench.core.layout import get_layout

            return get_layout(cls.layout).sniff(Path(data_dir))
        return False

    def layout_errors(self, data_dir: Path) -> list[str]:
        """Human-readable problems with DATA_DIR as a dataset for this task."""
        if self.layout is not None:
            from cvbench.core.layout import get_layout

            return get_layout(self.layout).errors(Path(data_dir))
        return []

    # ---- layout / config ---------------------------------------------------

    def layout_adapter(self) -> LayoutAdapter | None:
        """The registered ``LayoutAdapter`` named by ``layout``, or None."""
        if self.layout is None:
            return None
        from cvbench.core.layout import get_layout

        return get_layout(self.layout)

    def resolve_layout(self, cfg: CVBenchConfig) -> DatasetSpec:
        """Inspect the dataset on disk and resolve it into a DatasetSpec.

        The sole writer of ``cfg.data.{train,val,test}_dir``, ``cfg.data.classes``
        and ``cfg.model.num_classes`` — no other code may assign these. Delegates to
        the task's ``LayoutAdapter`` when ``layout`` is set; otherwise override it.
        """
        adapter = self.layout_adapter()
        if adapter is None:
            raise NotImplementedError(f"{type(self).__name__} must set `layout` or override resolve_layout")
        return adapter.resolve(cfg, self)

    def validate_config(self, cfg: CVBenchConfig) -> list[str]:
        """Return human-readable errors for a config this task cannot run.

        The caller (``run_training``) raises when this is non-empty. The
        default checks the dataset layout on disk.
        """
        return self.layout_errors(Path(cfg.data.data_dir))

    @abc.abstractmethod
    def count_images(self, directory: str) -> int:
        """Count images under DIRECTORY for this task's layout."""

    # ---- data ---------------------------------------------------------------

    @abc.abstractmethod
    def build_datasets(
        self, cfg: CVBenchConfig, spec: DatasetSpec
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, int]:
        """Build (train_ds, val_ds, num_train_samples).

        Both go through ``core.input_pipeline.build_image_dataset``; train_ds
        is augmented with ``self.filter_transforms(cfg.augmentation.transforms)``.
        """

    @abc.abstractmethod
    def build_eval_dataset(self, cfg: CVBenchConfig, spec: DatasetSpec) -> tf.data.Dataset:
        """Build the dataset used by ``evaluate`` (typically over the test split)."""

    def filter_transforms(self, transforms: list) -> list:
        """Drop augmentation transforms this task cannot apply safely.

        The default keeps every transform (classification is box-free, so
        every transform — including the geometric ``keras_*`` ones — is safe).
        """
        return transforms

    def fit_class_weight(self, cfg: CVBenchConfig, spec: DatasetSpec) -> dict[int, float] | None:
        """Resolve a Keras-compatible ``{class_index: weight}`` dict, or None."""
        return None

    # ---- model ----------------------------------------------------------------

    @abc.abstractmethod
    def build_model(self, cfg: CVBenchConfig) -> keras.Model:
        """Build and compile a model for this task."""

    def build_models(self, cfg: CVBenchConfig, primary: keras.Model | None = None) -> dict[str, keras.Model]:
        """Every artifact's model by name, primary first; PRIMARY is the model ``build_model`` returned.

        The default is just that one model. A multi-model family (encoder + decoder,
        generator + discriminator) overrides this together with ``artifacts()`` to build the
        others around PRIMARY (``build_model`` then returns the primary one).
        """
        return {self.primary_artifact: primary if primary is not None else self.build_model(cfg)}

    # ---- artifacts ------------------------------------------------------------

    def artifacts(self) -> tuple[str, ...]:
        """Names of the models a run saves; the first is the primary one.

        The primary artifact lives at the checkpoint path itself (``best.keras``), every
        other one next to it (``best.<name>.keras``) — see ``artifact_path``. Evaluation,
        prediction and export operate on the primary artifact.
        """
        return ("model",)

    @property
    def primary_artifact(self) -> str:
        return self.artifacts()[0]

    def artifact_path(self, checkpoint: str | Path, name: str) -> Path:
        """Where artifact NAME is stored for the (primary) CHECKPOINT path."""
        checkpoint = Path(checkpoint)
        if name == self.primary_artifact:
            return checkpoint
        if name not in self.artifacts():
            raise ValueError(f"{self.name} has no artifact {name!r}; it has {self.artifacts()}")
        return checkpoint.with_name(f"{checkpoint.stem}.{name}{checkpoint.suffix}")

    def load_models(self, checkpoint: str) -> dict[str, keras.Model]:
        """Load every artifact of the run checkpoint CHECKPOINT (the primary artifact's path)."""
        models = {}
        for name in self.artifacts():
            path = self.artifact_path(checkpoint, name)
            if name != self.primary_artifact and not path.is_file():
                raise NotFoundError(f"Artifact '{name}' of {self.name} not found: {path}")
            models[name] = self.load_model(str(path))
        return models

    def prepare_resume(
        self, models: dict[str, keras.Model], cfg: CVBenchConfig, listener: Listener | None = None,
    ) -> None:
        """Adjust freshly reloaded MODELS for a resumed run (trainable flags, learning rate).

        The default re-applies ``fine_tune_from_layer`` to the primary model's backbone and
        syncs its optimizer learning rate to the config.
        """
        from cvbench.core.fit import KerasFitLoop

        KerasFitLoop.prepare_resume(models[self.primary_artifact], cfg, listener)

    # ---- training loop --------------------------------------------------------

    def fit(self, ctx: FitContext) -> FitResult:
        """Run the training loop over CTX and return what happened (see ``cvbench.core.fit``).

        The default is ``KerasFitLoop``: ``model.fit`` on the primary model with the standard
        callbacks. A custom loop (GAN, VAE) overrides this and reports each epoch through
        ``ctx.end_epoch`` so logging, checkpointing and events keep working.
        """
        from cvbench.core.fit import KerasFitLoop

        return KerasFitLoop().fit(ctx)

    # ---- export ---------------------------------------------------------------

    def export_targets(self) -> tuple[str, ...] | None:
        """Names from the Exporter registry (``services.export.EXPORTERS``) this task supports.

        None (the default) means every registered exporter.
        """
        return None

    def calibration_groups(self, cfg: CVBenchConfig) -> dict[str, list[Path]] | None:
        """Calibration image candidates for quantizing exporters, grouped for stratified sampling.

        None (the default) lets ``services/export/calibration.py`` take the training images
        grouped by class folder. Override for datasets that are not laid out that way.
        """
        return None

    def custom_objects(self) -> dict[str, Any]:
        """Custom losses/layers ``load_model`` needs to deserialize this task's models.

        A method, not an attribute, so an override can import them lazily.
        Passing them explicitly also forces the defining module to be imported
        *before* deserialization — Keras 3.5 only resolves a name registered
        via ``keras.saving.register_keras_serializable`` once its module is loaded.
        """
        return {}

    def load_model(self, path: str) -> keras.Model:
        """Load a saved model for evaluation/prediction/export."""
        import keras

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
            return keras.saving.load_model(path, custom_objects=self.custom_objects())

    # ---- reporting --------------------------------------------------------

    @abc.abstractmethod
    def headline_metrics(self, final_metrics: dict) -> dict:
        """Map a Keras History's final-epoch metrics to the run's headline metrics (None values are not recorded)."""

    def training_advisories(self, cfg: CVBenchConfig, final_metrics: dict) -> list[str]:
        """Advice for a run that finished training with FINAL_METRICS (none by default)."""
        return []

    @abc.abstractmethod
    def evaluate(
        self,
        model: keras.Model,
        eval_ds: tf.data.Dataset,
        cfg: CVBenchConfig,
        spec: DatasetSpec,
        run_dir: str,
        output_dir: str | None,
        conf: float | None = None,
        listener: Listener | None = None,
    ) -> dict:
        """Run evaluation and return the report dict (also written to disk).

        CONF overrides the detection score threshold; tasks without one ignore it.
        LISTENER receives an ``EvalProgress`` event per predicted batch.
        """

    def test_score(self, report: dict) -> tuple[str, float | None]:
        """Return (metric_name, value) — the run's single primary test score."""
        return self.primary_metric.name, report["overall"]["value"]

    # ---- prediction ---------------------------------------------------------

    @abc.abstractmethod
    def predict(
        self,
        images: list[str],
        outputs_list: list[list[Any]],
        cfg: CVBenchConfig,
        class_names: list[str] | None,
        conf: float | None = None,
        top_k: int | None = 1,
    ) -> list[dict]:
        """Shape raw model outputs (one list of head arrays per image) into per-image results."""

    @abc.abstractmethod
    def predict_one(
        self,
        outputs: list[Any],
        cfg: CVBenchConfig,
        conf: float | None = None,
        top_k: int | None = 1,
    ) -> dict:
        """Shape one image's raw head outputs (no batch axis) into a WebUI result."""

    # ---- presentation -----------------------------------------------------

    @abc.abstractmethod
    def view(self) -> TaskView:
        """This task's terminal/table presentation."""
