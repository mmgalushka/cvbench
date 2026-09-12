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
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import keras
    import tensorflow as tf

    from cvbench.core.config import CVBenchConfig


@dataclass(frozen=True)
class DatasetSpec:
    """The resolved shape of a dataset, as determined by ``Task.resolve_layout``."""

    class_names: list[str]
    train_dir: str
    val_dir: str  # "" when the task auto-splits from train_dir
    test_dir: str
    num_train_images: int


class Task(abc.ABC):
    """A training scenario: classification, detection, and (later) others.

    Instances are stateless — all state lives in the ``CVBenchConfig`` and
    ``DatasetSpec`` passed into every method — so ``cvbench.tasks.get_task``
    can hand out a fresh instance per call for free.
    """

    name: ClassVar[str]

    # ---- layout / config ---------------------------------------------------

    @abc.abstractmethod
    def resolve_layout(self, cfg: CVBenchConfig) -> DatasetSpec:
        """Inspect the dataset on disk and resolve it into a DatasetSpec.

        The sole writer of ``cfg.data.{train,val,test}_dir``, ``cfg.data.classes``
        and ``cfg.model.num_classes`` — no other code may assign these.
        """

    def validate_config(self, cfg: CVBenchConfig) -> list[str]:
        """Return human-readable errors for a config this task cannot run.

        The caller (``run_training``) raises when this is non-empty. The
        default accepts anything.
        """
        return []

    @abc.abstractmethod
    def count_images(self, directory: str) -> int:
        """Count images under DIRECTORY for this task's layout."""

    # ---- data ---------------------------------------------------------------

    @abc.abstractmethod
    def build_datasets(
        self, cfg: CVBenchConfig, spec: DatasetSpec
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, int]:
        """Build (train_ds, val_ds, num_train_samples)."""

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

    def load_model(self, path: str) -> keras.Model:
        """Load a saved model for evaluation/prediction/export.

        Overriding this is how a task forces the import of the module that
        registers its custom losses/layers via
        ``keras.saving.register_keras_serializable`` *before* deserialization
        — Keras 3.5 only resolves a registered name if its defining module
        has already been imported.
        """
        import keras

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
            return keras.saving.load_model(path)

    # ---- reporting --------------------------------------------------------

    @abc.abstractmethod
    def headline_metrics(self, final_metrics: dict) -> dict:
        """Map a Keras History's final-epoch metrics to update_run_status() kwargs."""

    @abc.abstractmethod
    def evaluate(
        self,
        model: keras.Model,
        eval_ds: tf.data.Dataset,
        cfg: CVBenchConfig,
        spec: DatasetSpec,
        run_dir: str,
        output_dir: str | None,
    ) -> dict:
        """Run evaluation and return the report dict (also written to disk)."""

    @abc.abstractmethod
    def test_score(self, report: dict) -> tuple[str, float | None]:
        """Return (metric_name, value) — the run's single primary test score."""
