"""Shared runtime setup for the training and evaluation services."""
from __future__ import annotations

from cvbench.core.events import DeviceDetected, Listener, emit


def report_devices(purpose: str, listener: Listener | None) -> None:
    """Tell LISTENER which GPUs TensorFlow found. PURPOSE is a present participle, e.g. "training"."""
    import tensorflow as tf

    gpus = tuple(g.name for g in tf.config.list_physical_devices("GPU"))
    emit(listener, DeviceDetected(purpose, gpus))
