"""Shared runtime setup for the training and evaluation services."""
from __future__ import annotations

import platform

from cvbench.core import _console


def print_device_banner(verb: str) -> None:
    """Print a GPU/CPU banner. VERB is the present participle, e.g. "training"."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print(_console.green(f"🟢 Apple Silicon GPU (Metal) detected — {verb} on {len(gpus)} device(s)"))
        else:
            names = ", ".join(g.name for g in gpus)
            print(_console.green(f"🟢 GPU detected: {len(gpus)} device(s) — {names}"))
    else:
        _console.warning(f"GPU not available, {verb} on CPU")
