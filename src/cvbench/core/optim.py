"""Optimizer construction shared by every task's model builder."""
from __future__ import annotations

import keras

from cvbench.core.config import CVBenchConfig


def build_optimizer(cfg: CVBenchConfig) -> keras.optimizers.Optimizer:
    opt = cfg.training.optimizer
    lr = cfg.training.learning_rate
    if opt.type == "sgd":
        return keras.optimizers.SGD(
            learning_rate=lr,
            weight_decay=opt.weight_decay,
            momentum=opt.momentum,
        )
    return keras.optimizers.Adam(
        learning_rate=lr,
        weight_decay=opt.weight_decay,
    )
