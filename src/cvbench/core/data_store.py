"""Resolve dataset directories under `data/`, the same way `core/exp_store.py`
resolves experiment directories under `experiments/`.

Resolution only — no dataset-structure validation here. `train` already
validates dataset contents (train/, val/, test/ subfolders) downstream once
the directory is resolved.
"""
from __future__ import annotations

from cvbench.core.registry import Registry

DATA_DIR = "data"

DATA_REG = Registry(
    DATA_DIR,
    not_found_label="Dataset directory",
    entity_name="dataset",
    param_hint="DATA_DIR",
)


def validate_data_name(name: str) -> str:
    """Raise ValueError if name is not safe to use as a dataset directory name."""
    return DATA_REG.validate_name(name, what="Name")


def resolve_data_dir(name: str) -> str:
    """Resolve a dataset name or path to an existing directory.

    Accepts a full path (data/my_dataset) or a bare dataset name (my_dataset).
    If the given value does not exist as-is, looks under DATA_DIR.
    """
    DATA_REG.base_dir = DATA_DIR
    return DATA_REG.resolve(name)
