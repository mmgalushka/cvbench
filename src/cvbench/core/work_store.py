"""Resolve named items under `workspace/`, the same way `core/exp_store.py`
resolves experiment directories under `experiments/`.

Generic placeholder for future workspace-scoped items (notebooks, scratch
dirs, ...) — not wired into any CLI command yet. `core/aug_store.py`'s
`workspace/augmentations/` remains its own store with its own `Registry`
rather than nesting under this one, since it is file-mode (`.yaml` configs)
while this one is directory-mode.
"""
from __future__ import annotations

from cvbench.core.registry import Registry

WORKSPACE_DIR = "workspace"

WORK_REG = Registry(
    WORKSPACE_DIR,
    not_found_label="Workspace item",
    entity_name="workspace item",
    param_hint="NAME",
)


def validate_work_name(name: str) -> str:
    """Raise ValueError if name is not safe to use as a workspace item name."""
    return WORK_REG.validate_name(name, what="Name")


def resolve_work_dir(name: str) -> str:
    """Resolve a workspace item name or path to an existing directory.

    Accepts a full path (workspace/my_item) or a bare name (my_item).
    If the given value does not exist as-is, looks under WORKSPACE_DIR.
    """
    WORK_REG.base_dir = WORKSPACE_DIR
    return WORK_REG.resolve(name)
