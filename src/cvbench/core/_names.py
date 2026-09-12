"""Shared filesystem-safe name validation.

Used wherever a user-supplied string becomes a directory or file name —
experiment run names (`core/runs.py`) and saved augmentation config names
(`core/augmentations_store.py`) share the exact same safety rules.
"""
import re

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]*$")


def validate_slug(name: str, what: str = "Name") -> str:
    """Raise ValueError if name is not safe to use as a directory/file name."""
    if not name or len(name) > 100:
        raise ValueError(f"{what} must be 1–100 characters.")
    if not _NAME_RE.match(name):
        raise ValueError(
            f"{what} may only contain letters, digits, underscores, and hyphens, "
            "and must start with a letter or digit."
        )
    return name
