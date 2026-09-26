"""Content hashing shared by ``data prep`` and ``data upsample``.

Hashes are derived from decoded pixel content (not file bytes), so a
re-encoded but visually identical image still hashes the same.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def hash_array(arr: np.ndarray) -> str:
    """Full MD5 hex digest of an image's pixel bytes."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def hash_image_file(path: Path) -> str:
    """Full MD5 hex digest of the image at PATH (decoded, RGB)."""
    from PIL import Image
    return hash_array(np.array(Image.open(path).convert("RGB")))
