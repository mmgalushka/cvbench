import numpy as np


def aug_salt_pepper(img: np.ndarray, density: float, seed: int = None) -> np.ndarray:
    """Replace `density` fraction of pixels with 0 or 255 (50/50 split)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    mask = rng.random(img.shape) < density
    salt = rng.random(img.shape) < 0.5
    out[mask & salt]  = 255
    out[mask & ~salt] = 0
    return out
