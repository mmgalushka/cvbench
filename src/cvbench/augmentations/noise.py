import numpy as np


def aug_salt_pepper(img: np.ndarray, density: float, seed: int = None) -> np.ndarray:
    """Replace `density` fraction of pixels with 0 or 255 (50/50 split)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    spatial = img.shape[:2]
    mask = rng.random(spatial) < density
    salt = rng.random(spatial) < 0.5
    out[mask & salt]  = 255
    out[mask & ~salt] = 0
    return out
