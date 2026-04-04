import numpy as np


def aug_gaussian_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise with std=sigma."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def aug_salt_pepper(img: np.ndarray, density: float, seed: int = 0) -> np.ndarray:
    """Replace `density` fraction of pixels with 0 or 255 (50/50 split)."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    mask = rng.random(img.shape) < density
    salt = rng.random(img.shape) < 0.5
    out[mask & salt]  = 255
    out[mask & ~salt] = 0
    return out
