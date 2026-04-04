import numpy as np


def aug_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction. gamma<1 brightens, gamma>1 darkens."""
    out = 255.0 * (img.astype(np.float32) / 255.0) ** gamma
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_exposure(img: np.ndarray, stops: float) -> np.ndarray:
    """Simulate exposure change by `stops` f-stops (positive=brighter)."""
    out = img.astype(np.float32) * (2.0 ** stops)
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_fog(img: np.ndarray, strength: float) -> np.ndarray:
    """Blend image with white. strength=0 -> no effect, strength=1 -> pure white."""
    out = img.astype(np.float32) * (1 - strength) + 255.0 * strength
    return np.clip(out, 0, 255).astype(np.uint8)
