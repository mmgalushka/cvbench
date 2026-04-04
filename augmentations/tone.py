import numpy as np


def aug_brightness(img: np.ndarray, delta: int) -> np.ndarray:
    """Shift pixel values by delta (+bright, -dark). Clips to [0, 255]."""
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def aug_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiply contrast around mid-grey (128) by factor."""
    out = 128 + factor * (img.astype(np.float32) - 128)
    return np.clip(out, 0, 255).astype(np.uint8)


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
