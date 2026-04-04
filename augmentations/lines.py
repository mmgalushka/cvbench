import numpy as np


def aug_lines_h(
    img: np.ndarray,
    n_lines: int = 5,
    width_range: tuple = (1, 3),
    brightness_range: tuple = (0, 255),
    seed: int = 0,
) -> np.ndarray:
    """Draw N random horizontal lines with random width and brightness."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h = img.shape[0]
    for _ in range(n_lines):
        y      = rng.integers(0, h)
        width  = rng.integers(width_range[0], width_range[1] + 1)
        bright = rng.integers(brightness_range[0], brightness_range[1] + 1)
        y0 = max(0, y - width // 2)
        y1 = min(h, y0 + width)
        out[y0:y1, :] = bright
    return out


def aug_lines_v(
    img: np.ndarray,
    n_lines: int = 5,
    width_range: tuple = (1, 3),
    brightness_range: tuple = (0, 255),
    seed: int = 0,
) -> np.ndarray:
    """Draw N random vertical lines with random width and brightness."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    w = img.shape[1]
    for _ in range(n_lines):
        x      = rng.integers(0, w)
        width  = rng.integers(width_range[0], width_range[1] + 1)
        bright = rng.integers(brightness_range[0], brightness_range[1] + 1)
        x0 = max(0, x - width // 2)
        x1 = min(w, x0 + width)
        out[:, x0:x1] = bright
    return out
