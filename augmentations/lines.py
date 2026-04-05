import numpy as np


def aug_lines_h(
    img: np.ndarray,
    n_lines: int = 5,
    width: int = 2,
    brightness: int = 128,
    seed: int = None,
) -> np.ndarray:
    """Draw N random horizontal lines with given width and brightness."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h = img.shape[0]
    for _ in range(n_lines):
        y  = rng.integers(0, h)
        y0 = max(0, y - width // 2)
        y1 = min(h, y0 + width)
        out[y0:y1, :] = brightness
    return out


def aug_lines_v(
    img: np.ndarray,
    n_lines: int = 5,
    width: int = 2,
    brightness: int = 128,
    seed: int = None,
) -> np.ndarray:
    """Draw N random vertical lines with given width and brightness."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    iw = img.shape[1]
    for _ in range(n_lines):
        x  = rng.integers(0, iw)
        x0 = max(0, x - width // 2)
        x1 = min(iw, x0 + width)
        out[:, x0:x1] = brightness
    return out
