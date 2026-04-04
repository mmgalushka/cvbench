import numpy as np


def aug_lines_h(
    img: np.ndarray,
    n_lines: int = 5,
    width: tuple = (1, 3),
    brightness: tuple = (0, 255),
    seed: int = 0,
) -> np.ndarray:
    """Draw N random horizontal lines with random width and brightness."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h = img.shape[0]
    for _ in range(n_lines):
        y      = rng.integers(0, h)
        w      = rng.integers(width[0], width[1] + 1)
        bright = rng.integers(brightness[0], brightness[1] + 1)
        y0 = max(0, y - w // 2)
        y1 = min(h, y0 + w)
        out[y0:y1, :] = bright
    return out


def aug_lines_v(
    img: np.ndarray,
    n_lines: int = 5,
    width: tuple = (1, 3),
    brightness: tuple = (0, 255),
    seed: int = 0,
) -> np.ndarray:
    """Draw N random vertical lines with random width and brightness."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    iw = img.shape[1]
    for _ in range(n_lines):
        x      = rng.integers(0, iw)
        w      = rng.integers(width[0], width[1] + 1)
        bright = rng.integers(brightness[0], brightness[1] + 1)
        x0 = max(0, x - w // 2)
        x1 = min(iw, x0 + w)
        out[:, x0:x1] = bright
    return out
