import numpy as np


def aug_lines(
    img: np.ndarray,
    n_lines: int = 5,
    width: int = 2,
    brightness: int = 128,
    orientation: str = "h",
    seed: int = None,
) -> np.ndarray:
    """Draw N random lines along the given orientation ('h' or 'v')."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    axis_size = h if orientation == "h" else w
    for _ in range(n_lines):
        pos = rng.integers(0, axis_size)
        p0 = max(0, pos - width // 2)
        p1 = min(axis_size, p0 + width)
        if orientation == "h":
            out[p0:p1, :] = brightness
        else:
            out[:, p0:p1] = brightness
    return out


def aug_lines_h(
    img: np.ndarray,
    n_lines: int = 5,
    width: int = 2,
    brightness: int = 128,
    seed: int = None,
) -> np.ndarray:
    """Draw N random horizontal lines with given width and brightness."""
    return aug_lines(img, n_lines, width, brightness, "h", seed)


def aug_lines_v(
    img: np.ndarray,
    n_lines: int = 5,
    width: int = 2,
    brightness: int = 128,
    seed: int = None,
) -> np.ndarray:
    """Draw N random vertical lines with given width and brightness."""
    return aug_lines(img, n_lines, width, brightness, "v", seed)
