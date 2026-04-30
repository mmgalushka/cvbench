import numpy as np


def aug_mask(
    img: np.ndarray,
    n_masks: int = 1,
    max_width: int = 30,
    fill_value: int = 0,
    orientation: str = "h",
    seed: int = None,
) -> np.ndarray:
    """Zero out N random bands along the given orientation ('h' or 'v').

    Each band width is sampled uniformly from [1, max_width].
    Forces the model to classify from the full pattern rather than
    single bright features or localised artifacts.
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    h, w = img.shape[:2]
    axis_size = h if orientation == "h" else w
    for _ in range(n_masks):
        width = rng.integers(1, max_width + 1)
        pos = rng.integers(0, max(1, axis_size - width))
        if orientation == "h":
            out[pos:pos + width, :] = fill_value
        else:
            out[:, pos:pos + width] = fill_value
    return out


def aug_mask_h(
    img: np.ndarray,
    n_masks: int = 1,
    max_width: int = 30,
    fill_value: int = 0,
    seed: int = None,
) -> np.ndarray:
    """Zero out N random horizontal bands (rows)."""
    return aug_mask(img, n_masks, max_width, fill_value, "h", seed)


def aug_mask_v(
    img: np.ndarray,
    n_masks: int = 1,
    max_width: int = 40,
    fill_value: int = 0,
    seed: int = None,
) -> np.ndarray:
    """Zero out N random vertical bands (columns)."""
    return aug_mask(img, n_masks, max_width, fill_value, "v", seed)
