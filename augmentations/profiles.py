import numpy as np


def aug_random_profile_h(
    img: np.ndarray,
    n_changes: int = 5,
    max_delta: float = 60.0,
    seed: int = None,
) -> np.ndarray:
    """
    Apply a random smooth brightness profile along the horizontal axis.
    n_changes: number of Gaussian bumps along the width
    max_delta: maximum brightness shift (+/-) at any bump centre
    """
    w = img.shape[1]
    rng = np.random.default_rng(seed)
    sigma = w / (n_changes * 1.5)
    positions  = rng.uniform(0, w, n_changes)
    amplitudes = rng.uniform(-max_delta, max_delta, n_changes)
    x = np.arange(w, dtype=np.float32)
    profile = np.zeros(w, dtype=np.float32)
    for pos, amp in zip(positions, amplitudes):
        profile += amp * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
    profile = profile.reshape((1, w) + (1,) * (img.ndim - 2))
    return np.clip(img.astype(np.float32) + profile, 0, 255).astype(np.uint8)


def aug_random_profile_v(
    img: np.ndarray,
    n_changes: int = 5,
    max_delta: float = 60.0,
    seed: int = None,
) -> np.ndarray:
    """
    Apply a random smooth brightness profile along the vertical axis.
    n_changes: number of Gaussian bumps along the height
    max_delta: maximum brightness shift (+/-) at any bump centre
    """
    h = img.shape[0]
    rng = np.random.default_rng(seed)
    sigma = h / (n_changes * 1.5)
    positions  = rng.uniform(0, h, n_changes)
    amplitudes = rng.uniform(-max_delta, max_delta, n_changes)
    y = np.arange(h, dtype=np.float32)
    profile = np.zeros(h, dtype=np.float32)
    for pos, amp in zip(positions, amplitudes):
        profile += amp * np.exp(-((y - pos) ** 2) / (2 * sigma ** 2))
    profile = profile.reshape((h, 1) + (1,) * (img.ndim - 2))
    return np.clip(img.astype(np.float32) + profile, 0, 255).astype(np.uint8)
