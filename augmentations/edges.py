import numpy as np


def aug_fade_horizontal(
    img: np.ndarray,
    fade_to: int = 128,
    side: str = 'right',
    strength: float = 1.0,
) -> np.ndarray:
    """
    Fade horizontal edge(s) toward fade_to grey value.
    side: 'left', 'right', or 'both'
    strength: 0=no effect, 1=full fade at edge
    """
    w = img.shape[1]
    if side == 'both':
        ramp = np.abs(np.linspace(-strength, strength, w, dtype=np.float32))
    else:
        ramp = np.linspace(0, strength, w, dtype=np.float32)
        if side == 'left':
            ramp = ramp[::-1]
    ramp = ramp.reshape((1, w) + (1,) * (img.ndim - 2))
    out = img.astype(np.float32) * (1 - ramp) + fade_to * ramp
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_fade_vertical(
    img: np.ndarray,
    fade_to: int = 128,
    side: str = 'bottom',
    strength: float = 1.0,
) -> np.ndarray:
    """
    Fade vertical edge(s) toward fade_to grey value.
    side: 'top', 'bottom', or 'both'
    """
    h = img.shape[0]
    if side == 'both':
        ramp = np.abs(np.linspace(-strength, strength, h, dtype=np.float32))
    else:
        ramp = np.linspace(0, strength, h, dtype=np.float32)
        if side == 'top':
            ramp = ramp[::-1]
    ramp = ramp.reshape((h, 1) + (1,) * (img.ndim - 2))
    out = img.astype(np.float32) * (1 - ramp) + fade_to * ramp
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_brighten_edges_h(
    img: np.ndarray,
    fade_to: int = 255,
    strength: float = 1.0,
    edge_fraction: float = 0.15,
) -> np.ndarray:
    """
    Brighten/darken both left and right edges with a Gaussian falloff.
    fade_to:       target pixel value at the edges (0=black, 255=white)
    strength:      peak blend weight at the outermost pixel
    edge_fraction: Gaussian sigma as fraction of half-width
    """
    w = img.shape[1]
    x = np.arange(w, dtype=np.float32)
    dist_from_edge = np.minimum(x, w - 1 - x)
    sigma = edge_fraction * (w / 2.0)
    weight = strength * np.exp(-(dist_from_edge ** 2) / (2 * sigma ** 2))
    weight = weight.reshape((1, w) + (1,) * (img.ndim - 2))
    out = img.astype(np.float32) * (1 - weight) + fade_to * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_brighten_edges_v(
    img: np.ndarray,
    fade_to: int = 255,
    strength: float = 1.0,
    edge_fraction: float = 0.15,
) -> np.ndarray:
    """
    Brighten/darken both top and bottom edges with a Gaussian falloff.
    fade_to:       target pixel value at the edges (0=black, 255=white)
    strength:      peak blend weight at the outermost pixel
    edge_fraction: Gaussian sigma as fraction of half-height
    """
    h = img.shape[0]
    y = np.arange(h, dtype=np.float32)
    dist_from_edge = np.minimum(y, h - 1 - y)
    sigma = edge_fraction * (h / 2.0)
    weight = strength * np.exp(-(dist_from_edge ** 2) / (2 * sigma ** 2))
    weight = weight.reshape((h, 1) + (1,) * (img.ndim - 2))
    out = img.astype(np.float32) * (1 - weight) + fade_to * weight
    return np.clip(out, 0, 255).astype(np.uint8)
