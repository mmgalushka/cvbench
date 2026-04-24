import numpy as np


def aug_fade(
    img: np.ndarray,
    fade_to: int = 128,
    side: str = 'end',
    strength: float = 1.0,
    orientation: str = 'h',
) -> np.ndarray:
    """Fade an edge toward fade_to along the given orientation ('h' or 'v').
    side: 'start', 'end', or 'both'
    strength: 0=no effect, 1=full fade at edge
    """
    axis_size = img.shape[1] if orientation == 'h' else img.shape[0]
    if side == 'both':
        ramp = np.abs(np.linspace(-strength, strength, axis_size, dtype=np.float32))
    else:
        ramp = np.linspace(0, strength, axis_size, dtype=np.float32)
        if side == 'start':
            ramp = ramp[::-1]
    if orientation == 'h':
        ramp = ramp.reshape((1, axis_size) + (1,) * (img.ndim - 2))
    else:
        ramp = ramp.reshape((axis_size, 1) + (1,) * (img.ndim - 2))
    out = img.astype(np.float32) * (1 - ramp) + fade_to * ramp
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_fade_horizontal(
    img: np.ndarray,
    fade_to: int = 128,
    side: str = 'right',
    strength: float = 1.0,
) -> np.ndarray:
    """Fade horizontal edge(s) toward fade_to grey value.
    side: 'left', 'right', or 'both'
    strength: 0=no effect, 1=full fade at edge
    """
    return aug_fade(img, fade_to, {'left': 'start', 'right': 'end', 'both': 'both'}[side], strength, 'h')


def aug_fade_vertical(
    img: np.ndarray,
    fade_to: int = 128,
    side: str = 'bottom',
    strength: float = 1.0,
) -> np.ndarray:
    """Fade vertical edge(s) toward fade_to grey value.
    side: 'top', 'bottom', or 'both'
    """
    return aug_fade(img, fade_to, {'top': 'start', 'bottom': 'end', 'both': 'both'}[side], strength, 'v')


def aug_brighten_edges(
    img: np.ndarray,
    fade_to: int = 255,
    strength: float = 1.0,
    edge_fraction: float = 0.15,
    orientation: str = 'h',
) -> np.ndarray:
    """Brighten/darken both edges with a Gaussian falloff along the given orientation ('h' or 'v').
    fade_to:       target pixel value at the edges (0=black, 255=white)
    strength:      peak blend weight at the outermost pixel
    edge_fraction: Gaussian sigma as fraction of half-axis
    """
    axis_size = img.shape[1] if orientation == 'h' else img.shape[0]
    x = np.arange(axis_size, dtype=np.float32)
    dist_from_edge = np.minimum(x, axis_size - 1 - x)
    sigma = edge_fraction * (axis_size / 2.0)
    weight = strength * np.exp(-(dist_from_edge ** 2) / (2 * sigma ** 2))
    if orientation == 'h':
        weight = weight.reshape((1, axis_size) + (1,) * (img.ndim - 2))
    else:
        weight = weight.reshape((axis_size, 1) + (1,) * (img.ndim - 2))
    out = img.astype(np.float32) * (1 - weight) + fade_to * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_brighten_edges_h(
    img: np.ndarray,
    fade_to: int = 255,
    strength: float = 1.0,
    edge_fraction: float = 0.15,
) -> np.ndarray:
    """Brighten/darken both left and right edges with a Gaussian falloff."""
    return aug_brighten_edges(img, fade_to, strength, edge_fraction, 'h')


def aug_brighten_edges_v(
    img: np.ndarray,
    fade_to: int = 255,
    strength: float = 1.0,
    edge_fraction: float = 0.15,
) -> np.ndarray:
    """Brighten/darken both top and bottom edges with a Gaussian falloff."""
    return aug_brighten_edges(img, fade_to, strength, edge_fraction, 'v')
