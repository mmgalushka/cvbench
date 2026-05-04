import numpy as np
from scipy.ndimage import gaussian_filter


def _normalize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    a = arr - arr.mean()
    if arr.std() > 1e-9:
        a = a / arr.std() * std
    return np.clip(a + mean, 0, 255).astype(np.uint8)


def _perlin_like(H: int, W: int, scale: float, octaves: int, rng: np.random.Generator) -> np.ndarray:
    noise, amp, freq = np.zeros((H, W)), 1.0, 1.0
    for _ in range(octaves):
        noise += amp * gaussian_filter(rng.standard_normal((H, W)), sigma=max(1.0, scale / freq))
        amp *= 0.5
        freq *= 2.0
    return noise


def _scanline(H: int, W: int, rng: np.random.Generator, orientation: str) -> np.ndarray:
    bg = gaussian_filter(rng.standard_normal((H, W)), sigma=3.0)
    ori = rng.choice(["horizontal", "vertical"]) if orientation == "random" else orientation
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    if ori == "horizontal":
        stripes = np.sin(2 * np.pi * 0.25 * y) * 0.4
    elif ori == "vertical":
        stripes = np.sin(2 * np.pi * 0.25 * x) * 0.4
    else:  # both
        stripes = (np.sin(2 * np.pi * 0.25 * y) + np.sin(2 * np.pi * 0.25 * x)) * 0.2
    return _normalize(bg + stripes, mean=60, std=12)


def _stripes(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    base = gaussian_filter(rng.standard_normal((H, W)), sigma=(40.0, 6.0))
    fine = gaussian_filter(rng.standard_normal((H, W)), sigma=1.5)
    return _normalize(base + 0.15 * fine, mean=62, std=18)


def _turbulent(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    wx = (xs + 25.0 * _perlin_like(H, W, 30.0, 5, rng)).clip(0, W - 1).astype(int)
    wy = (ys + 25.0 * _perlin_like(H, W, 30.0, 5, rng)).clip(0, H - 1).astype(int)
    base = _perlin_like(H, W, 30.0, 5, rng)
    return _normalize(base[wy, wx], mean=49, std=25)


def _flow(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    bands = gaussian_filter(rng.standard_normal((H, W)), sigma=(8.0, 60.0))
    fine  = gaussian_filter(rng.standard_normal((H, W)), sigma=2.0)
    return _normalize(bands + 0.1 * fine, mean=52, std=18)


_PATTERNS = ["scanline", "stripes", "turbulent", "flow"]


def aug_interference(
    img: np.ndarray,
    pattern: str   = "random",
    alpha_min: float = 0.3,
    alpha_max: float = 0.6,
    orientation: str = "random",
    seed: int = None,
) -> np.ndarray:
    """Blend a synthetic interference pattern over a spectrogram image.

    pattern     : 'scanline' | 'stripes' | 'turbulent' | 'flow' | 'random'
    alpha_min   : minimum blend weight of the interference layer
    alpha_max   : maximum blend weight of the interference layer
    orientation : scanline only — 'horizontal' | 'vertical' | 'both' | 'random'
    """
    rng = np.random.default_rng(seed)

    pat   = rng.choice(_PATTERNS) if pattern == "random" else pattern
    alpha = float(rng.uniform(alpha_min, alpha_max))

    H, W = img.shape[:2]

    if pat == "scanline":
        noise = _scanline(H, W, rng, orientation)
    elif pat == "stripes":
        noise = _stripes(H, W, rng)
    elif pat == "turbulent":
        noise = _turbulent(H, W, rng)
    elif pat == "flow":
        noise = _flow(H, W, rng)
    else:
        raise ValueError(f"Unknown pattern: {pat!r}")

    layer = noise.astype(float)
    if img.ndim == 3:
        layer = layer[:, :, None]

    out = (1.0 - alpha) * img.astype(float) + alpha * layer
    return np.clip(out, 0, 255).astype(img.dtype)
