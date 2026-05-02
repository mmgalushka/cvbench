import numpy as np


def aug_rf_transmission(
    img: np.ndarray,
    bandwidth: float = 0.06,
    brightness_delta: int = 30,
    rectangular: bool = True,
    edge_rolloff: float = 0.003,
    ripple: float = 0.05,
    drift_speed: float = 0.02,
    noise_floor: float = 0.08,
    seed: int = None,
) -> np.ndarray:
    """
    Inject a synthetic RF transmission into the spectrogram image.

    Carrier frequency is placed automatically in the quietest available band.
    Signal strength is calibrated to the 95th-percentile brightness of the
    existing image ± brightness_delta counts.

    rectangular=True  → hard flat-top band with cosine-tapered edges (modulated carrier)
    rectangular=False → Gaussian rolloff (beacon / unmodulated narrowband signal)
    """
    if isinstance(rectangular, str):
        rectangular = rectangular.lower() not in ("false", "0", "no")

    rng = np.random.default_rng(seed)
    rows, cols = img.shape[0], img.shape[1]

    gray = img[..., 0] if img.ndim == 3 else img   # (H, W) for energy analysis

    # ── find quietest frequency band ──────────────────────────────────────
    bw_px = max(1, int(bandwidth * cols))
    col_energy = gray.mean(axis=0).astype(np.float32)

    kernel = np.ones(bw_px, dtype=np.float32) / bw_px
    windowed = np.convolve(col_energy, kernel, mode="same")

    margin = bw_px // 2
    if margin > 0 and 2 * margin < cols:
        windowed[:margin] = np.inf
        windowed[cols - margin:] = np.inf

    carrier = float(np.argmin(windowed)) / cols     # normalised [0..1]

    # ── calibrate strength to existing brightness ──────────────────────────
    ref = float(np.percentile(gray, 95))
    delta = int(rng.integers(-brightness_delta, brightness_delta + 1))
    target_peak = int(np.clip(ref + delta, 1, 255))
    strength = target_peak / 255.0

    # ── synthesise transmission in frequency domain ────────────────────────
    fx = np.linspace(0.0, 1.0, cols, dtype=np.float32)
    half_bw = bandwidth / 2.0
    dist = np.abs(fx - carrier)

    if rectangular:
        amp = np.zeros(cols, dtype=np.float32)
        inside = dist <= (half_bw - edge_rolloff)
        taper  = (~inside) & (dist <= (half_bw + edge_rolloff))

        amp[inside] = 1.0
        if edge_rolloff > 0 and taper.any():
            t = (dist[taper] - (half_bw - edge_rolloff)) / (2 * edge_rolloff)
            amp[taper] = 0.5 * (1.0 + np.cos(np.pi * t))

        occupied = dist < half_bw
        if occupied.any():
            amp[occupied] *= 1.0 - ripple * (
                0.5 - 0.5 * np.cos(np.pi * dist[occupied] / half_bw)
            )
    else:
        # Gaussian — fuzzy beacon / unmodulated narrowband signal
        sigma = half_bw * 0.5
        amp = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)

    # ── amplitude fading (random walk over time axis) ─────────────────────
    steps = rng.uniform(-1.0, 1.0, rows).astype(np.float32) * drift_speed * 2
    fade = np.clip(np.cumsum(steps) + 0.7, 0.1, 1.0)

    # ── band-limited noise ────────────────────────────────────────────────
    band_mask = (dist <= half_bw + edge_rolloff).astype(np.float32)
    tx_noise = rng.normal(0, noise_floor, (rows, cols)).astype(np.float32)
    tx_noise *= band_mask[np.newaxis, :]

    # ── assemble synthetic layer (H, W) ───────────────────────────────────
    layer = np.clip(strength * np.outer(fade, amp) + tx_noise, 0.0, 1.0)

    # ── composite additively onto input ───────────────────────────────────
    out = img.astype(np.float32)
    out += (layer * 255.0)[:, :, np.newaxis] if img.ndim == 3 else layer * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)
