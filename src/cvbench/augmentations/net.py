import numpy as np


def _line_mask(angle_deg: float, num_lines: int, line_width: int, h: int, w: int) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    diag      = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    cx, cy    = w / 2, h / 2

    line_dx = np.cos(angle_rad)
    line_dy = np.sin(angle_rad)
    perp_dx = np.cos(angle_rad + np.pi / 2)
    perp_dy = np.sin(angle_rad + np.pi / 2)

    spacing   = diag / max(num_lines, 1)
    start_off = -(num_lines - 1) / 2.0 * spacing
    half_w    = line_width // 2

    mask = np.zeros((h, w), dtype=bool)

    for i in range(num_lines):
        off = start_off + i * spacing
        lx  = cx + perp_dx * off
        ly  = cy + perp_dy * off

        for t in range(-diag, diag + 1):
            px = int(round(lx + line_dx * t))
            py = int(round(ly + line_dy * t))

            for dy in range(-half_w, half_w + 1):
                for dx in range(-half_w, half_w + 1):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        mask[ny, nx] = True

    return mask


def _stripe_mask(orientation: str, num_stripes: int, fill_pct: float, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    axis = h if orientation == "horizontal" else w
    period = max(1, axis // num_stripes)
    band   = max(1, int(round(period * fill_pct / 100.0)))

    if orientation == "horizontal":
        for start in range(0, h, period):
            mask[start:min(start + band, h), :] = True
    else:
        for start in range(0, w, period):
            mask[:, start:min(start + band, w)] = True

    return mask


def aug_net(
    img: np.ndarray,
    mode: str        = "both",
    angle: float     = 30.0,
    num_lines: int   = 20,
    line_width: int  = 1,
    line_brightness: int = 200,
    stripe: str      = "none",
    num_stripes: int = 4,
    stripe_fill: float = 50.0,
    seed: int        = None,
) -> np.ndarray:
    """Overlay a diagonal fishing-net line pattern on a grayscale spectrogram.

    mode            : which directions to draw — 'both', 'positive', or 'negative'
    angle           : line angle in degrees (1-89); sign is controlled by mode
    num_lines       : lines per direction
    line_width      : pixel thickness
    line_brightness : grayscale intensity of lines (0=black, 255=white)
    stripe      : confine lines to bands — 'none', 'horizontal', or 'vertical'
    num_stripes : number of visible stripe bands (used when stripe != 'none')
    stripe_fill : % of each stripe period that is visible (used when stripe != 'none')
    """
    _angle = float(np.clip(abs(angle), 1.0, 89.0))
    _lval  = float(np.clip(line_brightness, 0, 255))

    h, w = img.shape[:2]
    out  = img.astype(np.float32).copy()

    smask = _stripe_mask(stripe, num_stripes, stripe_fill, h, w) if stripe != "none" else None

    angles = {
        "positive": [_angle],
        "negative": [-_angle],
        "both":     [_angle, -_angle],
    }.get(mode, [_angle, -_angle])

    for a in angles:
        lmask = _line_mask(a, num_lines, line_width, h, w)
        if smask is not None:
            lmask = lmask & smask
        out[lmask] = _lval

    return np.clip(out, 0, 255).astype(img.dtype)
