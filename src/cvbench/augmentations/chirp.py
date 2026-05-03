from __future__ import annotations

import random
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageDraw

JustificationMode = Literal["left", "right", "center", "random"]
_JUSTIFY_MODES = ("left", "right", "center")


def aug_chirp_artifacts(
    img: np.ndarray,
    num_chirps: int = 10,
    min_length: int = 30,
    max_length: int = 180,
    min_thickness: int = 1,
    max_thickness: int = 6,
    brightness: float = 0.6,
    fade_ratio: float = 0.2,
    cluster_vertically: bool = True,
    num_clusters: int = 3,
    cluster_spread: float = 15.0,
    justification: str = "random",
    num_anchors: int = 1,
    seed: int = None,
) -> np.ndarray:
    """Overlay horizontal chirp artifacts onto a spectrogram image using screen blending."""
    if justification not in (*_JUSTIFY_MODES, "random"):
        raise ValueError(
            f"justification must be one of {(*_JUSTIFY_MODES, 'random')}, got {justification!r}"
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    float_input = img.dtype in (np.float32, np.float64)
    if float_input:
        img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)

    grayscale = img_u8.ndim == 2
    if grayscale:
        img_u8 = np.stack([img_u8] * 3, axis=-1)

    has_alpha = img_u8.shape[2] == 4
    if has_alpha:
        alpha_channel = img_u8[:, :, 3:4]
        img_u8 = img_u8[:, :, :3]

    H, W = img_u8.shape[:2]
    mask = _render_chirp_mask(
        height=H,
        width=W,
        num_chirps=num_chirps,
        min_length=min_length,
        max_length=max_length,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        brightness=brightness,
        fade_ratio=fade_ratio,
        cluster_vertically=cluster_vertically,
        num_clusters=num_clusters,
        cluster_spread=cluster_spread,
        justification=justification,
        num_anchors=num_anchors,
    )

    img_f = img_u8.astype(np.float32) / 255.0
    mask_f = mask.astype(np.float32) / 255.0
    blended_f = 1.0 - (1.0 - img_f) * (1.0 - mask_f)
    blended_u8 = (np.clip(blended_f, 0.0, 1.0) * 255).astype(np.uint8)

    if has_alpha:
        blended_u8 = np.concatenate([blended_u8, alpha_channel], axis=2)
    if grayscale:
        blended_u8 = blended_u8[:, :, 0]

    if float_input:
        return blended_u8.astype(np.float32) / 255.0
    return blended_u8


def _render_chirp_mask(
    height: int,
    width: int,
    num_chirps: int,
    min_length: int,
    max_length: int,
    min_thickness: int,
    max_thickness: int,
    brightness: float,
    fade_ratio: float,
    cluster_vertically: bool,
    num_clusters: int,
    cluster_spread: float,
    justification: str,
    num_anchors: int,
) -> np.ndarray:
    mask_img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(mask_img)

    if cluster_vertically:
        k = max(1, num_clusters)
        cluster_centers = [random.randint(20, height - 20) for _ in range(k)]

    if justification != "random":
        anchors = [random.randint(0, width - 1) for _ in range(max(1, num_anchors))]

    for i in range(num_chirps):
        length = random.randint(min_length, max_length)
        thickness = random.randint(min_thickness, max_thickness)

        mode = random.choice(_JUSTIFY_MODES) if justification == "random" else justification

        if justification == "random":
            anchor = random.randint(0, width - 1)
        else:
            anchor = anchors[i % len(anchors)]

        if mode == "left":
            x0 = anchor
        elif mode == "right":
            x0 = anchor - length
        else:
            x0 = anchor - length // 2

        x0 = int(np.clip(x0, 0, max(0, width - length)))

        if cluster_vertically:
            center = cluster_centers[i % len(cluster_centers)]
            y0 = int(np.clip(
                np.random.normal(center, cluster_spread),
                thickness, height - thickness,
            ))
        else:
            y0 = random.randint(thickness, height - thickness)

        _draw_chirp(draw, x0, y0, length, thickness, brightness, fade_ratio, width)

    return np.array(mask_img)


def _draw_chirp(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    length: int,
    thickness: int,
    brightness: float,
    fade_ratio: float,
    canvas_width: int,
) -> None:
    fade_pixels = max(1, int(length * fade_ratio))
    half_h = thickness // 2
    y_top = y0 - half_h
    y_bot = y0 + (thickness - half_h)
    peak = int(brightness * 255)

    for col in range(length):
        if col < fade_pixels:
            alpha = col / fade_pixels
        elif col >= length - fade_pixels:
            alpha = (length - col) / fade_pixels
        else:
            alpha = 1.0
        v = int(peak * alpha)
        draw.rectangle([x0 + col, y_top, x0 + col, y_bot], fill=(v, v, v))
