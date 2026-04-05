import numpy as np
from PIL import Image, ImageFilter


def aug_blur(img: np.ndarray, radius: float) -> np.ndarray:
    """Apply Gaussian blur with given radius (pixels)."""
    single_channel = img.ndim == 3 and img.shape[2] == 1
    work = img[:, :, 0] if single_channel else img
    blurred = np.array(Image.fromarray(work).filter(ImageFilter.GaussianBlur(radius=radius)))
    return blurred[:, :, np.newaxis] if single_channel else blurred
