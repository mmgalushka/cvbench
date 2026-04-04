import numpy as np
from PIL import Image, ImageFilter


def aug_blur(img: np.ndarray, radius: float) -> np.ndarray:
    """Apply Gaussian blur with given radius (pixels)."""
    pil = Image.fromarray(img, mode='L')
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)
