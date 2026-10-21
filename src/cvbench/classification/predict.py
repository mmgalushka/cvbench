"""Shape classification model outputs into prediction results (NumPy only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cvbench.core.names import class_label


def build_classification_result(
    probs, class_names: list[str] | None, top_k: int | None = 1
) -> dict:
    """Shape one image's class probabilities; ``top_k=None`` keeps every class.

    The single classification result builder — `predict` and the WebUI both use it,
    so a given image yields the same fields from either.
    """
    order = [int(i) for i in np.argsort(-np.asarray(probs), kind="stable")]
    top_idx = order[0]
    return {
        "class_index": top_idx,
        "confidence": float(probs[top_idx]),
        "class_name": class_label(class_names, top_idx),
        "top_k": [
            {"class_name": class_label(class_names, i), "confidence": float(probs[i])}
            for i in order[: top_k or len(order)]
        ],
        "probs": [float(x) for x in probs],
    }


def build_results(
    images: list[str],
    probs_list: list[np.ndarray],
    class_names: list[str] | None,
    top_k: int | None = 1,
) -> list[dict]:
    """Per-image classification results, tagged with the file name."""
    return [
        {"filename": Path(img_path).name, **build_classification_result(probs, class_names, top_k)}
        for img_path, probs in zip(images, probs_list, strict=True)
    ]
