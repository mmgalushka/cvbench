"""Shape detection model outputs into prediction results (NumPy only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cvbench.core._names import class_label


def build_detection_result(outputs: list[np.ndarray], cfg, conf: float | None) -> dict:
    """Decode one image's raw head outputs (no batch axis) into detections.

    The single detection result builder. Exported models don't guarantee output
    order (TFLite sorts tensors by name), so scales are re-ordered to match
    ``cfg.detection.strides`` — finest stride first means largest grid first.
    ``conf`` overrides the run's ``detection.conf_threshold``.
    """
    from cvbench.detection.decode import decode_batch

    det = cfg.detection
    if not det.anchors or not det.strides:
        raise ValueError(
            "This detection run predates anchor-based decoding and cannot be "
            "used for inference — retrain with the current detection head."
        )
    class_names = cfg.data.classes
    scales = sorted(outputs, key=lambda o: o.shape[0], reverse=True)
    dets = decode_batch(
        [o[None] for o in scales],
        num_classes=len(class_names),
        anchors=det.anchors,
        strides=det.strides,
        conf_threshold=det.conf_threshold if conf is None else conf,
        max_detections=det.max_detections,
        nms_iou_threshold=det.iou_threshold,
    )[0]
    return {
        "detections": [
            {
                "class_index": d["class_id"],
                "class_name": class_label(class_names, d["class_id"]),
                "confidence": d["confidence"],
                "x": d["x"],
                "y": d["y"],
                "w": d["w"],
                "h": d["h"],
            }
            for d in dets
        ],
    }


def build_detection_results(
    images: list[str],
    outputs_list: list[list[np.ndarray]],
    cfg,
    conf: float | None,
) -> list[dict]:
    """Per-image detection results, tagged with the file name."""
    return [
        {"filename": Path(img_path).name, **build_detection_result(outs, cfg, conf)}
        for img_path, outs in zip(images, outputs_list, strict=True)
    ]
