"""Detection-only config section (TF-free)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    """Detection-only settings (``DetectionTask.config_section``): the loader and
    ``save_config`` only read and write this section for detection runs."""
    strides: list = field(default_factory=lambda: [16, 32])  # output strides of the YOLO heads
    anchors: list = field(default_factory=list)  # resolved once (see detection/anchors.py),
    # then persisted here — [[ [w, h] x anchors_per_scale ] x len(strides)], smallest-area
    # anchors first (assigned to the smallest stride).
    anchors_per_scale: int = 3
    ignore_iou_threshold: float = 0.5  # non-assigned anchors overlapping a GT above this are
    # excluded from the objectness loss entirely (neither positive nor negative)
    noobj_weight: float = 0.5     # objectness loss weight on negative (non-object) cells
    conf_threshold: float = 0.25  # minimum score for a decoded detection to count
    iou_threshold: float = 0.5    # IoU at which a prediction counts as matching a ground truth
    max_detections: int = 100     # top-k cap on decoded detections per image
