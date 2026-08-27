"""Tests for cvbench.detection.anchors — IoU-distance k-means + persistence."""
import numpy as np
import pytest

from cvbench.detection.anchors import kmeans_anchors


def test_kmeans_anchors_is_deterministic_under_a_fixed_seed():
    rng = np.random.RandomState(0)
    boxes = rng.rand(50, 2) * 0.5 + 0.05

    a1 = kmeans_anchors(boxes, k=3, seed=42)
    a2 = kmeans_anchors(boxes, k=3, seed=42)
    np.testing.assert_array_equal(a1, a2)


def test_kmeans_anchors_returns_k_anchors_sorted_by_ascending_area():
    boxes = np.array([[0.1, 0.1], [0.11, 0.09], [0.5, 0.5], [0.52, 0.48], [0.9, 0.9]])
    anchors = kmeans_anchors(boxes, k=3, seed=0)
    assert anchors.shape == (3, 2)
    areas = anchors[:, 0] * anchors[:, 1]
    assert np.all(np.diff(areas) >= 0)


def test_kmeans_anchors_iou_distance_prefers_shape_match_over_euclidean():
    # A small square and a large, very thin box can be Euclidean-close while
    # having near-zero shape IoU — IoU-distance k-means must not merge them.
    boxes = np.array([
        [0.05, 0.05], [0.06, 0.04], [0.04, 0.06],  # cluster A: small square-ish
        [0.9, 0.02], [0.85, 0.03], [0.95, 0.015],  # cluster B: very thin/wide
    ])
    anchors = kmeans_anchors(boxes, k=2, seed=0)
    # One anchor should be small/square-ish, the other thin/wide.
    aspect = [max(w, h) / min(w, h) for w, h in anchors]
    assert min(aspect) < 2.0
    assert max(aspect) > 5.0


def test_kmeans_anchors_raises_on_zero_boxes():
    with pytest.raises(ValueError):
        kmeans_anchors(np.zeros((0, 2)), k=3)


def test_kmeans_anchors_caps_k_at_available_distinct_boxes():
    boxes = np.array([[0.2, 0.2], [0.2, 0.2]])
    anchors = kmeans_anchors(boxes, k=5, seed=0)
    assert len(anchors) <= 2


# ---------------------------------------------------------------------------
# resolve_anchors — persistence into cfg.detection.anchors
# ---------------------------------------------------------------------------

def test_resolve_anchors_returns_existing_anchors_unchanged():
    from cvbench.core.config import CVBenchConfig
    from cvbench.detection.anchors import resolve_anchors

    cfg = CVBenchConfig()
    existing = [[[0.1, 0.1]], [[0.5, 0.5]]]
    cfg.detection.anchors = existing
    assert resolve_anchors(cfg) is existing


def test_resolve_anchors_derives_and_persists_from_labels(tmp_path):
    from cvbench.core.config import CVBenchConfig
    from cvbench.detection.anchors import resolve_anchors

    root = tmp_path / "yolo"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    from PIL import Image
    for i in range(6):
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img_dir / f"{i}.jpg")
        (lbl_dir / f"{i}.txt").write_text(f"0 0.5 0.5 {0.1 + i * 0.05} {0.1 + i * 0.05}\n")

    cfg = CVBenchConfig()
    cfg.data.data_dir = str(root)
    cfg.data.train_dir = str(img_dir)
    cfg.detection.strides = [16, 32]
    cfg.detection.anchors_per_scale = 2

    anchors = resolve_anchors(cfg)
    assert len(anchors) == 2  # one group per stride
    assert all(len(scale) == 2 for scale in anchors)  # anchors_per_scale
    assert cfg.detection.anchors is anchors  # persisted back into cfg


def test_resolve_anchors_raises_when_no_boxes_found(tmp_path):
    from cvbench.core.config import CVBenchConfig
    from cvbench.detection.anchors import resolve_anchors

    root = tmp_path / "yolo"
    img_dir = root / "images" / "train"
    img_dir.mkdir(parents=True)

    cfg = CVBenchConfig()
    cfg.data.data_dir = str(root)
    cfg.data.train_dir = str(img_dir)

    with pytest.raises(ValueError):
        resolve_anchors(cfg)
