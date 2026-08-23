"""Hand-computed tests for cvbench.detection.metrics — pure NumPy, no TensorFlow."""
import pytest

from cvbench.detection.metrics import bucket_samples, compute_detection_metrics, iou


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------

def test_iou_identical_boxes_is_one():
    box = (0.1, 0.1, 0.2, 0.2)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)) == 0.0


def test_iou_half_overlap():
    # Two 0.2x0.2 boxes overlapping by half their width.
    a = (0.0, 0.0, 0.2, 0.2)
    b = (0.1, 0.0, 0.2, 0.2)
    inter = 0.1 * 0.2
    union = 0.2 * 0.2 * 2 - inter
    assert iou(a, b) == pytest.approx(inter / union)


# ---------------------------------------------------------------------------
# compute_detection_metrics
# ---------------------------------------------------------------------------

def test_perfect_detector_gets_ap_one():
    class_names = ["a"]
    gts = [
        [{"class_id": 0, "box": (0.1, 0.1, 0.2, 0.2)}],
        [{"class_id": 0, "box": (0.4, 0.4, 0.2, 0.2)}],
    ]
    preds = [
        [{"class_id": 0, "confidence": 0.9, "box": (0.1, 0.1, 0.2, 0.2)}],
        [{"class_id": 0, "confidence": 0.95, "box": (0.4, 0.4, 0.2, 0.2)}],
    ]
    result = compute_detection_metrics(gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.5)
    assert result["per_class"]["a"]["ap"] == pytest.approx(1.0)
    assert result["map50"] == pytest.approx(1.0)
    assert result["counts"] == {"tp": 2, "fp": 0, "fn": 0}


def test_missed_detection_gets_ap_zero():
    class_names = ["a"]
    gts = [[{"class_id": 0, "box": (0.1, 0.1, 0.2, 0.2)}]]
    preds = [[]]  # no predictions at all
    result = compute_detection_metrics(gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.5)
    assert result["per_class"]["a"]["ap"] == pytest.approx(0.0)
    assert result["per_class"]["a"]["recall"] == 0.0
    assert result["counts"] == {"tp": 0, "fp": 0, "fn": 1}


def test_hand_computed_ap_with_one_false_positive():
    """1 class, 2 images: preds ranked [TP(0.9), FP(0.8), TP(0.7)] over 2 GT boxes.

    recall  = [0.5, 0.5, 1.0]
    precision = [1.0, 0.5, 2/3]
    AP (all-point interpolation) = 0.5*1.0 + 0.5*(2/3) = 5/6.
    """
    class_names = ["a"]
    gts = [
        [{"class_id": 0, "box": (0.1, 0.1, 0.2, 0.2)}],
        [{"class_id": 0, "box": (0.6, 0.6, 0.2, 0.2)}],
    ]
    preds = [
        [
            {"class_id": 0, "confidence": 0.9, "box": (0.1, 0.1, 0.2, 0.2)},   # TP
            {"class_id": 0, "confidence": 0.8, "box": (0.9, 0.9, 0.05, 0.05)},  # FP, no GT nearby
        ],
        [
            {"class_id": 0, "confidence": 0.7, "box": (0.6, 0.6, 0.2, 0.2)},   # TP
        ],
    ]
    result = compute_detection_metrics(gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.75)
    assert result["per_class"]["a"]["ap"] == pytest.approx(5 / 6, abs=1e-4)
    assert result["map50"] == pytest.approx(5 / 6, abs=1e-4)
    # At conf_threshold=0.75 only the two 0.9/0.8-confidence detections on
    # image 0 count: one TP, one FP; image 1's 0.7 prediction is excluded.
    assert result["counts"] == {"tp": 1, "fp": 1, "fn": 1}


def test_class_with_no_ground_truth_excluded_from_map():
    class_names = ["a", "b"]
    gts = [[{"class_id": 0, "box": (0.1, 0.1, 0.2, 0.2)}]]
    preds = [[
        {"class_id": 0, "confidence": 0.9, "box": (0.1, 0.1, 0.2, 0.2)},
        {"class_id": 1, "confidence": 0.9, "box": (0.5, 0.5, 0.2, 0.2)},  # class b has no GT anywhere
    ]]
    result = compute_detection_metrics(gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.5)
    assert result["per_class"]["a"]["ap"] == pytest.approx(1.0)
    assert result["per_class"]["b"]["ap"] is None
    assert result["per_class"]["b"]["support"] == 0
    # mAP is the mean over classes WITH ground truth only -> just class a's AP.
    assert result["map50"] == pytest.approx(1.0)


def test_no_ground_truth_anywhere_gives_none_map():
    result = compute_detection_metrics([[]], [[]], ["a"], iou_threshold=0.5, conf_threshold=0.5)
    assert result["map50"] is None
    assert result["per_class"]["a"]["ap"] is None


def test_iou_below_threshold_does_not_match():
    class_names = ["a"]
    gts = [[{"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)}]]
    # Shifted far enough that IoU < 0.5.
    preds = [[{"class_id": 0, "confidence": 0.9, "box": (0.15, 0.0, 0.2, 0.2)}]]
    result = compute_detection_metrics(gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.5)
    assert result["per_class"]["a"]["ap"] == pytest.approx(0.0)
    assert result["counts"] == {"tp": 0, "fp": 1, "fn": 1}


def test_each_gt_box_claimed_at_most_once():
    """Two predictions both overlapping the same single GT box: only the
    higher-confidence one should count as TP, the other as FP."""
    class_names = ["a"]
    gts = [[{"class_id": 0, "box": (0.1, 0.1, 0.2, 0.2)}]]
    preds = [[
        {"class_id": 0, "confidence": 0.9, "box": (0.1, 0.1, 0.2, 0.2)},
        {"class_id": 0, "confidence": 0.8, "box": (0.1, 0.1, 0.2, 0.2)},
    ]]
    result = compute_detection_metrics(gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.5)
    assert result["counts"] == {"tp": 1, "fp": 1, "fn": 0}


# ---------------------------------------------------------------------------
# bucket_samples
# ---------------------------------------------------------------------------

def test_bucket_samples_classifies_tp_fp_fn_confusion():
    class_names = ["a", "b"]
    paths = ["img0.jpg"]
    gts = [[
        {"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)},   # will be matched correctly -> tp
        {"class_id": 1, "box": (0.5, 0.5, 0.2, 0.2)},   # will be matched with wrong class -> confusion
        {"class_id": 0, "box": (0.8, 0.8, 0.1, 0.1)},   # unmatched -> fn
    ]]
    preds = [[
        {"class_id": 0, "confidence": 0.9, "box": (0.0, 0.0, 0.2, 0.2)},
        {"class_id": 0, "confidence": 0.8, "box": (0.5, 0.5, 0.2, 0.2)},  # spatially matches class-1 box
        {"class_id": 1, "confidence": 0.7, "box": (0.3, 0.9, 0.05, 0.05)},  # no nearby GT -> fp
    ]]
    samples = bucket_samples(paths, gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.5)
    buckets = sorted(s["bucket"] for s in samples)
    assert buckets == ["confusion", "fn", "fp", "tp"]


def test_bucket_samples_caps_per_bucket():
    class_names = ["a"]
    n = 30
    paths = [f"img{i}.jpg" for i in range(n)]
    gts = [[] for _ in range(n)]  # no GT anywhere -> every prediction is FP
    preds = [
        [{"class_id": 0, "confidence": 0.9, "box": (0.1, 0.1, 0.1, 0.1)}]
        for _ in range(n)
    ]
    samples = bucket_samples(paths, gts, preds, class_names, iou_threshold=0.5, conf_threshold=0.5,
                              max_per_bucket=5)
    assert len([s for s in samples if s["bucket"] == "fp"]) == 5
