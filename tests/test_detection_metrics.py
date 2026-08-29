"""Hand-computed tests for cvbench.detection.metrics — pure NumPy, no TensorFlow."""
import pytest

from cvbench.detection.metrics import (
    build_detection_samples,
    compute_detection_metrics,
    detection_confusion,
    iou,
    localization_metrics,
    match_detections,
)


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
# localization view
# ---------------------------------------------------------------------------

def test_localization_mean_iou_is_average_over_matched_pairs():
    class_names = ["a"]
    gts = [
        [{"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)}],
        [{"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)}],
    ]
    preds = [
        [{"class_id": 0, "confidence": 0.9, "box": (0.0, 0.0, 0.2, 0.2)}],   # IoU 1.0
        [{"class_id": 0, "confidence": 0.9, "box": (0.1, 0.0, 0.2, 0.2)}],   # IoU 1/3
    ]
    result = compute_detection_metrics(gts, preds, class_names, iou_threshold=0.25, conf_threshold=0.5)
    assert result["localization"]["mean_iou"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_localization_recall_sweep():
    class_names = ["a"]
    # Two 0.2x0.2 boxes overlapping by half -> IoU = 1/3 ≈ 0.333.
    gts = [[{"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)}]]
    preds = [[{"class_id": 0, "confidence": 0.9, "box": (0.1, 0.0, 0.2, 0.2)}]]
    result = compute_detection_metrics(gts, preds, class_names, iou_threshold=0.25, conf_threshold=0.5)
    sweep = result["localization"]["recall_sweep"]
    assert sweep["0.5"] == 0.0
    assert sweep["0.75"] == 0.0
    assert sweep["0.9"] == 0.0

    # A near-perfect box clears every threshold.
    preds2 = [[{"class_id": 0, "confidence": 0.9, "box": (0.0, 0.0, 0.2, 0.2)}]]
    sweep2 = compute_detection_metrics(gts, preds2, class_names, conf_threshold=0.5)["localization"]["recall_sweep"]
    assert sweep2 == {"0.5": 1.0, "0.75": 1.0, "0.9": 1.0}


def test_localization_reports_ap50_and_ap75():
    class_names = ["a"]
    gts = [[{"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)}]]
    preds = [[{"class_id": 0, "confidence": 0.9, "box": (0.0, 0.0, 0.2, 0.2)}]]
    loc = compute_detection_metrics(gts, preds, class_names, conf_threshold=0.5)["localization"]
    assert loc["ap50"] == pytest.approx(1.0)
    assert loc["ap75"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# classification view — confusion matrix
# ---------------------------------------------------------------------------

def test_detection_confusion_matrix_has_background_row_and_column():
    class_names = ["a", "b"]
    paths = ["img0.jpg"]
    gts = [[
        {"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)},   # matched, correct class
        {"class_id": 1, "box": (0.5, 0.5, 0.2, 0.2)},   # matched, wrong class -> a->b confusion
        {"class_id": 0, "box": (0.8, 0.8, 0.1, 0.1)},   # missed -> background column
    ]]
    preds = [[
        {"class_id": 0, "confidence": 0.9, "box": (0.0, 0.0, 0.2, 0.2)},
        {"class_id": 0, "confidence": 0.8, "box": (0.5, 0.5, 0.2, 0.2)},   # predicts a on the b box
        {"class_id": 1, "confidence": 0.7, "box": (0.3, 0.95, 0.03, 0.03)},  # spurious -> background row
    ]]
    matches = match_detections(paths, gts, preds, class_names, iou_floor=0.5, conf_threshold=0.5)
    cm = detection_confusion(matches, class_names)
    assert cm["classes"] == ["a", "b", "background"]
    # rows = true, cols = predicted; index 2 = background
    assert cm["matrix"] == [
        [1, 0, 1],   # true a: 1 correct, 1 missed
        [1, 0, 0],   # true b: predicted as a (class confusion)
        [0, 1, 0],   # background: 1 spurious "b" prediction
    ]
    # row-normalized
    assert cm["matrix_normalized"][0] == [0.5, 0.0, 0.5]

    result = compute_detection_metrics(gts, preds, class_names, matches, conf_threshold=0.5)
    assert result["counts"] == {"tp": 2, "fp": 1, "fn": 1}
    # class a: tp=1, fp (col a) = 1 (the b box), fn (row a) = 1 (missed) -> P=R=0.5
    assert result["per_class"]["a"]["precision"] == pytest.approx(0.5)
    assert result["per_class"]["a"]["recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# match_detections + build_detection_samples
# ---------------------------------------------------------------------------

def _matches(paths, gts, preds, class_names, **kw):
    kw.setdefault("iou_floor", 0.5)
    kw.setdefault("conf_threshold", 0.5)
    return match_detections(paths, gts, preds, class_names, **kw)


def test_match_detections_records_iou_and_counterpart_indices():
    class_names = ["a"]
    paths = ["img0.jpg"]
    gts = [[{"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)}]]
    preds = [[{"class_id": 0, "confidence": 0.9, "box": (0.0, 0.0, 0.2, 0.2)}]]
    m = _matches(paths, gts, preds, class_names)[0]
    assert m["pred"][0]["matched_gt"] == 0
    assert m["pred"][0]["iou"] == pytest.approx(1.0)
    assert m["gt"][0]["matched_pred"] == 0


def test_build_detection_samples_uses_matched_background_tags_and_cells():
    class_names = ["a", "b"]
    paths = ["img0.jpg"]
    gts = [[
        {"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)},   # matched correctly
        {"class_id": 1, "box": (0.5, 0.5, 0.2, 0.2)},   # matched, wrong class
        {"class_id": 0, "box": (0.8, 0.8, 0.1, 0.1)},   # missed
    ]]
    preds = [[
        {"class_id": 0, "confidence": 0.9, "box": (0.0, 0.0, 0.2, 0.2)},
        {"class_id": 0, "confidence": 0.8, "box": (0.5, 0.5, 0.2, 0.2)},
        {"class_id": 1, "confidence": 0.7, "box": (0.3, 0.95, 0.03, 0.03)},  # spurious
    ]]
    samples = build_detection_samples(_matches(paths, gts, preds, class_names), class_names)
    assert len(samples) == 1
    s = samples[0]
    assert s["path"] == "img0.jpg"
    assert s["counts"] == {"matched": 2, "fp": 1, "fn": 1}
    assert [b["match"] for b in s["gt"]] == ["matched", "matched", "background"]
    assert [b["match"] for b in s["pred"]] == ["matched", "matched", "background"]
    assert sorted(s["cells"]) == [["a", "a"], ["a", "background"], ["b", "a"], ["background", "b"]]
    assert all("confidence" in b for b in s["pred"])
    assert not any("confidence" in b for b in s["gt"])
    assert s["pred"][0]["iou"] == pytest.approx(1.0)


def test_build_detection_samples_caps_per_cell_by_distinct_images():
    class_names = ["a"]
    n = 30
    paths = [f"img{i}.jpg" for i in range(n)]
    gts = [[] for _ in range(n)]  # no GT anywhere -> every prediction is spurious
    preds = [[{"class_id": 0, "confidence": 0.9, "box": (0.1, 0.1, 0.1, 0.1)}] for _ in range(n)]
    samples = build_detection_samples(_matches(paths, gts, preds, class_names), class_names, max_per_cell=5)
    fp_samples = [s for s in samples if ["background", "a"] in s["cells"]]
    assert len(fp_samples) == 5
    assert len({s["path"] for s in fp_samples}) == 5


def test_build_detection_samples_skips_empty_images():
    class_names = ["a"]
    paths = ["empty.jpg", "img1.jpg"]
    gts = [[], [{"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)}]]
    preds = [[], []]
    samples = build_detection_samples(_matches(paths, gts, preds, class_names), class_names)
    assert [s["path"] for s in samples] == ["img1.jpg"]


def test_build_detection_samples_confidence_filtering():
    class_names = ["a"]
    paths = ["img0.jpg"]
    gts = [[{"class_id": 0, "box": (0.0, 0.0, 0.2, 0.2)}]]
    preds = [[
        {"class_id": 0, "confidence": 0.9, "box": (0.0, 0.0, 0.2, 0.2)},
        {"class_id": 0, "confidence": 0.1, "box": (0.7, 0.7, 0.1, 0.1)},  # below threshold
    ]]
    samples = build_detection_samples(_matches(paths, gts, preds, class_names), class_names)
    assert len(samples[0]["pred"]) == 1
    assert samples[0]["counts"] == {"matched": 1, "fp": 0, "fn": 0}
