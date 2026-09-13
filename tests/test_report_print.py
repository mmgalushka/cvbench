"""The shared eval_report.json terminal renderer — no TF/Keras dependency.

Mirrors test_confusion_printer.py's style: build a synthetic report dict
(the same shape classification/evaluator.py and detection/evaluator.py write)
and assert the printed output surfaces the right numbers, without needing a
real model or dataset.
"""
from cvbench.core.report_print import print_classification_body, print_detection_body


def _classification_report(top3=0.98):
    return {
        "n_images": 100,
        "overall_accuracy": 0.9123,
        "top3_accuracy": top3,
        "per_class": {
            "cat": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 50},
            "dog": {"precision": 0.85, "recall": 0.9, "f1": 0.87, "support": 50},
        },
        "confusion_matrix": {"classes": ["cat", "dog"], "matrix": [[40, 10], [5, 45]]},
    }


def test_classification_body_prints_accuracy_and_top3(capsys):
    print_classification_body(_classification_report())
    out = capsys.readouterr().out
    assert "91.2%" in out
    assert "Top-3 accuracy" in out and "98.0%" in out
    assert "cat" in out and "dog" in out
    assert "Confusion matrix" in out


def test_classification_body_omits_top3_when_absent(capsys):
    print_classification_body(_classification_report(top3=None))
    out = capsys.readouterr().out
    assert "Top-3 accuracy" not in out


def _detection_report():
    return {
        "n_images": 20,
        "detection": {
            "map50": 0.75,
            "conf_threshold": 0.25,
            "iou_threshold": 0.5,
            "counts": {"tp": 30, "fp": 5, "fn": 3},
            "localization": {
                "mean_iou": 0.82, "ap50": 0.75, "ap75": 0.6,
                "recall_sweep": {"0.5": 0.9, "0.75": 0.7, "0.9": 0.4},
            },
            "class_breakdown": {
                "classes": ["cat", "dog"],
                "rows": {
                    "cat": {"instances": 15, "matched": 12, "mislocated": 1,
                            "confused": 1, "missed": 1, "confused_as": {"dog": 1}},
                    "dog": {"instances": 18, "matched": 16, "mislocated": 0,
                            "confused": 0, "missed": 2, "confused_as": {}},
                },
                "duplicate": {"cat": 1, "dog": 0},
                "duplicate_suppressible": {"cat": 1, "dog": 0},
                "spurious": {"cat": 0, "dog": 1},
                "dup_suppress_iou": 0.5,
            },
            "confusion_matrix": {
                "classes": ["cat", "dog", "background"],
                "matrix": [[12, 1, 2], [0, 16, 2], [1, 1, 0]],
            },
        },
        "per_class": {
            "cat": {"ap": 0.7, "ap75": 0.55, "precision": 0.85, "recall": 0.8, "f1": 0.82, "support": 15},
            "dog": {"ap": 0.8, "ap75": 0.65, "precision": 0.9, "recall": 0.88, "f1": 0.89, "support": 18},
        },
    }


def test_detection_body_prints_localization_and_per_class_ap(capsys):
    print_detection_body(_detection_report())
    out = capsys.readouterr().out
    assert "Localization" in out
    assert "82.0%" in out  # mean IoU
    assert "75.0% / 60.0%" in out  # AP@50 / AP@75
    assert "AP@50" in out and "AP@75" in out
    assert "0.5500" in out  # cat's per-class AP@75
    assert "Per-class outcomes" in out
    assert "→dog 1" in out  # cat's confused_as note
    assert "Class-confusion matrix" in out
    assert "TP 30" in out and "FP 5" in out and "FN 3" in out


def test_detection_body_handles_missing_ap75(capsys):
    report = _detection_report()
    report["per_class"]["cat"]["ap75"] = None
    print_detection_body(report)
    out = capsys.readouterr().out
    assert "n/a" in out


def test_detection_body_skips_outcomes_table_without_breakdown(capsys):
    report = _detection_report()
    del report["detection"]["class_breakdown"]
    print_detection_body(report)
    out = capsys.readouterr().out
    assert "Per-class outcomes" not in out
    assert "Detection quality" in out
