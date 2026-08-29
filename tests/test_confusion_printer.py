"""The shared ANSI confusion-matrix printer — both layouts render without raising."""
import numpy as np

from cvbench.core._confusion import print_confusion_matrix


def test_normal_layout_renders(capsys):
    cm = np.array([[5, 1], [2, 4]])
    print_confusion_matrix(cm, ["a", "b"], title="Test matrix:")
    out = capsys.readouterr().out
    assert "Test matrix:" in out
    assert "a" in out and "b" in out


def test_staircase_layout_renders_for_many_long_labels(capsys):
    names = [f"very_long_class_name_{i}" for i in range(12)]
    cm = np.eye(12, dtype=int) * 3
    print_confusion_matrix(cm, names)
    out = capsys.readouterr().out
    assert names[0] in out
    assert names[-1] in out


def test_background_label_prints_like_any_other(capsys):
    cm = np.array([[3, 0, 1], [1, 2, 0], [0, 1, 0]])
    print_confusion_matrix(cm, ["a", "b", "background"], title="Detection confusion:")
    out = capsys.readouterr().out
    assert "background" in out
