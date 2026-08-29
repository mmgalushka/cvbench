"""Shared terminal confusion-matrix printer.

Used by both the classification and detection evaluators — detection's matrix
just carries an extra trailing ``background`` label, which prints like any
other row/column.
"""
from __future__ import annotations

import shutil

import numpy as np

# ANSI 256-colour blue ramp: white -> pure blue (no cyan tint)
# 231=#ffffff  189=#d7d7ff  147=#afafff  105=#8787ff  63=#5f5fff  21=#0000ff
_BLUE_RAMP = [231, 189, 147, 105, 63, 21]
_RESET = "\033[0m"


def print_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    *,
    title: str = "Confusion matrix (rows = true, cols = predicted):",
) -> None:
    """Print a colour-coded confusion matrix to the terminal using ANSI 256 colours.

    Uses the normal layout (horizontal column headers) when the matrix fits the
    terminal width, otherwise falls back to the staircase layout where column
    labels are right-aligned and connected by L-shaped pseudo-graphic lines.
    """
    cm = np.asarray(cm)
    n = len(class_names)
    label_w = max(len(cls) for cls in class_names)
    max_val = int(cm.max()) if cm.max() > 0 else 1
    term_w = shutil.get_terminal_size((80, 24)).columns

    def _fmt_cell(val: int, is_diag: bool, cell_w: int) -> str:
        idx = min(int(val / max_val * (len(_BLUE_RAMP) - 1)), len(_BLUE_RAMP) - 1)
        bg = f"\033[48;5;{_BLUE_RAMP[idx]}m"
        fg = "\033[30m" if idx < 3 else "\033[97m"
        bold = "\033[1m" if is_diag else ""
        return f"{bg}{fg}{bold}{val:^{cell_w}}{_RESET}"

    # --- measure whether normal layout fits ---
    num_w = len(str(max_val)) + 2
    normal_cell_w = max(num_w, label_w)
    row_prefix_w = 3 + label_w + 3
    grid_w = n * normal_cell_w + (n - 1)
    normal_fits = (row_prefix_w + grid_w) <= term_w

    print(f" {title}")

    if normal_fits:
        # ── normal layout ──────────────────────────────────────────────
        pad = " " * (label_w + 3)
        col_header = " ".join(f"{cls:^{normal_cell_w}}" for cls in class_names)
        print(f"   {pad}{col_header}")
        for i, true_cls in enumerate(class_names):
            cells = " ".join(_fmt_cell(int(cm[i, j]), i == j, normal_cell_w) for j in range(n))
            print(f"   {true_cls:<{label_w}} | {cells}")
    else:
        # ── staircase layout ───────────────────────────────────────────
        cell_w = num_w

        col_offsets = [row_prefix_w + j * (cell_w + 1) + cell_w // 2
                       for j in range(n)]

        align_col = col_offsets[-1] + 3

        for i, cls in enumerate(class_names):
            line = " " * col_offsets[0]
            for j in range(i):
                gap = col_offsets[j + 1] - col_offsets[j] - 1
                line += "│" + " " * gap
            n_dashes = align_col - col_offsets[i] - 2
            line += "┌" + "─" * n_dashes + " " + cls
            print(line)

        vert_row = ""
        for j in range(n):
            vert_row = vert_row.ljust(col_offsets[j]) + "│"
        print(vert_row)

        for i, true_cls in enumerate(class_names):
            cells = " ".join(_fmt_cell(int(cm[i, j]), i == j, cell_w) for j in range(n))
            print(f"   {true_cls:<{label_w}} | {cells}")

    print()
