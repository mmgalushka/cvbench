"""Terminal formatting helpers for CVBench CLI output.

Respects NO_COLOR env var and non-tty stdout for piped output / accessibility.
"""
from __future__ import annotations

import os
import shutil
import sys


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def term_width(fallback: int = 80) -> int:
    return shutil.get_terminal_size((fallback, 24)).columns


def _c(code: str, text: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _c("1", text)


def dim(text: str) -> str:
    return _c("2", text)


def green(text: str) -> str:
    return _c("92", text)


def yellow(text: str) -> str:
    return _c("93", text)


def blue(text: str) -> str:
    return _c("94", text)


def rule(width: int | None = None, color: str = "dim") -> str:
    """Separator line.

    width  — explicit width; defaults to full terminal width.
    color  — 'dim' (gray) or 'white' (bright white).
    """
    w = width if width is not None else term_width()
    line = "─" * w
    if not _color_enabled():
        return line
    if color == "white":
        return f"\033[97m{line}\033[0m"
    return f"\033[2m{line}\033[0m"
