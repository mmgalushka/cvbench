"""Shared rich-based console for CVBench CLI output.

Every CLI module should route colorized/formatted output through this module
instead of hand-rolling ANSI escapes. It replaces the legacy ``_fmt`` helper
(now deleted) with the same NO_COLOR / non-TTY gating, implemented once on top
of `rich <https://rich.readthedocs.io/>`_ instead of raw escape codes.

Colour is disabled (falls back to plain text) whenever the ``NO_COLOR`` env
var is set, or stdout is not a TTY (e.g. output is piped) — matching prior
behaviour so piped/redirected output and accessibility tooling keep working.

Note: :mod:`cvbench.cli.overview`'s markdown-generating functions
(``render_markdown``, ``render_quickstart_markdown``) are deliberately left
untouched — they produce Markdown *source* text for files/docs, not a
terminal rendering, and rich has no facility to emit Markdown source (only to
render existing Markdown for display).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from rich import box
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _make_console(
    *,
    color_system: Literal["auto", "standard", "256", "truecolor", "windows"] = "standard",
    **kwargs: Any,
) -> Console:
    """A fresh Console reflecting the *current* NO_COLOR/TTY state.

    Built fresh per call (cheap) rather than cached at import time, so
    changes to the environment or stdout (as tests do) take effect
    immediately — matching the old ``_fmt`` module's per-call checks.

    ``color_system`` defaults to "standard" — the 16-colour ANSI codes the
    legacy ``_fmt`` helper used, for byte-identical output on terminals that
    already looked right. Callers needing the wider 256-colour ramp (e.g. the
    confusion-matrix heatmap) pass ``color_system="256"``.
    """
    enabled = _color_enabled()
    return Console(
        force_terminal=enabled or None,
        no_color=not enabled,
        color_system=color_system,
        highlight=False,
        **kwargs,
    )


def _style_text(text: str, style: str) -> str:
    if not _color_enabled():
        # rich's `no_color` only strips colour, not attributes like bold/dim —
        # we want *no* escape codes at all under NO_COLOR/non-TTY, matching
        # the legacy `_fmt` helper.
        return text
    console = _make_console()
    with console.capture() as cap:
        console.print(text, style=style, end="")
    return cap.get()


def term_width(fallback: int = 80) -> int:
    return shutil.get_terminal_size((fallback, 24)).columns


def bold(text: str) -> str:
    return _style_text(text, "bold")


def dim(text: str) -> str:
    return _style_text(text, "dim")


def green(text: str) -> str:
    return _style_text(text, "bright_green")


def yellow(text: str) -> str:
    return _style_text(text, "bright_yellow")


def blue(text: str) -> str:
    return _style_text(text, "bright_blue")


def rule(width: int | None = None, color: str = "dim", thick: bool = False) -> str:
    """Separator line.

    width  — explicit width; defaults to full terminal width.
    color  — 'dim' (gray) or 'white' (bright white).
    thick  — use heavy box-drawing character ━ instead of ─.
    """
    w = width if width is not None else term_width()
    line = ("━" if thick else "─") * w
    style = "bright_white" if color == "white" else "dim"
    return _style_text(line, style)


def banner(line: str, *, width: int | None = None, thick: bool = False) -> None:
    """Print the common 3-line ``rule / content / rule`` header used across the CLI.

    ``line`` is the already-formatted content (typically built with `bold`/
    `dim`) — banner only wraps it, it doesn't add its own styling.
    """
    r = rule(width, "white", thick=thick)
    print(r)
    print(line if line.startswith(" ") else f" {line}")
    print(r)


def success(msg: str) -> None:
    print(f" {green('✅')} {msg}")


def warning(msg: str) -> None:
    print(f" {yellow('⚠️')}  {msg}")


def error(msg: str) -> None:
    print(f" {_style_text('❌', 'bold red')} {msg}")


def info(msg: str) -> None:
    print(f" {dim(msg)}")


def table(
    columns: Sequence[str | tuple[str, Literal["left", "center", "right"]]],
    rows: Iterable[Iterable[Any]],
    *,
    title: str | None = None,
) -> None:
    """Print a simple table via `rich.table.Table`.

    ``columns`` is a list of header strings, or ``(header, justify)`` pairs
    (justify one of "left"/"center"/"right", default "left"). No box borders —
    just a header, a rule under it, and column spacing — to keep tables
    compact instead of spending width on a full grid.
    """
    header_style = "bold" if _color_enabled() else None
    t = Table(title=title, header_style=header_style, box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    for col in columns:
        header, justify = col if isinstance(col, tuple) else (col, "left")
        t.add_column(header, justify=justify)
    for row in rows:
        t.add_row(*(str(v) for v in row))
    _make_console().print(t)


def syntax(code: str, lexer: str = "text", *, theme: str = "ansi_dark") -> None:
    """Print syntax-highlighted code, falling back to plain text under NO_COLOR/non-TTY."""
    if not _color_enabled():
        print(code, end="")
        return
    _make_console().print(Syntax(code, lexer, theme=theme, background_color="default", word_wrap=True))


def render(renderable: Any) -> str:
    """Capture any rich renderable to a plain/ANSI string, honoring NO_COLOR/TTY.

    For call sites that need a string rather than a direct print — e.g.
    Click's `HelpFormatter`, which buffers text itself.
    """
    console = _make_console()
    with console.capture() as cap:
        console.print(renderable, end="")
    out = cap.get()
    if not _color_enabled():
        # `no_color` strips colour but keeps attributes like bold/dim; scrub
        # any that slipped through so a disabled console is truly plain.
        out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    return out
