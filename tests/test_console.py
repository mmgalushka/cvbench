"""cvbench.core._console — NO_COLOR / non-TTY gating for every helper.

Under pytest, capsys already replaces stdout with a non-TTY stream, so the
"disabled" cases exercise the default; the "enabled" cases force a TTY via
monkeypatch to check colour actually turns on when it should.
"""
import sys

import pytest

from cvbench.core import _console

ESC = "\x1b["


def _force_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


@pytest.mark.parametrize("helper", [_console.bold, _console.dim, _console.green, _console.yellow, _console.blue])
def test_style_helpers_plain_when_not_a_tty(helper):
    assert helper("hi") == "hi"


@pytest.mark.parametrize("helper", [_console.bold, _console.dim, _console.green, _console.yellow, _console.blue])
def test_style_helpers_plain_under_no_color(monkeypatch, helper):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert helper("hi") == "hi"


@pytest.mark.parametrize("helper", [_console.bold, _console.dim, _console.green, _console.yellow, _console.blue])
def test_style_helpers_colorize_on_a_real_tty(monkeypatch, helper):
    _force_tty(monkeypatch)
    out = helper("hi")
    assert out != "hi"
    assert out.startswith(ESC)
    assert "hi" in out


def test_rule_plain_when_not_a_tty():
    r = _console.rule(10)
    assert r == "─" * 10
    assert ESC not in r


def test_rule_colorized_on_a_real_tty(monkeypatch):
    _force_tty(monkeypatch)
    r = _console.rule(10, "white", thick=True)
    assert "━" * 10 in r
    assert ESC in r


def test_banner_prints_rule_content_rule(capsys):
    _console.banner("hello")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == out[2]
    assert out[1] == " hello"


def test_success_warning_error_glyphs(capsys):
    _console.success("ok")
    _console.warning("careful")
    _console.error("boom")
    _console.info("fyi")
    out = capsys.readouterr().out
    assert "✅ ok" in out
    assert "⚠️  careful" in out
    assert "❌ boom" in out
    assert "fyi" in out
    assert ESC not in out  # not a tty here


def test_table_prints_headers_and_rows_without_ansi_when_disabled(capsys):
    _console.table(["Name", ("Count", "right")], [("a", 1), ("b", 2)])
    out = capsys.readouterr().out
    assert "Name" in out and "Count" in out
    assert "a" in out and "1" in out
    assert ESC not in out


def test_syntax_falls_back_to_plain_text_when_not_a_tty(capsys):
    _console.syntax("x: 1", "yaml")
    out = capsys.readouterr().out
    assert out == "x: 1"


def test_render_strips_any_ansi_when_disabled():
    from rich.text import Text

    out = _console.render(Text("hi", style="bold red"))
    assert out == "hi"


def test_render_keeps_ansi_on_a_real_tty(monkeypatch):
    from rich.text import Text

    _force_tty(monkeypatch)
    out = _console.render(Text("hi", style="bold red"))
    assert ESC in out
    assert "hi" in out


def test_term_width_falls_back_when_no_terminal(monkeypatch):
    import os
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size(fallback))
    assert _console.term_width(42) == 42
