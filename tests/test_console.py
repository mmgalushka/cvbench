"""cvbench.core.console — NO_COLOR / non-TTY gating for every helper.

Under pytest, capsys already replaces stdout with a non-TTY stream, so the
"disabled" cases exercise the default; the "enabled" cases force a TTY via
monkeypatch to check colour actually turns on when it should.
"""
import sys

import pytest

from cvbench.core import console

ESC = "\x1b["


def _force_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)


@pytest.mark.parametrize("helper", [console.bold, console.dim, console.green, console.yellow, console.blue])
def test_style_helpers_plain_when_not_a_tty(helper):
    assert helper("hi") == "hi"


@pytest.mark.parametrize("helper", [console.bold, console.dim, console.green, console.yellow, console.blue])
def test_style_helpers_plain_under_no_color(monkeypatch, helper):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert helper("hi") == "hi"


@pytest.mark.parametrize("helper", [console.bold, console.dim, console.green, console.yellow, console.blue])
def test_style_helpers_colorize_on_a_real_tty(monkeypatch, helper):
    _force_tty(monkeypatch)
    out = helper("hi")
    assert out != "hi"
    assert out.startswith(ESC)
    assert "hi" in out


def test_rule_plain_when_not_a_tty():
    r = console.rule(10)
    assert r == "─" * 10
    assert ESC not in r


def test_rule_colorized_on_a_real_tty(monkeypatch):
    _force_tty(monkeypatch)
    r = console.rule(10, "white", thick=True)
    assert "━" * 10 in r
    assert ESC in r


def test_banner_prints_rule_content_rule(capsys):
    console.banner("hello")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == out[2]
    assert out[1] == " hello"


def test_success_warning_error_glyphs(capsys):
    console.success("ok")
    console.warning("careful")
    console.error("boom")
    console.info("fyi")
    out = capsys.readouterr().out
    assert "✅ ok" in out
    assert "⚠️  careful" in out
    assert "❌ boom" in out
    assert "fyi" in out
    assert ESC not in out  # not a tty here


def test_table_prints_headers_and_rows_without_ansi_when_disabled(capsys):
    console.table(["Name", ("Count", "right")], [("a", 1), ("b", 2)])
    out = capsys.readouterr().out
    assert "Name" in out and "Count" in out
    assert "a" in out and "1" in out
    assert ESC not in out


def test_syntax_falls_back_to_plain_text_when_not_a_tty(capsys):
    console.syntax("x: 1", "yaml")
    out = capsys.readouterr().out
    assert out == "x: 1"


def test_render_strips_any_ansi_when_disabled():
    from rich.text import Text

    out = console.render(Text("hi", style="bold red"))
    assert out == "hi"


def test_render_keeps_ansi_on_a_real_tty(monkeypatch):
    from rich.text import Text

    _force_tty(monkeypatch)
    out = console.render(Text("hi", style="bold red"))
    assert ESC in out
    assert "hi" in out


def test_term_width_falls_back_when_no_terminal(monkeypatch):
    import os
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback: os.terminal_size(fallback))
    assert console.term_width(42) == 42


def test_progress_is_silent_and_callable_when_not_a_tty(capsys):
    with console.progress(5, "Working") as advance:
        for _ in range(5):
            advance(1)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_progress_is_a_noop_for_an_empty_total(monkeypatch, capsys):
    _force_tty(monkeypatch)
    with console.progress(0, "Working") as advance:
        advance(1)
    assert capsys.readouterr().out == ""


def test_progress_draws_a_bar_on_a_real_tty(monkeypatch, capsys):
    _force_tty(monkeypatch)
    with console.progress(3, "Working") as advance:
        for _ in range(3):
            advance(1)
    out = capsys.readouterr().out
    assert "Working" in out
    assert "3/3" in out


def test_progress_under_no_color_has_no_colour_codes(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with console.progress(2, "Working") as advance:
        advance(2)
    out = capsys.readouterr().out
    assert "2/2" in out
    assert not any(f"{ESC}{c}" in out for c in ("3", "9", "38;", "48;"))


def test_quiet_native_stderr_swallows_output_on_success(capfd):
    import os

    with console.quiet_native_stderr():
        os.write(2, b"noise\n")
    assert "noise" not in capfd.readouterr().err


def test_quiet_native_stderr_replays_output_on_failure(capfd):
    import os

    with pytest.raises(RuntimeError), console.quiet_native_stderr():
        os.write(2, b"converter detail\n")
        raise RuntimeError("boom")
    assert "converter detail" in capfd.readouterr().err
