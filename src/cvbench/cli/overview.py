"""``commands`` — the single signpost for the whole CVBench container.

Everything a newcomer at the container prompt can do: the CVBench CLI commands
(enumerated live from the Click registry — never hand-listed, so this can't rot
the way the old bash function did), plus the container conveniences that are not
Click commands (tmux sessions, TensorBoard, JupyterLab).

The same data renders three ways:

* :func:`render`                    — the terminal screen shown by ``commands``
* :func:`render_markdown`           — the ``## CLI reference`` block in README.md
* :func:`render_quickstart_markdown` — the ``## Quickstart`` block in README.md
                                       (both regenerated together by ``./helper.sh docs``)

Must stay TensorFlow-free: importing every CLI module here is what makes
``commands`` feel instant. ``tests/test_import_boundaries.py`` locks that in.
"""
from __future__ import annotations

import importlib

import click

from cvbench.cli import _help
from cvbench.core import _fmt

# name → "module:attribute" — the console scripts from pyproject [project.scripts],
# minus `commands` itself. This is a list of *where the commands are*, not a copy
# of what they do; summaries, options and examples all come from the Click objects.
ENTRY_POINTS: tuple[tuple[str, str], ...] = (
    ("train", "cvbench.cli.train:train"),
    ("evaluate", "cvbench.cli.evaluate:evaluate"),
    ("predict", "cvbench.cli.predict:predict"),
    ("serve", "cvbench.cli.serve:serve"),
    ("data", "cvbench.cli.data:data"),
    ("runs", "cvbench.cli.runs:runs"),
    ("augmentations", "cvbench.cli.augmentations:augmentations"),
)

# The happy path, top to bottom — rendered into the README's Quickstart section
# (never the terminal screen; `commands` is command reference only). Placeholders
# in <angle brackets> are not resolved against the registry by the tests.
QUICKSTART: tuple[tuple[str, str], ...] = (
    ("commands", "show this screen again any time"),
    ("tm -n <name>", "start a tmux session so training survives closing your terminal"),
    ("data generate", "make a 4-class synthetic dataset in data/synthetic/"),
    ("train data/synthetic --epochs 5", "train a model — prints the run name when it finishes"),
    ("runs list", "see every run, newest first"),
    ("evaluate <run-name>", "score that run on the held-out test split"),
    ("serve --host 0.0.0.0 --port 8000", "browse it all in the WebUI → http://<server-ip>:8000"),
)

# Container conveniences that are NOT Click commands. Declared once, here, and
# cross-checked by tests (tm flags vs scripts/bashrc). TensorBoard and JupyterLab
# are deliberately omitted — the container provides those as services, they are
# not something the user starts by hand.
EXTRAS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Sessions (tmux) — keep training alive after you close the terminal", (
        ("tm -n <name>", "new session"),
        ("tm -c <name>", "connect / attach"),
        ("tm -d <name>", "delete session"),
        ("tm -l", "list sessions"),
        ("Ctrl+B then D", "detach from the session you are in"),
    )),
)

_TOP_LEVEL = {"train", "evaluate", "predict", "serve"}


def load_command(spec: str) -> click.Command:
    """Resolve a ``"module:attribute"`` entry-point spec to its Click command."""
    module_name, attr = spec.split(":")
    return getattr(importlib.import_module(module_name), attr)


def _walk(cmd: click.Command, path: str):
    yield path, cmd
    if isinstance(cmd, click.Group):
        ctx = click.Context(cmd, info_name=path)
        for name in sorted(cmd.list_commands(ctx)):
            sub = cmd.get_command(ctx, name)
            if sub is not None and not sub.hidden:
                yield from _walk(sub, f"{path} {name}")


def iter_commands():
    """Yield ``(path, command)`` for every command and subcommand, depth-first.

    ``path`` is what the user types, e.g. ``"data"`` then ``"data split"``.
    """
    for name, spec in ENTRY_POINTS:
        yield from _walk(load_command(spec), name)


def _short_help(cmd: click.Command) -> str:
    return cmd.get_short_help_str(120)


def _grouped():
    """Return (top_level, groups) where groups is [(group_name, group, [(sub_path, cmd), ...])]."""
    top: list[tuple[str, click.Command]] = []
    groups: list[tuple[str, click.Group, list[tuple[str, click.Command]]]] = []
    for name, spec in ENTRY_POINTS:
        cmd = load_command(spec)
        if isinstance(cmd, click.Group):
            subs = [(p, c) for p, c in _walk(cmd, name) if p != name]
            groups.append((name, cmd, subs))
        else:
            top.append((name, cmd))
    return top, groups


def render() -> str:
    w = min(_fmt.term_width(80), 90)
    rule = _fmt.rule(w)
    thick = _fmt.rule(w, "white")
    top, groups = _grouped()

    cmd_col = max(len(p) for p, _ in iter_commands()) + 3
    # Short shell aliases share a column; long one-liners (tensorboard, jupyter)
    # exceed it and wrap their description onto the next line.
    x_col = max((len(c) for _, rows in EXTRAS for c, _ in rows if len(c) <= 20), default=16) + 2

    def dl_row(indent, term, desc, col, colour=_fmt.green):
        pad = " " * indent
        if len(term) <= col:
            return f"{pad}{colour(f'{term:<{col}}')}{_fmt.dim(desc)}"
        return f"{pad}{colour(term)}\n{pad}{' ' * col}{_fmt.dim(desc)}"

    out: list[str] = [thick]
    out.append(f" {_fmt.bold('CVBench')} {_fmt.dim('—')} Computer Vision Training Sandbox")
    out.append(thick)
    out.append("")
    out.append(f" {_fmt.bold('CVBench commands')}")
    out.append(f"   {_fmt.dim('Every command explains itself:')}  train --help    data split --help")
    out.append("")
    for name, cmd in top:
        out.append(dl_row(3, name, _short_help(cmd), cmd_col, colour=_fmt.green))
    for name, grp, subs in groups:
        out.append("")
        out.append(dl_row(3, name, _short_help(grp), cmd_col))
        for sub_path, sub in subs:
            out.append(dl_row(5, sub_path, _short_help(sub), cmd_col - 2))
    out.append("")
    out.append(rule)
    for i, (title, rows) in enumerate(EXTRAS):
        if i:
            out.append("")
        out.append(f" {_fmt.bold(title)}")
        out.append("")
        for cmd, desc in rows:
            out.append(dl_row(3, cmd, desc, x_col))
    out.append("")
    out.append(rule)
    return "\n".join(out)


def render_markdown() -> str:
    """The ``## CLI reference`` block for README.md — command paths + one-line help."""
    lines = ["```"]
    top, groups = _grouped()
    width = max(len(p) for p, _ in iter_commands()) + 2
    for name, cmd in top:
        lines.append(f"{name:<{width}}{_short_help(cmd)}")
    for name, grp, subs in groups:
        lines.append("")
        lines.append(f"{name:<{width}}{_short_help(grp)}")
        for sub_path, sub in subs:
            lines.append(f"{sub_path:<{width}}{_short_help(sub)}")
    lines.append("```")
    lines.append("")
    lines.append("Every command has worked examples in its `--help`. In the container, run "
                 "`commands` for the full picture (CLI plus the tmux session helpers).")
    return "\n".join(lines)


def render_quickstart_markdown() -> str:
    """The ``## Quickstart`` block for README.md — the happy path, top to bottom."""
    width = max(len(c) for c, _ in QUICKSTART) + 3
    lines = ["```"]
    for i, (cmd, desc) in enumerate(QUICKSTART, 1):
        lines.append(f"{i}  {cmd:<{width}}# {desc}")
    lines.append("```")
    return "\n".join(lines)


@_help.command(
    "commands",
    examples=[
        ("commands", "Show everything you can do in this container"),
        ("<any-command> --help", "Worked examples and options for one command"),
    ],
)
@click.option("--markdown", is_flag=True, hidden=True,
              help="Emit the README CLI-reference block instead of the terminal screen.")
def commands(markdown):
    """Show every CVBench command plus the container's tmux session helpers."""
    click.echo(render_markdown() if markdown else render())


if __name__ == "__main__":
    commands()
