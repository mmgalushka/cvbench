"""Shared Click help presentation — every CVBench command renders through this.

Commands opt in by using :func:`command` / :func:`group` from this module
instead of ``click.command`` / ``click.group``. Groups propagate the styling to
their subcommands automatically (``CVBenchGroup.command_class``), so a
``@some_group.command(...)`` decorator needs no extra wiring.

Worked examples and "what to run next" hints are passed as structured data::

    @_help.command(
        examples=[("train data/synthetic --epochs 5", "Smoke-test the pipeline")],
        see_also=[("evaluate <run>", "score it on the held-out test split")],
    )

rendered by us (never through Click's paragraph rewrapper, so command lines stay
copy-pasteable) and reused verbatim by the ``commands`` overview.

All colour goes through :mod:`cvbench.core._console`, which already drops ANSI under
``NO_COLOR`` and when stdout is not a TTY.
"""
from __future__ import annotations

import click
from click.formatting import HelpFormatter

from cvbench.core import _console

_MAX_WIDTH = 100

# (command_line, one-line description) — used by --help and by the overview.
Example = tuple[str, str]


class CVBenchHelpFormatter(HelpFormatter):
    """Widen past Click's 80-col cap and render headings bold, without a colon."""

    def __init__(self, indent_increment: int = 2, width=None, max_width=None):
        if width is None and max_width is None:
            max_width = min(_console.term_width(80), _MAX_WIDTH)
        super().__init__(indent_increment, width, max_width)

    def write_heading(self, heading: str) -> None:
        self.write(f"{'':>{self.current_indent}}{_console.bold(heading)}\n")


class CVBenchContext(click.Context):
    formatter_class = CVBenchHelpFormatter


class HelpMixin:
    """Shared help rendering for :class:`CVBenchCommand` and :class:`CVBenchGroup`."""

    context_class = CVBenchContext

    def __init__(self, *args, examples=(), see_also=(), **kwargs):
        self.examples: tuple[Example, ...] = tuple(examples)
        self.see_also: tuple[Example, ...] = tuple(see_also)
        super().__init__(*args, **kwargs)

    def format_usage(self, ctx, formatter):
        w = formatter.width
        formatter.write(_console.rule(w, "white") + "\n")
        formatter.write(f" {_console.bold('CVBench')} {_console.dim('—')} {ctx.command_path}\n")
        formatter.write(_console.rule(w, "white") + "\n")
        formatter.write_paragraph()
        pieces = self.collect_usage_pieces(ctx)
        formatter.write_usage(ctx.command_path, " ".join(pieces))

    def format_epilog(self, ctx, formatter):
        if self.examples:
            formatter.write_paragraph()
            formatter.write_heading("Examples")
            formatter.indent()
            for i, (line, desc) in enumerate(self.examples):
                if i:
                    formatter.write_paragraph()
                pad = " " * formatter.current_indent
                formatter.write(f"{pad}{_console.dim('# ' + desc)}\n")
                formatter.write(f"{pad}{_console.green(line)}\n")
            formatter.dedent()

        if self.see_also:
            formatter.write_paragraph()
            formatter.write_heading("Next")
            formatter.indent()
            formatter.write_dl(
                [(_console.green(cmd), why) for cmd, why in self.see_also], col_max=38
            )
            formatter.dedent()

        formatter.write_paragraph()
        formatter.write(_console.rule(formatter.width) + "\n")


class CVBenchCommand(HelpMixin, click.Command):
    pass


class CVBenchGroup(HelpMixin, click.Group):
    command_class = CVBenchCommand


CVBenchGroup.group_class = CVBenchGroup  # nested groups keep the styling


def command(*args, **kwargs):
    """``click.command`` with the CVBench help formatter and ``examples=`` support."""
    kwargs.setdefault("cls", CVBenchCommand)
    return click.command(*args, **kwargs)


def group(*args, **kwargs):
    """``click.group`` with the CVBench help formatter, propagated to subcommands."""
    kwargs.setdefault("cls", CVBenchGroup)
    return click.group(*args, **kwargs)
