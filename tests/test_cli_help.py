"""Help / discoverability tests — all TensorFlow-free (issue #75).

These lock in the acceptance criteria (every command has a worked example, the
README reflects the interface) and guard against the drift that made the old
hand-written command lists wrong: every command line printed anywhere is
resolved against the live Click registry.
"""
import re
import shlex
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from cvbench.cli import overview

REPO = Path(__file__).resolve().parent.parent

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ENTRY = {name: overview.load_command(spec) for name, spec in overview.ENTRY_POINTS}
_ENTRY["commands"] = overview.commands


def _paths():
    return [p for p, _ in overview.iter_commands()]


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

def test_every_command_has_a_worked_example():
    missing = [p for p, cmd in overview.iter_commands()
               if isinstance(cmd, click.Command) and not getattr(cmd, "examples", ())]
    assert not missing, f"commands with no worked example in --help: {missing}"


@pytest.mark.parametrize("path", _paths())
def test_help_renders_without_ansi_under_clirunner(path):
    root_name, *rest = path.split()
    result = CliRunner().invoke(_ENTRY[root_name], [*rest, "--help"])
    assert result.exit_code == 0, result.output
    assert not _ANSI.search(result.output), f"{path} --help leaked ANSI when not a TTY"
    if path.count(" ") == 0 or not isinstance(dict(overview.iter_commands())[path], click.Group):
        assert "Examples" in result.output


# ---------------------------------------------------------------------------
# Anti-rot: every printed command line must resolve against the registry
# ---------------------------------------------------------------------------

def _resolve(line: str):
    tokens = shlex.split(line)
    assert tokens[0] in _ENTRY, f"{line!r}: unknown command {tokens[0]!r}"
    cmd = _ENTRY[tokens[0]]
    args = tokens[1:]
    while isinstance(cmd, click.Group) and args and not args[0].startswith("-"):
        ctx = click.Context(cmd)
        sub = cmd.get_command(ctx, args[0])
        if sub is None:
            pytest.fail(f"{line!r}: {tokens[0]} has no subcommand {args[0]!r}")
        cmd, args = sub, args[1:]
    known = {o for p in cmd.get_params(click.Context(cmd)) for o in (*p.opts, *p.secondary_opts)}
    for tok in args:
        if tok.startswith("--"):
            name = tok.split("=")[0]
            assert name in known, f"{line!r}: {cmd.name} has no option {name}"


def _all_example_lines():
    for _, cmd in overview.iter_commands():
        for line, _desc in (*getattr(cmd, "examples", ()), *getattr(cmd, "see_also", ())):
            yield line
    for line, _desc in overview.QUICKSTART:
        yield line


# Container conveniences (tmux helpers, see EXTRAS in overview.py) that are
# documented alongside the CLI commands but aren't Click commands themselves —
# `_resolve` walks the Click registry, so there's nothing for it to check
# these against. Cross-checked instead by
# test_bashrc_tm_flags_are_all_in_the_overview.
_NON_CLICK_HELPERS = {"tm"}


def _resolvable_example_lines():
    for line in set(_all_example_lines()):
        if shlex.split(line)[0] not in _NON_CLICK_HELPERS:
            yield line


@pytest.mark.parametrize("line", sorted(_resolvable_example_lines()))
def test_example_lines_resolve(line):
    # A placeholder positional arg (`<run>`) or a trailing `--help` doesn't
    # need special-casing here: `_resolve` only checks the command/subcommand
    # prefix and any `--options`, never positional values, and `--help` is a
    # real option Click adds to every command.
    _resolve(line)


# ---------------------------------------------------------------------------
# Single source of truth
# ---------------------------------------------------------------------------

def _pyproject_scripts():
    text = (REPO / "pyproject.toml").read_text()
    block = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"^([\w-]+)\s*=\s*\"([^\"]+)\"", block, re.MULTILINE)
    }


def test_entry_points_match_pyproject():
    scripts = _pyproject_scripts()
    assert scripts.get("commands") == "cvbench.cli.overview:commands"
    assert {n for n, _ in overview.ENTRY_POINTS} | {"commands"} == set(scripts)
    for name, spec in overview.ENTRY_POINTS:
        assert scripts[name] == spec


def test_overview_lists_every_command_and_the_tmux_helpers():
    # `commands` is command reference only — no quickstart/tutorial content (that
    # lives in the README's Quickstart section instead).
    screen = _ANSI.sub("", overview.render())
    for path in _paths():
        assert path in screen, f"{path} missing from `commands` output"
    for token in ("tm -n", "tm -c", "tm -d", "tm -l"):
        assert token in screen, f"{token} missing from `commands` output"


def test_overview_omits_container_managed_services():
    # TensorBoard / JupyterLab are container services, not commands the user runs.
    screen = _ANSI.sub("", overview.render()).lower()
    assert "tensorboard" not in screen
    assert "jupyter" not in screen


def test_bashrc_tm_flags_are_all_in_the_overview():
    bashrc = (REPO / "scripts" / "bashrc").read_text()
    flags = set(re.findall(r"tm (-[ncdl])\b", bashrc))
    assert flags, "no tm flags found in scripts/bashrc"
    extras = "".join(c for _, rows in overview.EXTRAS for c, _ in rows)
    for flag in flags:
        assert f"tm {flag}" in extras, f"tm {flag} is in bashrc but not the overview"


def test_overview_ports_are_exposed_by_the_dockerfile():
    expose = re.search(r"^EXPOSE (.+)$", (REPO / "Dockerfile").read_text(), re.MULTILINE)
    exposed = set(expose.group(1).split())
    blob = " ".join(
        c + " " + d
        for src in (overview.QUICKSTART, *[rows for _, rows in overview.EXTRAS])
        for c, d in src
    )
    for port in re.findall(r"http://\S*?:(\d{4})", blob):
        assert port in exposed, f"port {port} referenced but not EXPOSEd"


def test_helper_sh_dispatches_every_entry_point():
    helper = (REPO / "helper.sh").read_text()
    case_block = helper.split("case $1 in", 1)[1].split("esac", 1)[0]
    labels = set(re.findall(r"^\s*([a-z]+)\)", case_block, re.MULTILINE))
    expected = {n for n, _ in overview.ENTRY_POINTS} | {"init", "test", "release", "docs"}
    assert expected <= labels, f"helper.sh missing dispatch for: {expected - labels}"


def test_readme_cli_reference_is_up_to_date():
    readme = (REPO / "README.md").read_text()
    m = re.search(
        r"<!-- BEGIN CLI REFERENCE -->\n(.*?)\n<!-- END CLI REFERENCE -->",
        readme, re.DOTALL,
    )
    assert m, "CLI reference markers not found in README.md"
    assert m.group(1) == overview.render_markdown(), \
        "README CLI reference is stale — run `./helper.sh docs`"


def test_readme_quickstart_is_up_to_date():
    readme = (REPO / "README.md").read_text()
    m = re.search(
        r"<!-- BEGIN QUICKSTART -->\n(.*?)\n<!-- END QUICKSTART -->",
        readme, re.DOTALL,
    )
    assert m, "Quickstart markers not found in README.md"
    assert m.group(1) == overview.render_quickstart_markdown(), \
        "README Quickstart is stale — run `./helper.sh docs`"


def test_no_stale_umbrella_command_references():
    pattern = re.compile(
        r"\bcvbench\s+(train|evaluate|predict|runs|data|serve|commands)\b"
    )
    offenders = []
    for rel in ("src", "tests", "scripts", "README.md", "helper.sh", "Dockerfile"):
        target = REPO / rel
        files = target.rglob("*") if target.is_dir() else [target]
        for path in files:
            if not path.is_file() or path.suffix in {".pyc", ".png", ".jpg"}:
                continue
            try:
                lines = path.read_text().splitlines()
            except (UnicodeDecodeError, IsADirectoryError):
                continue
            for i, line in enumerate(lines, 1):
                if pattern.search(line) and "docker" not in line:
                    offenders.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, "there is no `cvbench` umbrella command:\n" + "\n".join(offenders)
