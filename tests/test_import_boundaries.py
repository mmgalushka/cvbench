"""Package boundary tests — enforce the architecture, not just document it.

* ``cvbench.datasets`` must stay TensorFlow-free: the web API imports it to
  browse dataset folders without paying for a TensorFlow import.
* ``cvbench.core`` must never import a task package (``cvbench.classification``,
  ``cvbench.detection``) — task-specific code belongs above core, not inside it.
* Only the task packages and ``cvbench.tasks`` may compare against a task name:
  everything else asks the ``Task`` (see ``core/task.py``).
"""
import ast
import subprocess
import sys
from pathlib import Path

from cvbench.tasks import TASK_NAMES

SRC = Path(__file__).resolve().parent.parent / "src" / "cvbench"


def test_datasets_package_does_not_import_tensorflow():
    # Run in a subprocess so an earlier import in this test session can't
    # mask the real cold-import behaviour.
    code = (
        "import sys\n"
        "import cvbench.datasets\n"
        "import cvbench.datasets.layout\n"
        "import cvbench.datasets.shapes\n"
        "import cvbench.datasets.synth\n"
        "import cvbench.datasets.stats\n"
        "import cvbench.datasets.hashify\n"
        "import cvbench.datasets.prep\n"
        "import cvbench.datasets.split\n"
        "import cvbench.datasets.flatten\n"
        "assert 'tensorflow' not in sys.modules, sorted(sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_package_never_imports_tensorflow_at_module_scope():
    # `commands` enumerates every CLI module to build the overview; if any of
    # them pulled TensorFlow at import time, the overview (and every --help)
    # would take seconds to start. Run in a subprocess for a true cold import.
    code = (
        "import sys, importlib\n"
        "from cvbench.cli.overview import ENTRY_POINTS\n"
        "import cvbench.cli.overview\n"
        "for _, spec in ENTRY_POINTS:\n"
        "    importlib.import_module(spec.split(':')[0])\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'tensorflow')\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _imported_top_level_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_core_never_imports_a_task_package():
    forbidden = set(TASK_NAMES)
    offenders = []
    for py_file in (SRC / "core").rglob("*.py"):
        for node in ast.walk(ast.parse(py_file.read_text(), filename=str(py_file))):
            module = None
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if module.startswith("cvbench.") and module.split(".")[1] in forbidden:
                        offenders.append((py_file, module))
                continue
            if module and module.startswith("cvbench.") and module.split(".")[1] in forbidden:
                offenders.append((py_file, module))
    assert not offenders, f"core/ must not import task packages: {offenders}"


def test_task_names_are_never_compared_outside_task_packages():
    allowed = {SRC / "tasks.py", *SRC.joinpath("web", "static").rglob("*")}
    offenders = []
    for py_file in SRC.rglob("*.py"):
        if py_file in allowed or py_file.relative_to(SRC).parts[0] in TASK_NAMES:
            continue
        for node in ast.walk(ast.parse(py_file.read_text(), filename=str(py_file))):
            if isinstance(node, ast.Compare):
                operands = [node.left, *node.comparators]
                for op in operands:
                    consts = [op] if isinstance(op, ast.Constant) else getattr(op, "elts", [])
                    if any(isinstance(c, ast.Constant) and c.value in TASK_NAMES for c in consts):
                        offenders.append((py_file.relative_to(SRC), node.lineno))
    assert not offenders, f"compare against a Task, not a task-name string: {offenders}"
