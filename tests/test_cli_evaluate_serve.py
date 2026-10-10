"""`evaluate` and `serve` command wiring, without TensorFlow or a web server."""
import sys
from types import SimpleNamespace

from click.testing import CliRunner

from cvbench.cli import evaluate as evaluate_mod
from cvbench.cli.serve import serve


def _fake_evaluation(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        run_evaluation=lambda **kw: calls.append(("run", kw)),
        run_sweep_evaluation=lambda name: calls.append(("sweep", name)),
    )
    monkeypatch.setitem(sys.modules, "cvbench.services.evaluation", fake)
    return calls


def test_evaluate_run_passes_options_to_service(monkeypatch):
    calls = _fake_evaluation(monkeypatch)
    monkeypatch.setattr("cvbench.core.exp_store.resolve_run_dir", lambda n, allow_sweep=False: n)
    monkeypatch.setattr("cvbench.core.exp_store.is_sweep_dir", lambda p: False)

    result = CliRunner().invoke(evaluate_mod.evaluate, ["my_run", "--output-dir", "out"])

    assert result.exit_code == 0, result.output
    assert calls == [("run", {"experiment": "my_run", "output_dir": "out", "conf": None})]


def test_evaluate_sweep_prints_trial_table(monkeypatch):
    calls = _fake_evaluation(monkeypatch)
    monkeypatch.setattr("cvbench.core.exp_store.resolve_run_dir", lambda n, allow_sweep=False: n)
    monkeypatch.setattr("cvbench.core.exp_store.is_sweep_dir", lambda p: True)
    manifest = SimpleNamespace(axes={"lr": [1, 2]}, metric="val_accuracy")
    monkeypatch.setattr(evaluate_mod, "summarize", lambda path: (manifest, ["row"]))
    table = []
    monkeypatch.setattr(
        "cvbench.cli.sweep.print_trial_table",
        lambda rows, axes, metric, show_test: table.append((rows, axes, metric, show_test)),
    )

    result = CliRunner().invoke(evaluate_mod.evaluate, ["my_sweep"])

    assert result.exit_code == 0, result.output
    assert calls == [("sweep", "my_sweep")]
    assert table == [(["row"], ["lr"], "val_accuracy", True)]


def test_evaluate_sweep_rejects_output_dir(monkeypatch):
    calls = _fake_evaluation(monkeypatch)
    monkeypatch.setattr("cvbench.core.exp_store.resolve_run_dir", lambda n, allow_sweep=False: n)
    monkeypatch.setattr("cvbench.core.exp_store.is_sweep_dir", lambda p: True)

    result = CliRunner().invoke(evaluate_mod.evaluate, ["my_sweep", "--output-dir", "out"])

    assert result.exit_code == 2
    assert "--output-dir is not supported" in result.output
    assert calls == []


def test_evaluate_passes_conf_to_service(monkeypatch):
    calls = _fake_evaluation(monkeypatch)
    monkeypatch.setattr("cvbench.core.exp_store.resolve_run_dir", lambda n, allow_sweep=False: n)
    monkeypatch.setattr("cvbench.core.exp_store.is_sweep_dir", lambda p: False)

    result = CliRunner().invoke(evaluate_mod.evaluate, ["my_run", "--conf", "0.4"])

    assert result.exit_code == 0, result.output
    assert calls == [("run", {"experiment": "my_run", "output_dir": None, "conf": 0.4})]


def test_evaluate_sweep_rejects_conf(monkeypatch):
    calls = _fake_evaluation(monkeypatch)
    monkeypatch.setattr("cvbench.core.exp_store.resolve_run_dir", lambda n, allow_sweep=False: n)
    monkeypatch.setattr("cvbench.core.exp_store.is_sweep_dir", lambda p: True)

    result = CliRunner().invoke(evaluate_mod.evaluate, ["my_sweep", "--conf", "0.4"])

    assert result.exit_code == 2
    assert "--conf is not supported" in result.output
    assert calls == []


def _fake_uvicorn(monkeypatch):
    runs = []
    monkeypatch.setitem(
        sys.modules, "uvicorn", SimpleNamespace(run=lambda *a, **kw: runs.append((a, kw)))
    )
    return runs


def test_serve_starts_uvicorn_on_given_host_and_port(monkeypatch):
    runs = _fake_uvicorn(monkeypatch)
    monkeypatch.delenv("CVBENCH_URL", raising=False)

    result = CliRunner().invoke(serve, ["--host", "0.0.0.0", "--port", "9001"])

    assert result.exit_code == 0, result.output
    assert "http://0.0.0.0:9001" in result.output and "CVBENCH_URL" in result.output
    assert runs == [(("cvbench.web.app:create_app",),
                     {"factory": True, "host": "0.0.0.0", "port": 9001})]


def test_serve_prints_cvbench_url_override(monkeypatch):
    _fake_uvicorn(monkeypatch)
    monkeypatch.setenv("CVBENCH_URL", "http://lab-box:8000")

    result = CliRunner().invoke(serve, [])

    assert "CVBench WebUI → http://lab-box:8000" in result.output
    assert "Set CVBENCH_URL" not in result.output


def test_serve_without_web_extras_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    result = CliRunner().invoke(serve, [])

    assert result.exit_code == 1
    assert "pip install cvbench[web]" in result.output
