"""Runs API with sweeps: listing, detail, delete, rename guard, trial back-link."""
import pytest

pytest.importorskip("fastapi", reason="requires the 'web' extra")

from fastapi import HTTPException  # noqa: E402

from cvbench.core.errors import NotFoundError  # noqa: E402
from cvbench.web.api import runs as api  # noqa: E402
from tests.test_sweep_runs import _sweep, _write_exp  # noqa: E402


@pytest.fixture
def exps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "experiments"
    d.mkdir()
    return d


@pytest.fixture
def sweep(exps):
    sw = _sweep(exps)
    _write_exp(sw, "sw_001", val_loss=0.5)
    _write_exp(sw, "sw_002", val_loss=0.2)
    return sw


def test_list_sweeps_and_runs_stay_separate(exps, sweep):
    _write_exp(exps, "manual")
    assert [s["name"] for s in api.list_sweeps()] == ["sw"]
    assert [r["name"] for r in api.list_runs()] == ["manual"]


def test_get_sweep_rows_and_best(sweep):
    data = api.get_sweep("sw")
    assert data["metric"] == "val_loss" and data["axes"] == {"lr": ["0.1", "0.01"]}
    assert [t["dir"] for t in data["trials"]] == ["sw_001", "sw_002"]
    assert [t["is_best"] for t in data["trials"]] == [False, True]


def test_get_sweep_missing_trial(exps):
    sw = _sweep(exps)
    _write_exp(sw, "sw_001", val_loss=0.5)
    statuses = [t["status"] for t in api.get_sweep("sw")["trials"]]
    assert statuses == ["done", "missing"]


def test_get_sweep_unknown_raises_not_found(exps):
    with pytest.raises(NotFoundError):
        api.get_sweep("nope")


def test_get_sweep_not_a_sweep(exps):
    _write_exp(exps, "manual")
    with pytest.raises(HTTPException) as e:
        api.get_sweep("manual")
    assert e.value.status_code == 404


def test_delete_sweep(exps, sweep):
    api.delete_run("sw")
    assert not sweep.exists()


def test_rename_sweep_and_trial_rejected(sweep):
    for name in ("sw", "sw_001"):
        with pytest.raises(HTTPException) as e:
            api.rename_run(name, api.RenameRequest(new_name="other"))
        assert e.value.status_code == 422


def test_get_run_sweep_field(exps, sweep):
    _write_exp(exps, "manual")
    assert api.get_run("sw_001")["sweep"] == "sw"
    assert api.get_run("manual")["sweep"] is None


def test_web_domain_error_handler_returns_404_with_suggestion_and_candidates(exps):
    from fastapi.testclient import TestClient

    from cvbench.web.app import create_app

    _write_exp(exps, "resnet_run")
    _write_exp(exps, "cls_a")
    _write_exp(exps, "cls_b")
    client = TestClient(create_app())

    missing = client.get("/api/runs/resnet_rum")
    assert missing.status_code == 404
    assert "Did you mean 'resnet_run'?" in missing.json()["detail"]
    assert missing.json()["candidates"] == ["resnet_run"]

    ambiguous = client.get("/api/runs/cls_")
    assert ambiguous.status_code == 409
    assert ambiguous.json()["candidates"] == ["cls_a", "cls_b"]
