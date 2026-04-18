"""Runs API — list and inspect experiment runs.

Planned endpoints
-----------------
    GET  /api/runs            → list all runs (name, status, val_accuracy, date)
    GET  /api/runs/{name}     → full run detail: config + metrics + eval report

Each handler should call cvbench.services (not core directly):
    from cvbench.core.runs import scan_experiments, resolve_run_dir
    from cvbench.core.config import load_config

TODO: implement (tracked in a follow-up GitHub issue)
"""

from fastapi import APIRouter

router = APIRouter()


# @router.get("/runs")
# def list_runs(): ...

# @router.get("/runs/{name}")
# def get_run(name: str): ...
