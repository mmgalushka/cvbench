"""Runs API — list, inspect, delete, and rename runs."""

import csv
import json
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cvbench.core import present
from cvbench.core.config import config_to_dict, load_config, update_run_status
from cvbench.core.exp_store import (
    EXPERIMENTS_DIR,
    assert_name_available,
    assert_renamable,
    is_sweep_dir,
    resolve_run_dir,
    scan_experiments,
    validate_run_name,
)
from cvbench.core.sweep_store import scan_sweeps, summarize
from cvbench.tasks import task_or_default

router = APIRouter()


class RenameRequest(BaseModel):
    new_name: str


def _with_display(entry: dict) -> dict:
    return {**entry, "display": present.present_run(entry)}


@router.get("/present")
def get_present():
    """Canonical labels, columns and empty-state texts — the same ones the CLI prints."""
    return present.labels()


@router.get("/runs")
def list_runs():
    return [_with_display(e) for e in scan_experiments(EXPERIMENTS_DIR)]


@router.get("/sweeps")
def list_sweeps():
    return [_with_display(e) for e in scan_sweeps(EXPERIMENTS_DIR)]


@router.get("/sweeps/{name}")
def get_sweep(name: str):
    sweep_dir = Path(resolve_run_dir(name, allow_sweep=True))
    if not is_sweep_dir(sweep_dir):
        raise HTTPException(status_code=404, detail=f"'{name}' is not a sweep")

    manifest, rows = summarize(sweep_dir)

    return {
        "name": manifest.name or sweep_dir.name,
        "date": manifest.date,
        "data_dir": manifest.data_dir,
        "strategy": manifest.strategy,
        "metric": manifest.metric,
        "direction": manifest.direction,
        "axes": manifest.axes,
        "trials": [
            {"index": r.index, "dir": r.dir, "params": r.params,
             "status": r.status, "value": r.value, "is_best": r.is_best,
             "test_value": r.test_value, "test_metric": r.test_metric, "is_best_test": r.is_best_test,
             "display": {"value": present.fmt_metric(manifest.metric, r.value),
                         "test": present.fmt_pct(r.test_value)}}
            for r in rows
        ],
    }


@router.get("/runs/{name}/images/{path:path}")
def get_run_image(name: str, path: str):
    run_dir = resolve_run_dir(name)

    cfg = load_config(run_dir)
    test_dir = Path(cfg.data.test_dir).resolve()
    img_path = (test_dir / path).resolve()

    try:
        img_path.relative_to(test_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied") from None

    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")

    return FileResponse(str(img_path))


@router.get("/runs/{name}")
def get_run(name: str):
    run_dir = resolve_run_dir(name)

    run_path = Path(run_dir)
    cfg = load_config(run_dir)

    training_log = []
    log_path = run_path / "training_log.csv"
    if log_path.exists():
        with open(log_path) as f:
            for row in csv.DictReader(f):
                training_log.append({k: float(v) for k, v in row.items()})

    eval_report = None
    eval_path = run_path / "eval_report.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)

    # While a run is still "running", cfg.run.epochs_run is stale (only written
    # once, at the very end) — the training_log rows just read above already give
    # an accurate live count, so use that instead of re-reading the file.
    epochs_run = len(training_log) if cfg.run.status == "running" else cfg.run.epochs_run

    if cfg.run.primary_value is not None:
        primary_value = cfg.run.primary_value
    else:
        report_overall = (eval_report or {}).get("overall")
        primary_value = (
            report_overall.get("value") if isinstance(report_overall, dict)
            # Back-compat: reports written before the "overall" envelope existed.
            else (eval_report or {}).get("overall_accuracy")
        )

    detail = {
        "name": cfg.run.name or run_path.name,
        "dir": run_dir,
        "sweep": run_path.parent.name if is_sweep_dir(run_path.parent) else None,
        "task": cfg.task,
        "status": cfg.run.status,
        "date": cfg.run.date,
        "backbone": cfg.model.backbone,
        "lr": cfg.training.learning_rate,
        "epochs": cfg.training.epochs,
        "epochs_run": epochs_run,
        "val_accuracy": cfg.run.val_accuracy,
        "val_loss": cfg.run.val_loss,
        "metrics": cfg.run.metrics,
        "primary_metric": cfg.run.primary_metric,
        "primary_value": primary_value,
        "task_meta": task_or_default(cfg.task).meta(),
        "resumable": cfg.run.resumable,
        "notes": cfg.run.notes,
        "cli_command": cfg.run.cli_command or "",
        "advisories": present.run_advisories(run_path, eval_report) if eval_report else [],
        "eval_display": (
            present.eval_display(eval_report, cfg.run.val_accuracy) if eval_report else None
        ),
        "config": config_to_dict(cfg),
        "training_log": training_log,
        "eval_report": eval_report,
    }
    detail["display"] = present.run_detail_display(detail)
    return detail


@router.patch("/runs/{name}")
def rename_run(name: str, body: RenameRequest):
    run_dir = Path(resolve_run_dir(name, allow_sweep=True))

    assert_renamable(run_dir)

    cfg = load_config(str(run_dir))
    if cfg.run.status == "running":
        raise HTTPException(status_code=409, detail="Cannot rename a currently running run.")

    validate_run_name(body.new_name)
    assert_name_available(body.new_name, current_dir=run_dir)

    new_dir = run_dir.parent / body.new_name
    os.rename(run_dir, new_dir)
    update_run_status(str(new_dir), name=body.new_name)
    return {"name": body.new_name}


@router.delete("/runs/{name}")
def delete_run(name: str):
    run_dir = Path(resolve_run_dir(name, allow_sweep=True))

    experiments_base = Path(EXPERIMENTS_DIR).resolve()
    try:
        run_dir.resolve().relative_to(experiments_base)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied") from None

    shutil.rmtree(run_dir)
    return {"deleted": name}
