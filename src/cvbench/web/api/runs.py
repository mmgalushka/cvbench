"""Runs API — list and inspect experiment runs."""

import csv
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from cvbench.core.runs import EXPERIMENTS_DIR, scan_experiments, resolve_run_dir
from cvbench.core.config import load_config, OneOfConfig, TransformConfig

router = APIRouter()


@router.get("/runs")
def list_runs():
    return scan_experiments(EXPERIMENTS_DIR)


@router.get("/runs/{name}/images/{path:path}")
def get_run_image(name: str, path: str):
    try:
        run_dir = resolve_run_dir(name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")

    cfg = load_config(run_dir)
    test_dir = Path(cfg.data.test_dir).resolve()
    img_path = (test_dir / path).resolve()

    try:
        img_path.relative_to(test_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")

    return FileResponse(str(img_path))


@router.get("/runs/{name}")
def get_run(name: str):
    try:
        run_dir = resolve_run_dir(name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Run '{name}' not found")

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

    return {
        "name": cfg.run.name or run_path.name,
        "dir": run_dir,
        "status": cfg.run.status,
        "date": cfg.run.date,
        "backbone": cfg.model.backbone,
        "lr": cfg.training.learning_rate,
        "epochs": cfg.training.epochs,
        "epochs_run": cfg.run.epochs_run,
        "val_accuracy": cfg.run.val_accuracy,
        "val_loss": cfg.run.val_loss,
        "test_accuracy": cfg.run.test_accuracy if cfg.run.test_accuracy is not None
            else (eval_report or {}).get("overall_accuracy"),
        "resumable": cfg.run.resumable,
        "notes": cfg.run.notes,
        "config": {
            "data": {
                "data_dir": cfg.data.data_dir,
                "classes": cfg.data.classes,
                "batch_size": cfg.data.batch_size,
                "input_size": cfg.model.input_size,
            },
            "model": {
                "backbone": cfg.model.backbone,
                "dropout": cfg.model.dropout,
                "fine_tune_from_layer": cfg.model.fine_tune_from_layer,
            },
            "training": {
                "epochs": cfg.training.epochs,
                "learning_rate": cfg.training.learning_rate,
                "class_weight": cfg.training.class_weight,
                "checkpoints_strategy": cfg.training.checkpoints.strategy,
                "lr_scheduler_patience": cfg.training.lr_scheduler.patience,
            },
            "augmentation": [
                {"name": t.name, "prob": t.prob, **t.params}
                if isinstance(t, TransformConfig)
                else {
                    "name": "one_of(" + ", ".join(c.name for c in t.candidates) + ")",
                    "prob": t.prob,
                }
                for t in cfg.augmentation.transforms
            ],
        },
        "training_log": training_log,
        "eval_report": eval_report,
    }
