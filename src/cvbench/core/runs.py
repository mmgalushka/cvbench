from __future__ import annotations

from datetime import date
from pathlib import Path

import click

from cvbench.core.config import CVBenchConfig, load_config


EXPERIMENTS_DIR = "experiments"


def resolve_run_dir(name: str) -> str:
    """Resolve a run name or path to an existing directory.

    Accepts a full path (experiments/my_run) or a bare run name (my_run).
    If the given value does not exist as-is, looks under EXPERIMENTS_DIR.
    """
    p = Path(name)
    if p.exists():
        return str(p)
    candidate = Path(EXPERIMENTS_DIR) / name
    if candidate.exists():
        return str(candidate)
    raise click.BadParameter(
        f"Run directory not found: '{name}' (also tried '{candidate}')",
        param_hint="EXPERIMENT",
    )


# ---------------------------------------------------------------------------
# Run name generation
# ---------------------------------------------------------------------------

def _lr_slug(lr: float) -> str:
    """Convert learning rate to a short slug, e.g. 1e-04 → 'lr1e4'."""
    s = f"{lr:.0e}".replace("-0", "").replace("+0", "").replace(".", "").replace("-", "")
    return f"lr{s}"


def make_run_name(cfg: CVBenchConfig) -> str:
    """Generate a run directory name from config fields + today's date.

    Pattern: {backbone_short}_{lr_slug}_{YYYY_MM_DD}
    Example: effnet_b0_lr1e4_2026_03_28
    """
    backbone = cfg.model.backbone.replace("efficientnet_", "effnet_")
    lr = _lr_slug(cfg.training.learning_rate)
    today = date.today().strftime("%Y_%m_%d")
    return f"{backbone}_{lr}_{today}"


def make_unique_dir(parent: str, name: str) -> Path:
    """Return parent/name, appending _2, _3, etc. if the path already exists."""
    p = Path(parent)
    candidate = p / name
    if not candidate.exists():
        return candidate
    n = 2
    while (p / f"{name}_{n}").exists():
        n += 1
    return p / f"{name}_{n}"


# ---------------------------------------------------------------------------
# Filesystem experiment index
# ---------------------------------------------------------------------------

def _read_entry(exp_dir: Path) -> dict | None:
    """Read config.yaml from an experiment dir and return a flat summary dict.
    Returns None if config.yaml is missing or unreadable.
    """
    try:
        cfg = load_config(str(exp_dir))
    except Exception:
        return None
    return {
        "name": cfg.run.name or exp_dir.name,
        "dir": str(exp_dir),
        "backbone": cfg.model.backbone,
        "lr": cfg.training.learning_rate,
        "epochs": cfg.training.epochs,
        "val_accuracy": cfg.run.val_accuracy,
        "val_loss": cfg.run.val_loss,
        "test_accuracy": cfg.run.test_accuracy,
        "epochs_run": cfg.run.epochs_run,
        "status": cfg.run.status,
        "date": cfg.run.date,
        "resumable": cfg.run.resumable,
        "resume_checkpoint": cfg.run.resume_checkpoint,
        "notes": cfg.run.notes,
    }


def scan_experiments(parent_dir: str, sort_by: str = "date") -> list[dict]:
    """Scan subdirectories of parent_dir and return experiment summaries.

    Directories without config.yaml are silently skipped.
    """
    parent = Path(parent_dir)
    if not parent.exists():
        return []

    entries = []
    for d in sorted(parent.iterdir()):
        if d.is_dir():
            entry = _read_entry(d)
            if entry is not None:
                entries.append(entry)

    valid_sorts = {"val_accuracy", "val_loss", "date", "backbone"}
    key = sort_by if sort_by in valid_sorts else "date"
    reverse = key != "backbone"
    return sorted(
        entries,
        key=lambda r: (r.get(key) is None, r.get(key, "")),
        reverse=reverse,
    )


def best_experiment(parent_dir: str, metric: str = "val_loss") -> dict | None:
    """Return the experiment with the best value for the given metric."""
    entries = [e for e in scan_experiments(parent_dir) if e.get(metric) is not None]
    if not entries:
        return None
    reverse = metric != "val_loss"
    return sorted(entries, key=lambda r: r.get(metric, 0), reverse=reverse)[0]
