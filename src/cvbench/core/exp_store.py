"""Manage experiment directories under `experiments/`.

Name resolution (bare name vs. literal path, availability checks) is
delegated to a shared `Registry` (see `core/registry.py`); this module also
carries run-name generation and the filesystem experiment index used by
`runs list` / `runs best` and the WebUI.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import click

from cvbench.core.config import CVBenchConfig, load_config
from cvbench.core.registry import Registry

EXPERIMENTS_DIR = "experiments"
SWEEP_MANIFEST = "sweep.yaml"

EXP_REG = Registry(
    EXPERIMENTS_DIR,
    not_found_label="Run directory",
    entity_name="experiment",
    param_hint="EXPERIMENT",
)


def validate_run_name(name: str) -> str:
    """Raise ValueError if name is not safe to use as an experiment directory name."""
    return EXP_REG.validate_name(name, what="Name")


def is_sweep_dir(path: Path | str) -> bool:
    """True for a sweep directory: has a `sweep.yaml` manifest and is not itself an experiment."""
    p = Path(path)
    return (p / SWEEP_MANIFEST).is_file() and not (p / "config.yaml").exists()


def sweep_dirs(parent_dir: str = EXPERIMENTS_DIR) -> list[Path]:
    """Sweep directories directly under parent_dir, sorted by name."""
    parent = Path(parent_dir)
    if not parent.is_dir():
        return []
    return [d for d in sorted(parent.iterdir()) if d.is_dir() and is_sweep_dir(d)]


def resolve_experiments_dir(name_or_path: str) -> str:
    """Resolve a directory argument for `runs list` / `runs best`.

    A literal path is used as-is; otherwise a bare name (e.g. a sweep) is looked up under
    EXPERIMENTS_DIR. Falls back to the given value so callers report their own "not found".
    """
    if Path(name_or_path).is_dir():
        return name_or_path
    candidate = Path(EXPERIMENTS_DIR) / name_or_path
    return str(candidate) if candidate.is_dir() else name_or_path


def assert_name_available(new_name: str, current_dir: Path | None = None) -> None:
    """Raise ValueError if new_name conflicts with an existing experiment or sweep trial directory."""
    EXP_REG.base_dir = EXPERIMENTS_DIR
    EXP_REG.assert_available(new_name, current_dir=current_dir)
    # Trials are addressed by bare name too, so their names must stay unique across sweeps.
    for sweep in sweep_dirs(EXPERIMENTS_DIR):
        for trial in sweep.iterdir():
            if not trial.is_dir() or trial.name.lower() != new_name.lower():
                continue
            if current_dir and trial.resolve() == current_dir.resolve():
                continue
            raise ValueError(f"A trial named '{trial.name}' already exists in sweep '{sweep.name}'.")


def assert_renamable(run_dir: Path) -> None:
    """Raise ValueError if run_dir is a sweep or a sweep trial (renaming would break the sweep)."""
    if is_sweep_dir(run_dir):
        raise ValueError(
            f"'{run_dir.name}' is a sweep; renaming it would break its trial names. "
            "Rename individual runs instead, or start a new sweep with --name."
        )
    if is_sweep_dir(run_dir.parent):
        raise ValueError(
            f"'{run_dir.name}' is a trial of sweep '{run_dir.parent.name}'; "
            "renaming it would make the sweep report it as missing."
        )


def resolve_run_dir(name: str, *, allow_sweep: bool = False) -> str:
    """Resolve a run name or path to an existing directory.

    Accepts a full path (experiments/my_run) or a bare run name (my_run).
    If the given value does not exist as-is, looks under EXPERIMENTS_DIR, then
    inside sweep directories (a sweep trial is addressable by its bare name).

    A sweep directory is not a single run (it has no config.yaml), so it is rejected with
    a pointer to its trials unless the caller handles sweeps (`allow_sweep=True`).
    """
    EXP_REG.base_dir = EXPERIMENTS_DIR
    try:
        resolved = EXP_REG.resolve(name)
    except click.BadParameter:
        for sweep in sweep_dirs(EXPERIMENTS_DIR):
            trial = sweep / name
            if (trial / "config.yaml").is_file():
                return str(trial)
        raise
    if not allow_sweep and is_sweep_dir(resolved):
        sweep_name = Path(resolved).name
        raise click.BadParameter(
            f"'{sweep_name}' is a sweep (a group of trials), not a single run. "
            f"Use one of its trials instead; `runs list {sweep_name}` shows them.",
            param_hint=EXP_REG.param_hint,
        )
    return resolved


# ---------------------------------------------------------------------------
# Run name generation
# ---------------------------------------------------------------------------

def _lr_slug(lr: float) -> str:
    """Convert learning rate to a short slug, e.g. 1e-04 → 'lr1e4'."""
    s = f"{lr:.0e}".replace("-0", "").replace("+0", "").replace(".", "").replace("-", "")
    return f"lr{s}"


def make_run_name(cfg: CVBenchConfig) -> str:
    """Generate a run directory name from config fields + today's date.

    Pattern: {task_prefix}_{backbone_short}_{lr_slug}_{YYYY_MM_DD}
    Example: cls_effnet_b0_lr1e4_2026_03_28, det_resnet_18_lr1e4_2026_09_01

    Same shape for every task — only the "cls"/"det" prefix changes — so a
    classification and a detection run trained the same day with the same
    backbone/LR no longer produce identical names.
    """
    task_prefix = "det" if cfg.task == "detection" else "cls"
    backbone = cfg.model.backbone.replace("efficientnet_", "effnet_")
    lr = _lr_slug(cfg.training.learning_rate)
    today = date.today().strftime("%Y_%m_%d")
    return f"{task_prefix}_{backbone}_{lr}_{today}"


def make_unique_dir(parent: str, name: str) -> Path:
    """Return parent/name, appending _2, _3, etc. if the path already exists."""
    return Registry(parent).unique_path(name)


# ---------------------------------------------------------------------------
# Filesystem experiment index
# ---------------------------------------------------------------------------

def _resolve_test_accuracy(exp_dir: Path, config_value):
    if config_value is not None:
        return config_value
    report_path = exp_dir / "eval_report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
            overall = report.get("overall")
            if isinstance(overall, dict):
                return overall.get("value")
            # Back-compat: reports written before the "overall" envelope existed.
            return report.get("overall_accuracy")
        except Exception:
            pass
    return None


def epochs_done_live(exp_dir: Path | str, cfg: CVBenchConfig) -> int:
    """Live epoch-progress count for a run.

    `cfg.run.epochs_run` is only written once, at the very end of training (see
    core/trainer.py), so while a run is still "running" it stays stuck at whatever
    it was when the run started/resumed. Keras's CSVLogger, on the other hand,
    appends one row to training_log.csv after every completed epoch — live, while
    model.fit() is still running — so for a running run that row count is the
    accurate live progress; for anything else cfg.run.epochs_run is already correct
    and cheaper (no file read).
    """
    if cfg.run.status != "running":
        return cfg.run.epochs_run
    log_path = Path(exp_dir) / "training_log.csv"
    try:
        with open(log_path) as f:
            return max(sum(1 for _ in f) - 1, 0)  # -1 for the header row
    except OSError:
        return cfg.run.epochs_run


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
        "task": cfg.task,
        "backbone": cfg.model.backbone,
        "lr": cfg.training.learning_rate,
        "epochs": cfg.training.epochs,
        "val_accuracy": cfg.run.val_accuracy,
        "val_loss": cfg.run.val_loss,
        "test_accuracy": _resolve_test_accuracy(exp_dir, cfg.run.test_accuracy),
        "test_metric": cfg.run.test_metric,
        "epochs_run": epochs_done_live(exp_dir, cfg),
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
