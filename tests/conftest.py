"""Shared fixtures for the whole suite.

Everything a test module used to copy from another one lives here: a global seed,
the experiment-directory helpers (``write_exp``, ``make_sweep``, ``exps``) and tiny
generated datasets (``tiny_cls_root``, ``tiny_yolo_root``, ``yolo_project``).
Test modules must never import helpers from each other.
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from cvbench.cli.generate import generate
from cvbench.core.config import build_config, save_config
from cvbench.core.sweep_store import SweepManifest, write_manifest

SEED = 1234


@pytest.fixture(autouse=True)
def _fresh_task_registry():
    """Drop the cached task registry so a test patching entry points sees its own plugins."""
    from cvbench import tasks

    tasks._registry.cache_clear()
    yield
    tasks._registry.cache_clear()


@pytest.fixture(autouse=True)
def _global_seed():
    """Seed every RNG a test might touch, so a failure reproduces on the next run."""
    random.seed(SEED)
    np.random.seed(SEED)
    if "keras" in sys.modules:  # never pay for TensorFlow in a test that doesn't use it
        import keras

        keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Experiments and sweeps
# ---------------------------------------------------------------------------

def _write_exp(parent: Path, name: str, **run_kwargs) -> Path:
    exp_dir = parent / name
    exp_dir.mkdir(parents=True)
    cfg = build_config("data")
    cfg.run.name = name
    cfg.run.date = "2026-01-01"
    cfg.run.status = "done"
    for k, v in run_kwargs.items():
        if k in ("val_accuracy", "val_loss"):  # shorthand for the run's headline metrics
            cfg.run.metrics[k] = v
        else:
            setattr(cfg.run, k, v)
    save_config(cfg, str(exp_dir))
    return exp_dir


def _make_sweep(experiments: Path, name: str = "sw", date: str = "2026-02-01") -> Path:
    d = experiments / name
    d.mkdir(parents=True)
    write_manifest(d, SweepManifest(
        version=1, name=name, date=date, data_dir="data", strategy="grid",
        metric="val_loss", direction="min", axes={"lr": ["0.1", "0.01"]},
    ))
    return d


@pytest.fixture
def write_exp():
    """``write_exp(parent, name, **run_fields)`` — a finished run's config.yaml on disk."""
    return _write_exp


@pytest.fixture
def make_sweep():
    """``make_sweep(experiments_dir, name="sw", date=...)`` — a sweep dir with its manifest."""
    return _make_sweep


@pytest.fixture
def exps(tmp_path, monkeypatch):
    """An empty ``experiments/`` under tmp_path, which becomes the cwd and the default store."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "experiments"
    path.mkdir()
    for target in (
        "cvbench.core.exp_store.EXPERIMENTS_DIR",
        "cvbench.web.api.runs.EXPERIMENTS_DIR",
        "cvbench.cli.runs.EXPERIMENTS_DIR",
        "cvbench.cli.runs._DEFAULT_EXPERIMENTS_DIR",
    ):
        monkeypatch.setattr(target, str(path), raising=False)
    return path


# ---------------------------------------------------------------------------
# Tiny datasets — generated once per session, copied per test (tests may mutate them)
# ---------------------------------------------------------------------------

def _generate(out: Path, *args: str) -> Path:
    result = CliRunner().invoke(generate, [str(out), *args])
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture(scope="session")
def _cls_template(tmp_path_factory) -> Path:
    return _generate(
        tmp_path_factory.mktemp("cls_template") / "cls",
        "--train", "4", "--val", "2", "--test", "2", "--image-size", "32", "--seed", "1",
    )


@pytest.fixture(scope="session")
def _yolo_template(tmp_path_factory) -> Path:
    return _generate(
        tmp_path_factory.mktemp("yolo_template") / "yolo",
        "--format", "yolo", "--train", "6", "--val", "3", "--test", "4",
        "--image-size", "64", "--max-objects", "2", "--seed", "1",
    )


@pytest.fixture
def tiny_cls_root(tmp_path, _cls_template) -> Path:
    """A classification dataset (train 4 / val 2 / test 2 per class, 32px) in tmp_path."""
    return Path(shutil.copytree(_cls_template, tmp_path / "tiny_cls")).resolve()


@pytest.fixture
def tiny_plugin_root(tiny_cls_root) -> Path:
    """The classification dataset plus the marker file the dummy plugin task sniffs for."""
    (tiny_cls_root / "PLUGIN_MARKER").touch()
    return tiny_cls_root


@pytest.fixture
def tiny_flat_root(tmp_path, _cls_template) -> Path:
    """An unlabeled dataset for the test-only autoencoder task: train/val/test hold images directly."""
    root = tmp_path / "tiny_flat"
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        for i, src in enumerate(sorted((_cls_template / split).rglob("*.*"))):
            shutil.copy(src, root / split / f"{i:03d}{src.suffix}")
    return root.resolve()


@pytest.fixture
def tiny_yolo_root(tmp_path, _yolo_template) -> Path:
    """A YOLO detection dataset (train 6 / val 3 / test 4, 64px) in tmp_path."""
    return Path(shutil.copytree(_yolo_template, tmp_path / "tiny_yolo")).resolve()


@pytest.fixture
def yolo_project(tmp_path, monkeypatch, _yolo_template) -> Path:
    """tmp_path as the cwd with a YOLO dataset at ``data/yolo`` — for end-to-end runs."""
    monkeypatch.chdir(tmp_path)
    shutil.copytree(_yolo_template, tmp_path / "data" / "yolo")
    return tmp_path


# ---------------------------------------------------------------------------
# Experiments backed by the tiny classification dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def cls_run(exps, tiny_cls_root, write_exp) -> Path:
    """A finished classification run ``cls_run`` pointing at ``tiny_cls_root`` (no checkpoint)."""
    from cvbench.core.config import load_config

    run_dir = write_exp(exps, "cls_run", val_accuracy=0.5, val_loss=0.7, epochs_run=2)
    cfg = load_config(str(run_dir))
    cfg.data.data_dir = str(tiny_cls_root)
    cfg.data.train_dir = str(tiny_cls_root / "train")
    cfg.data.val_dir = str(tiny_cls_root / "val")
    cfg.data.test_dir = str(tiny_cls_root / "test")
    cfg.data.classes = sorted(p.name for p in (tiny_cls_root / "train").iterdir())
    cfg.model.num_classes = len(cfg.data.classes)
    cfg.model.input_size = 32
    save_config(cfg, str(run_dir))
    return run_dir


@pytest.fixture
def cls_run_with_model(cls_run) -> Path:
    """``cls_run`` plus a real (untrained) Conv2D classifier saved as ``best.keras``."""
    import keras

    from cvbench.core.config import load_config

    n = load_config(str(cls_run)).model.num_classes
    inputs = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(4, 3, activation="relu", name="last_conv")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(n, activation="softmax")(x)
    keras.Model(inputs, outputs).save(str(cls_run / "best.keras"))
    return cls_run


# One dataset fixture per registered task. The lifecycle smoke test resolves its dataset
# through ``task_dataset``, so a task added to TASK_NAMES runs it as soon as it has an entry here.
_TASK_DATASET_FIXTURES = {
    "classification": "tiny_cls_root",
    "detection": "tiny_yolo_root",
    "plugin_cls": "tiny_plugin_root",
    "autoencoder": "tiny_flat_root",
}


@pytest.fixture
def task_dataset(request):
    """``task_dataset(task_name)`` — the tiny dataset fixture registered for that task."""

    def resolve(task_name: str) -> Path:
        fixture = _TASK_DATASET_FIXTURES.get(task_name)
        if fixture is None:
            pytest.fail(
                f"Task {task_name!r} has no tiny dataset fixture: add one to tests/conftest.py "
                "and register it in _TASK_DATASET_FIXTURES."
            )
        return request.getfixturevalue(fixture)

    return resolve
