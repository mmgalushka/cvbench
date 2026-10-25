"""Conformance suite for the Task registry — the "adding a third task is easy" harness.

Runs over every name in ``cvbench.tasks.TASK_NAMES``: each must instantiate,
declare a unique ``name`` matching its registry key, and implement every
abstract method on ``Task``. A new task that forgets a method fails here
first, not at some far-off orchestration call site.

``test_task_lifecycle`` goes one step further: it drives each registered task through
resolve_layout → build_datasets → build_model → evaluate → test_score/headline_metrics on a
tiny generated dataset, so a task whose pieces don't fit together fails here too. A task with
no dataset fixture in ``tests/conftest.py`` fails the test with instructions rather than
being skipped.
"""
import types

import pytest

from cvbench import tasks as tasks_mod
from cvbench.core.errors import ConfigError
from cvbench.tasks import TASK_NAMES as BUILTIN_TASK_NAMES
from cvbench.tasks import get_task

# The suite also runs over test-only plugin tasks, registered through faked entry points:
# a classification clone, and an unlabeled autoencoder with two artifacts and its own fit loop.
PLUGIN_EP = types.SimpleNamespace(
    name="plugin_cls", module="tests.plugin_task", attr="PluginTask",
    value="tests.plugin_task:PluginTask",
)
AUTOENCODER_EP = types.SimpleNamespace(
    name="autoencoder", module="tests.autoencoder_task", attr="AutoEncoderTask",
    value="tests.autoencoder_task:AutoEncoderTask",
)
PLUGIN_EPS = {ep.name: ep for ep in (PLUGIN_EP, AUTOENCODER_EP)}
TASK_NAMES = (*BUILTIN_TASK_NAMES, *PLUGIN_EPS)


@pytest.fixture(autouse=True)
def _plugin_task_installed(request, monkeypatch):
    # Only tests parametrized over a plugin see it, so other tests keep the built-in registry.
    if getattr(request.node, "callspec", None) is None:
        return
    ep = PLUGIN_EPS.get(request.node.callspec.params.get("name"))
    if ep is None:
        return
    real = tasks_mod.entry_points
    monkeypatch.setattr(
        tasks_mod, "entry_points",
        lambda group: [*real(group=group), ep] if group == "cvbench.tasks" else real(group=group),
    )


def test_task_names_nonempty():
    assert TASK_NAMES
    assert "classification" in TASK_NAMES


@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_instantiates(name):
    from cvbench.core.task import Task

    task = get_task(name)
    assert isinstance(task, Task)


@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_name_matches_registry_key(name):
    task = get_task(name)
    assert task.name == name


@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_metadata_is_consistent(name):
    task = get_task(name)
    schema = {m.name: m for m in task.metrics_schema()}
    assert task.primary_metric.name in schema
    assert task.sweep_metric in task.selection_metric_names
    assert set(task.selection_metric_names) <= set(schema)
    assert all(m.direction in ("min", "max") for m in schema.values())
    assert task.predict_options() <= {"conf", "top_k"}
    assert task.run_prefix == task.short_name


def test_short_names_are_unique():
    shorts = [get_task(n).short_name for n in BUILTIN_TASK_NAMES]
    assert len(set(shorts)) == len(shorts)


@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_meta_is_json_serializable(name):
    import json

    meta = json.loads(json.dumps(get_task(name).meta()))
    assert meta["name"] == name
    assert meta["primary_metric"] in [m["name"] for m in meta["metrics"]]


def test_task_packages_import_without_tensorflow():
    """Metadata, report rendering and prediction shaping must not pay for TensorFlow."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from cvbench.tasks import TASK_NAMES, get_task\n"
        "for n in TASK_NAMES:\n"
        "    t = get_task(n)\n"
        "    t.meta(); t.metrics_schema(); t.predict_options()\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('tensorflow', 'keras'))\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_check_predict_options_names_the_supporting_tasks():
    from cvbench.tasks import check_predict_options

    cls, det = get_task("classification"), get_task("detection")
    check_predict_options(cls, None, 3)
    check_predict_options(det, 0.4, 1)
    with pytest.raises(ConfigError, match="--conf only applies to detection runs"):
        check_predict_options(cls, 0.4)
    with pytest.raises(ConfigError, match="--top-k and --probs only apply to classification runs"):
        check_predict_options(det, None, 3)


def test_detect_task_name_and_layout_errors_dispatch_through_tasks(tmp_path):
    from cvbench.tasks import detect_task_name, layout_errors

    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels").mkdir()
    assert detect_task_name(tmp_path) == "detection"
    assert any("images/val" in e for e in layout_errors(tmp_path))


@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_implements_every_abstract_method(name):
    """ABCs refuse to instantiate with unimplemented abstract methods —
    get_task() succeeding above is itself most of this proof. This test
    additionally confirms nothing overrides an abstract method with a
    non-callable, and that no abstract method is left on the base class name.
    """

    from cvbench.core.task import Task

    task = get_task(name)
    for method_name in Task.__abstractmethods__:
        method = getattr(task, method_name)
        assert callable(method), f"{name}.{method_name} must be callable"
        assert method.__qualname__.split(".")[0] != "Task", (
            f"{name} does not override abstract method {method_name!r}"
        )


@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_view_implements_every_abstract_view_method(name):
    """``view()`` returns a TaskView that overrides every abstract view method."""
    from cvbench.core.task import TaskView

    view = get_task(name).view()
    assert isinstance(view, TaskView)
    for method_name in TaskView.__abstractmethods__:
        method = getattr(view, method_name)
        assert callable(method), f"{name} view.{method_name} must be callable"
        assert method.__qualname__.split(".")[0] != "TaskView", (
            f"{name} view does not override abstract method {method_name!r}"
        )


def test_task_view_cannot_be_instantiated_directly():
    from cvbench.core.task import TaskView

    with pytest.raises(TypeError):
        TaskView()


def test_get_task_unknown_name_lists_valid_options():
    with pytest.raises(ConfigError, match="Unknown task 'nonsense'"):
        get_task("nonsense")
    with pytest.raises(ConfigError, match="classification"):
        get_task("nonsense")


def test_task_cannot_be_instantiated_directly():
    from cvbench.core.task import Task

    with pytest.raises(TypeError):
        Task()


def test_dataset_spec_is_a_plain_frozen_record():
    from dataclasses import FrozenInstanceError

    from cvbench.core.task import DatasetSpec

    spec = DatasetSpec(
        class_names=["a", "b"],
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        num_train_images=10,
    )
    assert spec.class_names == ["a", "b"]
    with pytest.raises(FrozenInstanceError):
        spec.train_dir = "other"


@pytest.mark.tf
@pytest.mark.parametrize("name", TASK_NAMES)
def test_task_lifecycle(name, task_dataset, tmp_path):
    """One pass through every Task hook the orchestrators call, minus the training loop."""
    import numpy as np

    from cvbench.core.config import build_config
    from cvbench.core.task import DatasetSpec

    data_dir = task_dataset(name)
    task = get_task(name)
    assert task.layout_errors(data_dir) == []
    assert task.matches_layout(data_dir)

    cfg = build_config(str(data_dir), overrides={
        "task": name, "model.backbone": "efficientnet_b0", "model.weights": "none",
        "training.epochs": 1, "data.batch_size": 2, "model.input_size": 64,
    })
    assert task.validate_config(cfg) == []

    spec = task.resolve_layout(cfg)
    assert isinstance(spec, DatasetSpec)
    assert bool(spec.class_names) == task.label_spec.has_classes
    assert cfg.model.num_classes == len(spec.class_names)
    assert spec.num_train_images == task.count_images(spec.train_dir) > 0

    train_ds, val_ds, num_train = task.build_datasets(cfg, spec)
    assert num_train == spec.num_train_images
    x, _ = next(iter(train_ds.take(1)))
    assert tuple(x.shape[1:3]) == (64, 64) and x.shape[0] <= 2
    assert next(iter(val_ds.take(1))) is not None

    model = task.build_model(cfg)
    assert model.input_shape[1:3] == (64, 64)
    outputs = model.predict(np.zeros((1, 64, 64, 3), dtype="float32"), verbose=0)
    assert np.all(np.isfinite(np.concatenate([np.ravel(o) for o in
                                              (outputs if isinstance(outputs, (list, tuple)) else [outputs])])))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_ds = task.build_eval_dataset(cfg, spec)
    report = task.evaluate(model, eval_ds, cfg, spec, str(run_dir), None)
    assert report["task"] == name and report["overall"]["metric"] == task.primary_metric.name
    assert (run_dir / "eval_report.json").is_file()

    metric, value = task.test_score(report)
    assert metric == task.primary_metric.name and 0.0 <= value <= 1.0

    headline = task.headline_metrics({"val_loss": 0.5, "val_accuracy": 0.25})
    assert headline["val_loss"] == 0.5
    if task.label_spec.has_classes:
        assert set(headline) == {"val_accuracy", "val_loss"}
    else:
        assert set(headline) <= {m.name for m in task.metrics_schema()}

    assert task.view().eval_summary(report) and task.view().per_class_table(report)["rows"] is not None


@pytest.mark.parametrize("name", TASK_NAMES)
def test_config_section_is_built_for_runs_of_its_task(name):
    from cvbench.core.config import CVBenchConfig

    section = get_task(name).config_section
    if section is not None:
        attr, cls = section
        assert isinstance(getattr(CVBenchConfig(task=name), attr), cls)


def test_registry_is_computed_once_per_process(monkeypatch):
    calls = []
    real = tasks_mod.entry_points
    monkeypatch.setattr(
        tasks_mod, "entry_points", lambda group: calls.append(group) or real(group=group)
    )
    tasks_mod._registry.cache_clear()
    tasks_mod.task_names()
    tasks_mod.task_names()
    assert len(calls) == 1
