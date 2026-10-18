"""Conformance suite for the Task registry — the "adding a third task is easy" harness.

Runs over every name in ``cvbench.tasks.TASK_NAMES``: each must instantiate,
declare a unique ``name`` matching its registry key, and implement every
abstract method on ``Task``. A new task that forgets a method fails here
first, not at some far-off orchestration call site.
"""
import pytest

from cvbench.tasks import TASK_NAMES, get_task


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
    shorts = [get_task(n).short_name for n in TASK_NAMES]
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
    with pytest.raises(ValueError, match="--conf only applies to detection runs"):
        check_predict_options(cls, 0.4)
    with pytest.raises(ValueError, match="--top-k and --probs only apply to classification runs"):
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


def test_get_task_unknown_name_lists_valid_options():
    with pytest.raises(ValueError, match="Unknown task 'nonsense'"):
        get_task("nonsense")
    with pytest.raises(ValueError, match="classification"):
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
