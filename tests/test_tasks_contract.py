"""Conformance suite for the Task registry — the "adding a third task is easy" harness.

Runs over every name in ``cvbench.tasks.TASK_NAMES``: each must instantiate,
declare a unique ``name`` matching its registry key, and implement every
abstract method on ``Task``. A new task that forgets a method fails here
first, not at some far-off orchestration call site.
"""
import pytest

pytestmark = pytest.mark.tf  # get_task() imports a task module, which imports TensorFlow

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
def test_task_implements_every_abstract_method(name):
    """ABCs refuse to instantiate with unimplemented abstract methods —
    get_task() succeeding above is itself most of this proof. This test
    additionally confirms nothing overrides an abstract method with a
    non-callable, and that no abstract method is left on the base class name.
    """
    import inspect

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
