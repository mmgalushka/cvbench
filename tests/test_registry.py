import pytest

from core.registry import _REGISTRY, register_aug, resolve_aug


@pytest.fixture(autouse=True)
def isolated_registry():
    """Each test gets a clean registry snapshot."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_register_and_resolve():
    @register_aug("test_aug")
    def build(params):
        return f"layer:{params}"

    result = resolve_aug("test_aug", {"k": 1})
    assert result == "layer:{'k': 1}"


def test_register_returns_original_function():
    def build(params):
        return params

    decorated = register_aug("identity")(build)
    assert decorated is build


def test_resolve_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="not registered"):
        resolve_aug("does_not_exist", {})


def test_resolve_error_lists_available():
    @register_aug("aug_a")
    def build_a(params):
        return None

    with pytest.raises(KeyError, match="aug_a"):
        resolve_aug("missing", {})


def test_overwrite_registration():
    @register_aug("replaceable")
    def v1(params):
        return "v1"

    @register_aug("replaceable")
    def v2(params):
        return "v2"

    assert resolve_aug("replaceable", {}) == "v2"
