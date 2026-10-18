"""The UI schema registry must stay in step with the ``aug_*`` functions it describes."""
import inspect
import json

import pytest

import cvbench.augmentations as aug_mod
from cvbench.augmentations.registry import _RANGES, _SKIP, get_schema

SCHEMA = get_schema()
BY_NAME = {entry["name"]: entry for entry in SCHEMA}


def test_every_range_entry_resolves_to_a_function():
    """get_schema() silently skips an unknown name, so a typo would hide an augmentation."""
    assert set(BY_NAME) == set(_RANGES)


def test_schema_is_json_serialisable_and_ordered_like_the_ranges():
    assert json.loads(json.dumps(SCHEMA)) == SCHEMA
    assert [e["name"] for e in SCHEMA] == list(_RANGES)


@pytest.mark.parametrize("name", list(_RANGES))
def test_entry_matches_the_function_signature(name):
    entry = BY_NAME[name]
    sig = inspect.signature(getattr(aug_mod, name))

    assert entry["label"] == _RANGES[name]["label"]
    assert [p["name"] for p in entry["params"]] == [p for p in sig.parameters if p not in _SKIP]
    assert not {"img", "seed"} & {p["name"] for p in entry["params"]}


@pytest.mark.parametrize("name", list(_RANGES))
def test_range_metadata_only_names_real_parameters(name):
    params = set(inspect.signature(getattr(aug_mod, name)).parameters)

    assert set(_RANGES[name]["params"]) <= params


@pytest.mark.parametrize("name", list(_RANGES))
def test_every_param_has_a_usable_type_and_default(name):
    for p in BY_NAME[name]["params"]:
        assert p["type"] in {"int", "float", "choice"}, p
        assert p["default"] is not None, f"{name}.{p['name']} has no default for the UI"
        if p["type"] == "choice":
            assert p["default"] in p["choices"], p
        else:
            assert {"min", "max", "step"} <= set(p), f"{name}.{p['name']} lacks a numeric range"
            assert p["min"] <= p["default"] <= p["max"], p
            assert p["min"] < p["max"] and p["step"] > 0


@pytest.mark.parametrize("name", list(_RANGES))
def test_defaults_actually_run_the_function(name):
    import numpy as np

    kwargs = {p["name"]: p["default"] for p in BY_NAME[name]["params"]}
    img = np.full((48, 48, 3), 100, dtype=np.uint8)

    out = getattr(aug_mod, name)(img, **kwargs)

    assert out.shape == img.shape


def test_param_types_follow_the_default():
    gamma = next(p for p in BY_NAME["aug_gamma"]["params"] if p["name"] == "gamma")
    n_lines = next(p for p in BY_NAME["aug_lines_h"]["params"] if p["name"] == "n_lines")
    side = next(p for p in BY_NAME["aug_fade_horizontal"]["params"] if p["name"] == "side")

    assert (gamma["type"], n_lines["type"], side["type"]) == ("float", "int", "choice")


def test_schema_endpoint_data_covers_the_new_style_augmentations():
    assert {"aug_net", "aug_chirp_artifacts", "aug_interference", "aug_mask_h", "aug_mask_v"} <= set(BY_NAME)
