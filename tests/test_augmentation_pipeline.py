"""Augmentation pipeline: probabilities, one_of weighting, parameter sampling, tf-graph split."""
import random

import numpy as np
import pytest

pytestmark = pytest.mark.tf

from cvbench.augmentations import pipeline  # noqa: E402
from cvbench.core.config import OneOfCandidateConfig, OneOfConfig, TransformConfig  # noqa: E402

IMG = np.full((16, 16, 3), 100.0, dtype=np.float32)


def _t(name, prob=1.0, **params):
    return TransformConfig(name=name, prob=prob, params=params)


def _one_of(*candidates, prob=1.0):
    return OneOfConfig(prob=prob, candidates=[
        OneOfCandidateConfig(name=n, weight=w, params=p) for n, w, p in candidates
    ])


@pytest.fixture
def recorded(monkeypatch):
    """Replace the aug_* lookup with recorders so a test sees exactly which steps fired."""
    import cvbench.augmentations as aug_mod

    calls: list[tuple[str, dict]] = []

    def make(name):
        def fn(img, **kwargs):
            calls.append((name, kwargs))
            return img
        return fn

    for name in ("aug_a", "aug_b"):
        monkeypatch.setattr(aug_mod, name, make(name), raising=False)
    return calls


def test_prob_zero_never_fires_and_prob_one_always_does(recorded):
    never = pipeline.build_aug_pipeline([_t("aug_a", prob=0.0)])
    always = pipeline.build_aug_pipeline([_t("aug_b", prob=1.0)])

    for _ in range(20):
        never(IMG)
        always(IMG)

    assert [name for name, _ in recorded] == ["aug_b"] * 20


def test_steps_run_in_order_and_chain_their_output(recorded):
    out = pipeline.build_aug_pipeline([_t("aug_a"), _t("aug_b")])(IMG)

    assert [name for name, _ in recorded] == ["aug_a", "aug_b"]
    assert out.dtype == np.float32 and out.shape == IMG.shape


def test_intermediate_probabilities_fire_at_roughly_that_rate(recorded):
    fn = pipeline.build_aug_pipeline([_t("aug_a", prob=0.5)])

    for _ in range(400):
        fn(IMG)

    assert 140 < len(recorded) < 260


def test_one_of_picks_by_weight_and_never_a_zero_weight_candidate(recorded):
    fn = pipeline.build_aug_pipeline([_one_of(("aug_a", 0.0, {}), ("aug_b", 1.0, {}))])

    for _ in range(50):
        fn(IMG)

    assert {name for name, _ in recorded} == {"aug_b"}


def test_one_of_uses_every_candidate_over_many_draws(recorded):
    fn = pipeline.build_aug_pipeline([_one_of(("aug_a", 1.0, {}), ("aug_b", 1.0, {}))])

    for _ in range(100):
        fn(IMG)

    assert {name for name, _ in recorded} == {"aug_a", "aug_b"}


def test_one_of_respects_its_own_probability(recorded):
    fn = pipeline.build_aug_pipeline([_one_of(("aug_a", 1.0, {}), prob=0.0)])

    fn(IMG)

    assert recorded == []


def test_param_ranges_are_sampled_per_call(recorded):
    params = {"n": [2, 5], "x": [0.5, 1.5], "mode": ["l", "r", "c"], "flag": [True, False], "fixed": 7}
    fn = pipeline.build_aug_pipeline([_t("aug_a", **params)])

    for _ in range(60):
        fn(IMG)

    kwargs = [k for _, k in recorded]
    assert {k["n"] for k in kwargs} <= {2, 3, 4, 5} and len({k["n"] for k in kwargs}) > 1
    assert all(isinstance(k["n"], int) and 0.5 <= k["x"] <= 1.5 for k in kwargs)
    assert {k["mode"] for k in kwargs} == {"l", "r", "c"}
    assert {k["flag"] for k in kwargs} == {True, False}
    assert {k["fixed"] for k in kwargs} == {7}


def test_pipeline_decisions_are_reproducible_under_the_global_seed(recorded):
    transforms = [
        _t("aug_a", prob=0.6, n=[1, 9], x=[0.0, 1.0]),
        _one_of(("aug_a", 1.0, {"n": [1, 9]}), ("aug_b", 2.0, {})),
    ]

    def run():
        recorded.clear()
        random.seed(11)
        fn = pipeline.build_aug_pipeline(transforms)
        for _ in range(30):
            fn(IMG)
        return list(recorded)

    first = run()
    assert first and run() == first


def test_custom_fn_handles_single_images_and_batches():
    fn = pipeline.build_custom_aug_fn([_t("aug_blur", radius=1.0)])

    single = fn(IMG)
    batch = fn(np.stack([IMG, IMG]))

    assert single.shape == IMG.shape and single.dtype == np.float32
    assert batch.shape == (2, *IMG.shape) and batch.dtype == np.float32


def test_unknown_names_are_rejected_with_the_valid_options():
    with pytest.raises(ValueError, match="Unknown augmentation 'aug_nope'"):
        pipeline.build_aug_pipeline([_t("aug_nope")])
    with pytest.raises(ValueError, match="Unknown transform 'rotate_wildly'"):
        pipeline.build_aug_pipeline([_t("rotate_wildly")])


def test_custom_and_keras_builders_split_the_transform_list():
    mixed = [_t("keras_brightness"), _t("aug_blur", radius=1.0), _one_of(("aug_blur", 1.0, {"radius": 1.0}))]

    assert pipeline.build_custom_aug_fn([_t("keras_flip")]) is None
    assert pipeline.build_keras_aug_fn([_t("aug_blur", radius=1.0)]) is None
    assert pipeline.build_custom_aug_fn(mixed) is not None
    assert pipeline.build_keras_aug_fn(mixed) is not None
    assert pipeline.build_custom_aug_fn([_one_of(("keras_flip", 1.0, {}))]) is None


def test_keras_fn_runs_inside_a_tf_data_map_and_keeps_shapes():
    import tensorflow as tf

    fn = pipeline.build_keras_aug_fn([_t("keras_brightness", factor=0.3), _t("keras_noise", stddev=0.1)])
    ds = tf.data.Dataset.from_tensor_slices(np.stack([IMG] * 4)).batch(2).map(fn)

    batches = list(ds)

    assert [b.shape for b in batches] == [(2, 16, 16, 3)] * 2


def test_keras_steps_wrap_single_images_and_batches():
    fn = pipeline.build_aug_pipeline([_t("keras_noise", stddev=0.1)])

    assert fn(IMG).shape == IMG.shape
    assert fn(np.stack([IMG, IMG])).shape == (2, *IMG.shape)
