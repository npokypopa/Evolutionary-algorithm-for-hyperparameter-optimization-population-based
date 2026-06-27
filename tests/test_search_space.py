import numpy as np
import pytest

from pbt_xgb.search_space import (
    Hyperparameter,
    SearchSpace,
    default_classification_space,
)


def test_float_sample_within_bounds():
    rng = np.random.default_rng(0)
    hp = Hyperparameter("subsample", "float", low=0.5, high=1.0)
    for _ in range(200):
        v = hp.sample(rng)
        assert isinstance(v, float)
        assert 0.5 <= v <= 1.0


def test_int_sample_is_int_within_bounds():
    rng = np.random.default_rng(0)
    hp = Hyperparameter("max_depth", "int", low=2, high=10)
    for _ in range(200):
        v = hp.sample(rng)
        assert isinstance(v, int)
        assert 2 <= v <= 10


def test_log_sampling_spans_orders_of_magnitude():
    rng = np.random.default_rng(1)
    hp = Hyperparameter("eta", "float", low=1e-3, high=0.5, log=True)
    samples = np.array([hp.sample(rng) for _ in range(2000)])
    assert samples.min() >= 1e-3
    assert samples.max() <= 0.5
    # log-uniform => roughly half the mass below the geometric mean
    geo_mean = np.sqrt(1e-3 * 0.5)
    frac_below = np.mean(samples < geo_mean)
    assert 0.4 < frac_below < 0.6


def test_perturb_clips_to_bounds():
    rng = np.random.default_rng(2)
    hp = Hyperparameter("subsample", "float", low=0.5, high=1.0)
    # value at the ceiling: an upward factor must still clip to high
    for _ in range(100):
        assert hp.perturb(1.0, rng, factors=(1.2, 1.2)) == pytest.approx(1.0)
    # value at the floor: a downward factor must still clip to low
    for _ in range(100):
        assert hp.perturb(0.5, rng, factors=(0.8, 0.8)) == pytest.approx(0.5)


def test_categorical_resample_in_choices():
    rng = np.random.default_rng(3)
    hp = Hyperparameter("grow_policy", "categorical", choices=["depthwise", "lossguide"])
    for _ in range(50):
        assert hp.perturb("depthwise", rng) in ("depthwise", "lossguide")


def test_invalid_definitions_raise():
    with pytest.raises(ValueError):
        Hyperparameter("x", "float", low=1.0, high=0.0)  # high < low
    with pytest.raises(ValueError):
        Hyperparameter("x", "categorical")  # no choices
    with pytest.raises(ValueError):
        Hyperparameter("x", "float", low=0.0, high=1.0, log=True)  # log needs low>0


def test_searchspace_sample_and_perturb_keys():
    space = default_classification_space()
    rng = np.random.default_rng(4)
    params = space.sample(rng)
    assert set(params) == set(space.names)
    perturbed = space.perturb(params, rng)
    assert set(perturbed) == set(params)
    # passthrough keys not in the space are preserved
    params["objective"] = "binary:logistic"
    out = space.perturb(params, rng)
    assert out["objective"] == "binary:logistic"


def test_searchspace_rejects_duplicates():
    with pytest.raises(ValueError):
        SearchSpace([Hyperparameter("a", "float", 0, 1), Hyperparameter("a", "float", 0, 1)])
