"""Tests for the configurable evolutionary controls: selection, crossover,
mutation, and terminal conditions."""

import time

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pbt_xgb import PBTXGBClassifier
from pbt_xgb.search_space import default_classification_space


def _data(seed=0, n_classes=2):
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=6,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


# -- mutation rate ----------------------------------------------------------
def test_mutation_rate_zero_changes_nothing():
    space = default_classification_space()
    rng = np.random.default_rng(0)
    params = space.sample(rng)
    out = space.mutate(params, rng, mutation_rate=0.0)
    assert out == params


def test_mutation_rate_count_is_fraction_of_genes():
    space = default_classification_space()
    rng = np.random.default_rng(1)
    params = space.sample(rng)
    n_genes = len(space)
    out = space.mutate(params, rng, mutation_rate=0.5)
    changed = sum(1 for k in params if out[k] != params[k])
    # exactly round(0.5 * n_genes) genes targeted (a perturb could, rarely,
    # land on the same clipped value, so allow <=)
    assert changed <= round(0.5 * n_genes)
    assert changed >= 1


def test_mutation_rate_one_targets_all():
    space = default_classification_space()
    rng = np.random.default_rng(2)
    params = dict(space.sample(rng))
    # use upward-only factors so every numeric gene visibly changes (unless clipped)
    out = space.mutate(params, rng, mutation_rate=1.0, factors=(1.2, 1.2))
    changed = sum(1 for k in params if out[k] != params[k])
    assert changed >= len(space) - 2  # allow a couple already at the ceiling


# -- selection / top_k ------------------------------------------------------
def test_top_k_and_n_replace_resolved():
    X_tr, X_va, y_tr, y_va = _data()
    opt = PBTXGBClassifier(
        population_size=10,
        generations=2,
        step_rounds=5,
        top_k=3,
        n_replace=5,
        random_state=0,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt._top_k == 3
    assert opt._n_replace == 5


def test_n_replace_capped_to_non_elite():
    opt = PBTXGBClassifier(population_size=6, top_k=4, n_replace=10, random_state=0, verbose=False)
    X_tr, X_va, y_tr, y_va = _data()
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt._n_replace == 2  # only 6 - 4 = 2 non-elite members


def test_default_top_k_from_selection_fraction():
    opt = PBTXGBClassifier(
        population_size=12, selection_fraction=0.25, random_state=0, verbose=False
    )
    X_tr, X_va, y_tr, y_va = _data()
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt._top_k == 3


# -- crossover / dominance --------------------------------------------------
def test_crossover_dominance_one_takes_all_from_dominant():
    opt = PBTXGBClassifier(dominance=1.0, verbose=False)
    rng = np.random.default_rng(0)
    dom = {"eta": 0.1, "max_depth": 5}
    rec = {"eta": 0.9, "max_depth": 9}
    opt.search_space = default_classification_space()
    out = opt._crossover(dom, rec, rng)
    assert out["eta"] == dom["eta"]
    assert out["max_depth"] == dom["max_depth"]


def test_crossover_mixes_genes_with_intermediate_dominance():
    opt = PBTXGBClassifier(dominance=0.5, verbose=False)
    opt.search_space = default_classification_space()
    rng = np.random.default_rng(3)
    dom = opt.search_space.sample(np.random.default_rng(1))
    rec = opt.search_space.sample(np.random.default_rng(2))
    # over many draws, at least one gene should come from each parent
    from_dom = from_rec = 0
    for _ in range(50):
        child = opt._crossover(dom, rec, rng)
        for name in opt.search_space.names:
            if child[name] == dom[name]:
                from_dom += 1
            elif child[name] == rec[name]:
                from_rec += 1
    assert from_dom > 0 and from_rec > 0


def test_crossover_disabled_falls_back_to_clone():
    X_tr, X_va, y_tr, y_va = _data(seed=4)
    opt = PBTXGBClassifier(
        population_size=6,
        generations=3,
        step_rounds=6,
        crossover=False,
        mutation_rate=0.3,
        random_state=4,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt.best_booster_ is not None


# -- terminal conditions ----------------------------------------------------
def test_target_score_stops_early():
    X_tr, X_va, y_tr, y_va = _data(seed=5)
    opt = PBTXGBClassifier(
        population_size=6,
        generations=50,
        step_rounds=10,
        metric="roc_auc",
        target_score=0.5,  # trivially reached on gen 0
        random_state=5,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt.stop_reason_ == "target_score"
    assert opt.n_generations_ == 1


def test_patience_stops_when_no_improvement():
    X_tr, X_va, y_tr, y_va = _data(seed=6)
    opt = PBTXGBClassifier(
        population_size=6,
        generations=50,
        step_rounds=5,
        patience=2,
        tol=10.0,  # huge tol => nothing ever counts as improvement after gen 0
        random_state=6,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt.stop_reason_ == "patience"
    assert opt.n_generations_ <= 4


def test_max_time_stops_run():
    X_tr, X_va, y_tr, y_va = _data(seed=7)
    opt = PBTXGBClassifier(
        population_size=8,
        generations=1000,
        step_rounds=20,
        max_time=0.001,  # effectively immediate
        random_state=7,
        verbose=False,
    )
    t0 = time.perf_counter()
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt.stop_reason_ == "max_time"
    assert time.perf_counter() - t0 < 30


def test_runs_all_generations_when_no_early_stop():
    X_tr, X_va, y_tr, y_va = _data(seed=8)
    opt = PBTXGBClassifier(
        population_size=6,
        generations=4,
        step_rounds=5,
        random_state=8,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt.stop_reason_ == "max_generations"
    assert opt.n_generations_ == 4


# -- validation -------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"dominance": 0.3},
        {"dominance": 1.1},
        {"mutation_rate": -0.1},
        {"mutation_rate": 1.5},
        {"top_k": 0},
        {"top_k": 99},  # > population_size (default 12)
        {"patience": 0},
        {"max_time": 0},
        {"selection_fraction": 0.0},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        PBTXGBClassifier(verbose=False, **kwargs)
