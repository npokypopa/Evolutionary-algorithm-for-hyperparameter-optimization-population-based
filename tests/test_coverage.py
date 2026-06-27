"""Edge-case tests covering error paths, branches, and verbose output."""

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pbt_xgb import Hyperparameter, PBTXGBClassifier
from pbt_xgb.member import PopulationMember
from pbt_xgb.scoring import Scorer
from pbt_xgb.search_space import default_classification_space


def _data(seed=0, n_classes=2):
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


# -- search_space edge cases ------------------------------------------------
def test_hyperparameter_unknown_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        Hyperparameter("x", "weird", low=0, high=1)


def test_numeric_without_bounds_raises():
    with pytest.raises(ValueError, match="low and high"):
        Hyperparameter("x", "float")


def test_searchspace_iter_and_contains():
    space = default_classification_space()
    names = [hp.name for hp in space]  # exercises __iter__
    assert names == space.names
    assert "eta" in space  # __contains__
    assert "not_a_param" not in space


def test_perturb_skips_missing_names():
    space = default_classification_space()
    rng = np.random.default_rng(0)
    out = space.perturb({"eta": 0.1}, rng)  # only one of the space's genes present
    assert set(out) == {"eta"}


def test_mutate_rate_rounding_to_zero_is_noop():
    space = default_classification_space()  # 8 genes
    rng = np.random.default_rng(0)
    params = dict(space.sample(rng))
    out = space.mutate(params, rng, mutation_rate=0.05)  # round(0.4) == 0
    assert out == params


# -- scoring edge cases -----------------------------------------------------
def test_scorer_rejects_non_metric_type():
    with pytest.raises(TypeError):
        Scorer(123, num_class=2)


# -- member edge cases ------------------------------------------------------
def test_predict_proba_before_training_raises():
    member = PopulationMember(0, {})
    dmatrix = xgb.DMatrix(np.zeros((3, 4)))
    with pytest.raises(RuntimeError, match="not been trained"):
        member.predict_proba(dmatrix)


def test_clone_from_untrained_member_keeps_none_booster():
    donor = PopulationMember(0, {"eta": 0.123})  # never trained -> booster is None
    receiver = PopulationMember(1, {"eta": 0.999})
    receiver.clone_from(donor)
    assert receiver.booster is None
    assert receiver.params == {"eta": 0.123}


# -- pbt validation / branches ---------------------------------------------
def test_n_replace_negative_raises():
    with pytest.raises(ValueError, match="n_replace"):
        PBTXGBClassifier(n_replace=-1)


def test_single_class_target_raises():
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.zeros(20, dtype=int)  # only one class
    opt = PBTXGBClassifier(verbose=False)
    with pytest.raises(ValueError, match="two classes"):
        opt.fit(X, y, X, y)


def test_top_k_equals_population_skips_evolution():
    X_tr, X_va, y_tr, y_va = _data(seed=1)
    opt = PBTXGBClassifier(
        population_size=4,
        top_k=4,  # everyone is elite -> nothing replaced
        generations=2,
        step_rounds=5,
        random_state=1,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt._n_replace == 0
    assert opt.best_booster_ is not None


def test_improved_false_when_no_best_yet():
    opt = PBTXGBClassifier(verbose=False)
    opt.best_params_ = None
    assert opt._improved(-np.inf) is False


def test_minimize_metric_with_verbose_and_unreached_target(capsys):
    """Covers the minimize improvement branch, the 'target not reached' path,
    and verbose printing in one run."""
    X_tr, X_va, y_tr, y_va = _data(seed=2)
    opt = PBTXGBClassifier(
        population_size=4,
        generations=3,
        step_rounds=8,
        metric="log_loss",  # minimize
        target_score=-1.0,  # impossible for log_loss (>= 0) -> never reached
        random_state=2,
        verbose=True,  # exercise the print branches
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    out = capsys.readouterr().out
    assert "log_loss" in out
    assert "stopped after" in out
    assert opt.stop_reason_ == "max_generations"
    assert opt._scorer.maximize is False
