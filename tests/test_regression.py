"""Tests for PBTXGBRegressor and the regression scoring path.

All data is real (sklearn ``make_regression``) and fully seeded; no mocks.
"""

import csv
import json

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from pbt_xgb import PBTXGBRegressor, default_regression_space
from pbt_xgb.scoring import Scorer
from pbt_xgb.search_space import default_classification_space


def _reg_data(seed=0, n_features=12):
    X, y = make_regression(
        n_samples=400,
        n_features=n_features,
        n_informative=8,
        noise=10.0,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.3, random_state=seed)


# -- search space -----------------------------------------------------------
def test_default_regression_space_matches_classification_genes():
    reg = default_regression_space()
    clf = default_classification_space()
    assert reg.names == clf.names
    assert "eta" in reg
    assert "max_depth" in reg


# -- scoring ----------------------------------------------------------------
def test_default_regression_metric_is_r2_maximize():
    s = Scorer(None, regression=True)
    assert s.name == "r2"
    assert s.maximize is True
    assert s.worst == -np.inf


def test_regression_rmse_is_minimize():
    s = Scorer("rmse", regression=True)
    assert s.maximize is False
    assert s.worst == np.inf
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 3.0])
    assert s.score(y, p) == pytest.approx(0.0)


def test_regression_neg_rmse_and_neg_mae_and_mse():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 1.0])
    assert Scorer("neg_rmse", regression=True).score(y, p) == pytest.approx(-1.0)
    assert Scorer("neg_mae", regression=True).score(y, p) == pytest.approx(-1.0)
    assert Scorer("mae", regression=True).score(y, p) == pytest.approx(1.0)
    assert Scorer("neg_mean_squared_error", regression=True).score(y, p) == pytest.approx(-1.0)
    assert Scorer("mse", regression=True).score(y, p) == pytest.approx(1.0)


def test_regression_r2_perfect():
    s = Scorer("r2", regression=True)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert s.score(y, y) == pytest.approx(1.0)


def test_regression_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        Scorer("roc_auc", regression=True)  # a classification metric


def test_regression_custom_callable_requires_maximize():
    with pytest.raises(ValueError):
        Scorer(lambda y, p: 0.0, regression=True)


# -- fit / predict ----------------------------------------------------------
def test_fit_predict_shapes_and_improvement():
    X_tr, X_va, y_tr, y_va = _reg_data()
    opt = PBTXGBRegressor(
        population_size=8,
        generations=5,
        step_rounds=10,
        random_state=0,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)

    preds = opt.predict(X_va)
    assert preds.shape == (len(y_va),)
    assert preds.dtype.kind == "f"  # continuous values
    assert not hasattr(opt, "predict_proba")

    # default metric is r2 / maximize
    assert opt._scorer.name == "r2"
    assert opt._scorer.maximize is True

    # PBT should hold or improve global best vs the first generation's best
    gen0 = opt.history_[0]["best_score"]
    assert opt.best_score_ >= gen0 - 1e-9
    assert len(opt.history_) == 5


def test_warm_start_grows_trees_each_generation():
    X_tr, X_va, y_tr, y_va = _reg_data(seed=1)
    opt = PBTXGBRegressor(
        population_size=4,
        generations=3,
        step_rounds=7,
        random_state=1,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt.best_num_trees_ % 7 == 0
    assert opt.best_num_trees_ >= 7


def test_minimize_metric_path_with_verbose(capsys):
    X_tr, X_va, y_tr, y_va = _reg_data(seed=2)
    opt = PBTXGBRegressor(
        population_size=4,
        generations=3,
        step_rounds=8,
        metric="rmse",  # minimize
        random_state=2,
        verbose=True,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    out = capsys.readouterr().out
    assert "rmse" in out
    assert "stopped after" in out
    assert opt._scorer.maximize is False
    assert opt.stop_reason_ == "max_generations"


def test_custom_callable_metric():
    X_tr, X_va, y_tr, y_va = _reg_data(seed=3)

    def neg_max_abs_error(y_true, y_pred):
        return -float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    opt = PBTXGBRegressor(
        population_size=4,
        generations=2,
        step_rounds=8,
        metric=neg_max_abs_error,
        maximize=True,
        random_state=3,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    assert opt._scorer.name == "neg_max_abs_error"
    assert opt.best_booster_ is not None


def test_save_results_writes_files(tmp_path):
    X_tr, X_va, y_tr, y_va = _reg_data(seed=4)
    opt = PBTXGBRegressor(
        population_size=6,
        generations=3,
        step_rounds=8,
        metric="r2",
        random_state=4,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)

    out = opt.save_results(tmp_path / "results")
    assert (out / "best_model.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "history.csv").exists()

    summary = json.loads((out / "summary.json").read_text())
    assert summary["metric"] == "r2"
    assert summary["best_score"] == pytest.approx(opt.best_score_)
    assert summary["best_params"] == opt.best_params_
    assert "classes" not in summary  # regressors have no classes

    with open(out / "history.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == opt.n_generations_

    # saved booster reproduces the optimizer's predictions
    reloaded = xgb.Booster(model_file=str(out / "best_model.json"))
    saved = reloaded.predict(xgb.DMatrix(X_va))
    np.testing.assert_allclose(saved, opt.predict(X_va), rtol=1e-6)


# -- error paths ------------------------------------------------------------
def test_predict_before_fit_raises():
    opt = PBTXGBRegressor(verbose=False)
    with pytest.raises(RuntimeError):
        opt.predict(np.zeros((2, 3)))


def test_save_results_before_fit_raises(tmp_path):
    opt = PBTXGBRegressor(verbose=False)
    with pytest.raises(RuntimeError):
        opt.save_results(tmp_path / "results")


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        PBTXGBRegressor(population_size=1)
    with pytest.raises(ValueError):
        PBTXGBRegressor(dominance=0.1)
