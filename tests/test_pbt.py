import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pbt_xgb import PBTXGBClassifier
from pbt_xgb.member import PopulationMember
from pbt_xgb.search_space import default_classification_space


def _binary_data(seed=0):
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=6,
        n_classes=2,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def test_fit_predict_binary_shapes_and_improvement():
    X_tr, X_va, y_tr, y_va = _binary_data()
    opt = PBTXGBClassifier(
        population_size=8,
        generations=5,
        step_rounds=10,
        metric="roc_auc",
        random_state=0,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)

    # predict shapes / values
    proba = opt.predict_proba(X_va)
    assert proba.shape == (len(y_va), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    preds = opt.predict(X_va)
    assert preds.shape == (len(y_va),)
    assert set(np.unique(preds)).issubset(set(np.unique(y_tr)))

    # PBT should hold or improve global best vs the first generation's best
    gen0 = opt.history_[0]["best_score"]
    assert opt.best_score_ >= gen0 - 1e-9
    assert len(opt.history_) == 5


def test_warm_start_grows_trees_each_generation():
    X_tr, X_va, y_tr, y_va = _binary_data(seed=1)
    opt = PBTXGBClassifier(
        population_size=4,
        generations=3,
        step_rounds=7,
        random_state=1,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    # best booster should have been grown incrementally to a multiple of step_rounds
    assert opt.best_num_trees_ % 7 == 0
    assert opt.best_num_trees_ >= 7


def test_multiclass_runs_and_labels_preserved():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=2,
    )
    # non-contiguous string-ish labels to exercise the LabelEncoder
    y = np.where(y == 0, 10, np.where(y == 1, 20, 30))
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
    opt = PBTXGBClassifier(
        population_size=6,
        generations=3,
        step_rounds=8,
        random_state=2,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    proba = opt.predict_proba(X_va)
    assert proba.shape == (len(y_va), 3)
    assert set(np.unique(opt.predict(X_va))).issubset({10, 20, 30})
    np.testing.assert_array_equal(opt.classes_, np.array([10, 20, 30]))


def test_clone_from_copies_booster_and_score():
    import xgboost as xgb

    X_tr, X_va, y_tr, y_va = _binary_data(seed=3)
    space = default_classification_space()
    rng = np.random.default_rng(3)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
    }

    donor = PopulationMember(0, space.sample(rng))
    donor.train_step(dtrain, base, 10)
    donor.score = 0.99

    receiver = PopulationMember(1, space.sample(rng))
    receiver.clone_from(donor)

    assert receiver.num_trees == donor.num_trees
    assert receiver.score == donor.score
    assert receiver.params == donor.params
    # cloned booster yields identical predictions
    dvalid = xgb.DMatrix(X_va)
    np.testing.assert_allclose(
        receiver.predict_proba(dvalid), donor.predict_proba(dvalid), rtol=1e-6
    )


def test_fitness_scored_on_validation_not_train():
    """A member is trained on the train set; its fitness must be the score on
    the validation set. We verify by scoring the same trained booster on both
    splits and confirming evaluate() returns the validation score."""
    import xgboost as xgb

    from pbt_xgb.scoring import Scorer

    X_tr, X_va, y_tr, y_va = _binary_data(seed=11)
    space = default_classification_space()
    rng = np.random.default_rng(11)
    base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
    }
    scorer = Scorer("roc_auc", num_class=2)

    member = PopulationMember(0, space.sample(rng))
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    member.train_step(dtrain, base, 30)

    # fitness as computed by the optimizer
    dvalid = xgb.DMatrix(X_va)
    fitness = member.evaluate(dvalid, y_va, scorer)

    # manual validation score must match exactly
    val_score = scorer.score(y_va, member.predict_proba(dvalid))
    assert fitness == pytest.approx(val_score)

    # the train-set score is different (and typically higher) -> fitness is NOT
    # measured on the training data
    train_score = scorer.score(y_tr, member.predict_proba(dtrain))
    assert fitness != pytest.approx(train_score)


def test_validation_set_drives_fitness():
    """Same training data, different validation labels -> different fitness,
    proving the validation split is what's scored."""
    import xgboost as xgb

    from pbt_xgb.scoring import Scorer

    X_tr, X_va, y_tr, y_va = _binary_data(seed=12)
    space = default_classification_space()
    rng = np.random.default_rng(12)
    base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "verbosity": 0,
    }
    scorer = Scorer("accuracy", num_class=2)

    member = PopulationMember(0, space.sample(rng))
    member.train_step(xgb.DMatrix(X_tr, label=y_tr), base, 30)

    dvalid = xgb.DMatrix(X_va)
    good = member.evaluate(dvalid, y_va, scorer)
    flipped = member.evaluate(dvalid, 1 - y_va, scorer)
    assert good != pytest.approx(flipped)


def test_save_results_writes_files(tmp_path):
    import csv
    import json

    import xgboost as xgb

    X_tr, X_va, y_tr, y_va = _binary_data(seed=13)
    opt = PBTXGBClassifier(
        population_size=6,
        generations=3,
        step_rounds=8,
        metric="roc_auc",
        random_state=13,
        verbose=False,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)

    out = opt.save_results(tmp_path / "results")
    assert (out / "best_model.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "history.csv").exists()

    # summary.json round-trips the key results
    summary = json.loads((out / "summary.json").read_text())
    assert summary["metric"] == "roc_auc"
    assert summary["best_score"] == pytest.approx(opt.best_score_)
    assert summary["best_params"] == opt.best_params_
    assert summary["n_generations"] == opt.n_generations_

    # history.csv has one row per generation
    with open(out / "history.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == opt.n_generations_
    assert {"generation", "best_score", "global_best_score"} <= set(rows[0])

    # the saved booster reproduces the optimizer's predictions
    reloaded = xgb.Booster(model_file=str(out / "best_model.json"))
    saved_proba = reloaded.predict(xgb.DMatrix(X_va))
    np.testing.assert_allclose(saved_proba, opt.predict_proba(X_va)[:, 1], rtol=1e-6)


def test_save_results_before_fit_raises(tmp_path):
    opt = PBTXGBClassifier(verbose=False)
    with pytest.raises(RuntimeError):
        opt.save_results(tmp_path / "results")


def test_predict_before_fit_raises():
    opt = PBTXGBClassifier(verbose=False)
    with pytest.raises(RuntimeError):
        opt.predict(np.zeros((2, 3)))


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        PBTXGBClassifier(population_size=1)
    with pytest.raises(ValueError):
        PBTXGBClassifier(selection_fraction=1.5)
