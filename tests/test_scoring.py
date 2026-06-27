import numpy as np
import pytest

from pbt_xgb.scoring import Scorer


def test_default_binary_is_roc_auc_maximize():
    s = Scorer(None, num_class=2)
    assert s.name == "roc_auc"
    assert s.maximize is True
    assert s.worst == -np.inf


def test_default_multiclass_is_neg_log_loss_maximize():
    s = Scorer(None, num_class=3)
    assert s.name == "neg_log_loss"
    assert s.maximize is True


def test_binary_roc_auc_perfect_separation():
    s = Scorer("roc_auc", num_class=2)
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert s.score(y, proba) == pytest.approx(1.0)


def test_accuracy_uses_hard_labels_from_proba():
    s = Scorer("accuracy", num_class=2)
    y = np.array([0, 1, 1, 0])
    proba = np.array([0.2, 0.9, 0.6, 0.4])  # -> 0,1,1,0
    assert s.score(y, proba) == pytest.approx(1.0)


def test_multiclass_accuracy_argmax():
    s = Scorer("accuracy", num_class=3)
    y = np.array([0, 1, 2])
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    assert s.score(y, proba) == pytest.approx(1.0)


def test_is_better_respects_direction():
    maxi = Scorer("roc_auc", num_class=2)
    assert maxi.is_better(0.9, 0.8)
    mini = Scorer("log_loss", num_class=2)
    assert mini.is_better(0.2, 0.5)
    assert mini.worst == np.inf


def test_custom_callable_requires_maximize():
    with pytest.raises(ValueError):
        Scorer(lambda y, p: 0.0, num_class=2)


def test_custom_callable_used():
    fn = lambda y, p: float(np.mean(np.asarray(y)))
    s = Scorer(fn, num_class=2, maximize=True)
    assert s.maximize is True
    assert s.score(np.array([0, 1, 1]), np.array([0.5, 0.5, 0.5])) == pytest.approx(2 / 3)


def test_unknown_metric_name_raises():
    with pytest.raises(ValueError):
        Scorer("not_a_metric", num_class=2)
