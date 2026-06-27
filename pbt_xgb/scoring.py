"""Fitness scoring for Population Based Training.

A :class:`Scorer` turns an XGBoost probability output and the true labels into a
single fitness number, together with a ``maximize`` flag so the optimizer knows
which direction is "better".

It accepts three forms of metric specification:

* ``None``  – pick a sensible default for the task (ROC AUC for binary,
  log-loss for multiclass).
* ``str``   – a built-in metric name (see :data:`_BUILTIN_METRICS`).
* callable  – a user function ``fn(y_true, y_proba) -> float``; pair it with the
  ``maximize`` flag passed to :class:`~pbt_xgb.pbt.PBTXGBClassifier`.

The probability array handed to a scorer is normalized: shape ``(n,)`` giving
``P(class == 1)`` for binary problems, or shape ``(n, n_classes)`` for
multiclass problems.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

__all__ = ["Scorer"]


def _to_labels(proba: np.ndarray) -> np.ndarray:
    """Convert normalized probabilities to hard class predictions."""
    if proba.ndim == 1:
        return (proba >= 0.5).astype(int)
    return np.argmax(proba, axis=1)


# name -> (fn(y_true, proba) -> float, maximize)
# Each fn receives the *normalized* probability array described in the module
# docstring and is responsible for reducing it as needed.
_BUILTIN_METRICS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], float], bool]] = {
    "roc_auc": (
        lambda y, p: roc_auc_score(y, p if p.ndim == 1 else p, multi_class="ovr"),
        True,
    ),
    "average_precision": (lambda y, p: average_precision_score(y, p), True),
    "log_loss": (lambda y, p: log_loss(y, p), False),
    "neg_log_loss": (lambda y, p: -log_loss(y, p), True),
    "accuracy": (lambda y, p: accuracy_score(y, _to_labels(p)), True),
    "balanced_accuracy": (lambda y, p: balanced_accuracy_score(y, _to_labels(p)), True),
    "f1": (lambda y, p: f1_score(y, _to_labels(p)), True),
    "f1_macro": (lambda y, p: f1_score(y, _to_labels(p), average="macro"), True),
}


# name -> (fn(y_true, y_pred) -> float, maximize)
# Each fn receives the raw 1-D regression predictions produced by the booster.
_REGRESSION_METRICS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], float], bool]] = {
    "r2": (lambda y, p: r2_score(y, p), True),
    "neg_rmse": (lambda y, p: -np.sqrt(mean_squared_error(y, p)), True),
    "rmse": (lambda y, p: np.sqrt(mean_squared_error(y, p)), False),
    "neg_mean_squared_error": (lambda y, p: -mean_squared_error(y, p), True),
    "mse": (lambda y, p: mean_squared_error(y, p), False),
    "neg_mae": (lambda y, p: -mean_absolute_error(y, p), True),
    "mae": (lambda y, p: mean_absolute_error(y, p), False),
}


class Scorer:
    """Resolve a metric specification into a callable fitness function.

    The ``regression`` flag switches between the classification built-ins
    (keyed off ``num_class``) and the regression built-ins. Regression scorers
    receive the raw 1-D predictions from the booster instead of probabilities.
    """

    def __init__(
        self,
        metric: str | Callable | None,
        num_class: int = 0,
        maximize: bool | None = None,
        regression: bool = False,
    ):
        self.num_class = num_class
        self.regression = regression
        builtins = _REGRESSION_METRICS if regression else _BUILTIN_METRICS

        if metric is None:
            name = "r2" if regression else ("roc_auc" if num_class <= 2 else "neg_log_loss")
            self._fn, self.maximize = builtins[name]
            self.name = name
        elif isinstance(metric, str):
            if metric not in builtins:
                raise ValueError(f"Unknown metric {metric!r}. Known: {sorted(builtins)}")
            self._fn, default_max = builtins[metric]
            self.maximize = default_max if maximize is None else bool(maximize)
            self.name = metric
        elif callable(metric):
            if maximize is None:
                raise ValueError(
                    "When passing a custom metric callable you must also pass "
                    "maximize=True or maximize=False."
                )
            self._fn = metric
            self.maximize = bool(maximize)
            self.name = getattr(metric, "__name__", "custom")
        else:
            raise TypeError(f"Unsupported metric type: {type(metric)!r}")

    @property
    def worst(self) -> float:
        """The worst possible score, used to seed best-so-far tracking."""
        return -np.inf if self.maximize else np.inf

    def is_better(self, a: float, b: float) -> bool:
        """Return True if score ``a`` is strictly better than score ``b``."""
        return a > b if self.maximize else a < b

    def score(self, y_true: np.ndarray, predictions: np.ndarray) -> float:
        """Score true targets against model output (probabilities or raw values)."""
        return float(self._fn(np.asarray(y_true), np.asarray(predictions)))
