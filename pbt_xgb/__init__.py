"""pbt_xgb: Population Based Training for XGBoost classifiers and regressors."""

from __future__ import annotations

from .pbt import PBTXGBClassifier, PBTXGBRegressor
from .scoring import Scorer
from .search_space import (
    Hyperparameter,
    SearchSpace,
    default_classification_space,
    default_regression_space,
)

__all__ = [
    "PBTXGBClassifier",
    "PBTXGBRegressor",
    "SearchSpace",
    "Hyperparameter",
    "default_classification_space",
    "default_regression_space",
    "Scorer",
]

__version__ = "0.1.0"
