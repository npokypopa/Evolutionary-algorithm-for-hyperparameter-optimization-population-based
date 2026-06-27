"""A single member of the PBT population.

Each :class:`PopulationMember` owns an XGBoost :class:`~xgboost.Booster` that is
grown *incrementally*: every PBT generation appends ``step_rounds`` more trees
to the existing booster via ``xgb.train(..., xgb_model=self.booster)``. This is
what makes PBT cheap for gradient boosting — we never retrain from scratch.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import xgboost as xgb

from .scoring import Scorer

__all__ = ["PopulationMember"]


class PopulationMember:
    def __init__(self, member_id: int, params: Mapping[str, Any]):
        self.id = member_id
        self.params: dict[str, Any] = dict(params)
        self.booster: xgb.Booster | None = None
        self.num_trees: int = 0
        self.score: float = float("nan")

    # -- training -----------------------------------------------------------
    def train_step(
        self,
        dtrain: xgb.DMatrix,
        base_params: Mapping[str, Any],
        step_rounds: int,
    ) -> None:
        """Append ``step_rounds`` boosting rounds using the current hyperparams.

        The fixed ``base_params`` (objective, num_class, eval_metric, ...) are
        merged with this member's evolved hyperparameters. Evolved per-tree
        parameters take effect for the newly added trees only — expected PBT
        behavior for already-built trees.
        """
        full_params = {**base_params, **self.params}
        self.booster = xgb.train(
            full_params,
            dtrain,
            num_boost_round=step_rounds,
            xgb_model=self.booster,
        )
        self.num_trees += step_rounds

    # -- evaluation ---------------------------------------------------------
    def predict_raw(self, dmatrix: xgb.DMatrix) -> np.ndarray:
        """Raw booster output: probabilities for classifiers, values for regressors."""
        if self.booster is None:
            raise RuntimeError("Member has not been trained yet.")
        return np.asarray(self.booster.predict(dmatrix))

    # Backwards-compatible alias: the raw output of a classifier booster *is*
    # the probability array.
    predict_proba = predict_raw

    def evaluate(
        self,
        dvalid: xgb.DMatrix,
        y_valid: np.ndarray,
        scorer: Scorer,
    ) -> float:
        self.score = scorer.score(y_valid, self.predict_raw(dvalid))
        return self.score

    # -- exploit ------------------------------------------------------------
    def clone_from(self, other: PopulationMember) -> None:
        """Copy another member's booster state *and* hyperparameters.

        This is the PBT "exploit" step: a weak member adopts a stronger
        member's trees and configuration before exploring (perturbing).
        """
        self.params = copy.deepcopy(other.params)
        self.num_trees = other.num_trees
        self.score = other.score
        if other.booster is None:
            self.booster = None
        else:
            raw = other.booster.save_raw()
            cloned = xgb.Booster()
            cloned.load_model(bytearray(raw))
            self.booster = cloned
