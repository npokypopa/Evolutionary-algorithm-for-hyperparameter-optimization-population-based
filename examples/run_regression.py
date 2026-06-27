"""Demo: tune XGBoost regressors with Population Based Training.

Uses sklearn's California-housing-style synthetic data via ``make_regression``
to keep the example fully self-contained and seeded. A default-hyperparameter
XGBoost trained for the same number of rounds is shown as a baseline.

Run with:
    uv run python examples/run_regression.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pbt_xgb import PBTXGBRegressor

ROOT = Path(__file__).resolve().parent.parent


def baseline_val_r2(X_tr, y_tr, X_va, y_va, total_rounds):
    """Default-hyperparameter XGBoost trained for the same total rounds."""
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(np.asarray(X_tr), label=np.asarray(y_tr))
    booster = xgb.train(params, dtrain, num_boost_round=total_rounds)
    preds = booster.predict(xgb.DMatrix(np.asarray(X_va)))
    return r2_score(y_va, preds)


def main():
    print("=" * 70)
    print("REGRESSION: synthetic make_regression dataset")
    print("=" * 70)

    X, y = make_regression(
        n_samples=2000,
        n_features=20,
        n_informative=12,
        noise=15.0,
        random_state=42,
    )
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=0)
    print(
        f"train: {X_tr.shape[0]} rows x {X_tr.shape[1]} features  |  "
        f"validation: {X_va.shape[0]} rows"
    )

    opt = PBTXGBRegressor(
        population_size=12,
        generations=20,  # upper bound; early-stopping may end sooner
        step_rounds=20,
        metric="r2",  # None -> r2 (maximize) by default
        top_k=4,
        n_replace=4,
        crossover=True,
        dominance=0.8,
        mutation_rate=0.4,
        patience=4,
        max_time=120,
        random_state=0,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    print(f"stop reason             : {opt.stop_reason_} (after {opt.n_generations_} generations)")

    r2 = r2_score(y_va, opt.predict(X_va))
    base = baseline_val_r2(X_tr, y_tr, X_va, y_va, opt.best_num_trees_)

    print(f"\nbest val R2 (PBT)       : {opt.best_score_:.4f}")
    print(f"final val R2            : {r2:.4f}")
    print(f"baseline (default) R2   : {base:.4f}  ({opt.best_num_trees_} trees)")
    print(f"best params             : {opt.best_params_}")

    results_dir = opt.save_results(ROOT / "results_regression")
    print(f"\nresults saved to        : {results_dir}")
    print("  - best_model.json (trained booster)")
    print("  - summary.json    (best score + params + stop reason)")
    print("  - history.csv     (per-generation trace)")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
