"""Demo: tune XGBoost classifiers with Population Based Training.

The primary run uses the project's canonical dataset (``data/train.csv`` and
``data/validation.csv`` — the full 1000 train / 250 validation split produced by
``generate_data.py``). A second run on sklearn's multiclass ``wine`` dataset
shows the optimizer generalizes beyond the binary case.

Run with:
    uv run python examples/generate_data.py     # once, to create data/*.csv
    uv run python examples/run_classification.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pbt_xgb import PBTXGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def baseline_val_score(X_tr, y_tr, X_va, y_va, num_class, metric_fn, total_rounds):
    """Default-hyperparameter XGBoost trained for the same total rounds."""
    params = {"tree_method": "hist", "verbosity": 0}
    if num_class == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = num_class
    dtrain = xgb.DMatrix(np.asarray(X_tr), label=np.asarray(y_tr))
    booster = xgb.train(params, dtrain, num_boost_round=total_rounds)
    proba = booster.predict(xgb.DMatrix(np.asarray(X_va)))
    return metric_fn(y_va, proba)


def run_repo_binary():
    print("=" * 70)
    print("BINARY: project canonical dataset (data/train.csv + data/validation.csv)")
    print("=" * 70)
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"
    if not (train_path.exists() and val_path.exists()):
        raise SystemExit("Canonical data not found. Run:  uv run python examples/generate_data.py")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    X_tr = train_df.drop(columns=["target"]).to_numpy()
    y_tr = train_df["target"].to_numpy()
    X_va = val_df.drop(columns=["target"]).to_numpy()
    y_va = val_df["target"].to_numpy()
    # Use the COMPLETE dataset — every training and validation row, all features.
    print(
        f"train: {X_tr.shape[0]} rows x {X_tr.shape[1]} features  |  "
        f"validation: {X_va.shape[0]} rows"
    )

    opt = PBTXGBClassifier(
        population_size=12,
        generations=20,  # upper bound; early-stopping may end sooner
        step_rounds=20,
        metric="roc_auc",
        top_k=4,
        n_replace=4,
        crossover=True,
        dominance=0.8,
        mutation_rate=0.4,
        patience=4,
        target_score=0.999,
        max_time=120,
        random_state=0,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)
    print(f"stop reason             : {opt.stop_reason_} (after {opt.n_generations_} generations)")

    proba_pos = opt.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, proba_pos)
    acc = accuracy_score(y_va, opt.predict(X_va))
    base = baseline_val_score(X_tr, y_tr, X_va, y_va, 2, roc_auc_score, opt.best_num_trees_)

    print(f"\nbest val AUC (PBT)      : {opt.best_score_:.4f}")
    print(f"final val AUC           : {auc:.4f}")
    print(f"final val accuracy      : {acc:.4f}")
    print(f"baseline (default) AUC  : {base:.4f}  ({opt.best_num_trees_} trees)")
    print(f"best params             : {opt.best_params_}")

    results_dir = opt.save_results(ROOT / "results")
    print(f"\nresults saved to        : {results_dir}")
    print("  - best_model.json (trained booster)")
    print("  - summary.json    (best score + params + stop reason)")
    print("  - history.csv     (per-generation trace)")


def run_multiclass():
    print("\n" + "=" * 70)
    print("MULTICLASS: wine (3 classes)")
    print("=" * 70)
    data = load_wine()
    X, y = data.data, data.target
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    opt = PBTXGBClassifier(
        population_size=10,
        generations=6,
        step_rounds=15,
        metric=None,  # default: neg_log_loss for multiclass
        random_state=0,
    )
    opt.fit(X_tr, y_tr, X_va, y_va)

    acc = accuracy_score(y_va, opt.predict(X_va))
    print(f"\nbest val neg_log_loss   : {opt.best_score_:.4f}")
    print(f"final val accuracy      : {acc:.4f}")
    print(f"predict_proba shape     : {opt.predict_proba(X_va).shape}")
    print(f"best params             : {opt.best_params_}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    run_repo_binary()
    run_multiclass()
