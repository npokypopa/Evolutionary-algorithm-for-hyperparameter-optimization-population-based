# pbt_xgb — Population Based Training for XGBoost

[![CI](https://github.com/npokypopa/Evolutionary-algorithm-for-hyperparameter-optimization-population-based/actions/workflows/ci.yml/badge.svg)](https://github.com/npokypopa/Evolutionary-algorithm-for-hyperparameter-optimization-population-based/actions/workflows/ci.yml)

Tune XGBoost hyperparameters with **Population Based Training (PBT)** / an
evolutionary strategy. A population of models is trained **incrementally** (each
generation appends boosting rounds to a warm-started booster); after every
generation the optimizer **selects** the fittest members, **crosses over** their
hyperparameters (dominant/recessive inheritance), and **mutates** a fraction of
the genes — stopping on the first terminal condition (`generations`,
`target_score`, `patience`, or `max_time`).

Supports **classification** (`PBTXGBClassifier`) and **regression**
(`PBTXGBRegressor`), which share one evolutionary engine.

## Install

```bash
uv sync --group dev
```

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pbt_xgb import PBTXGBClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

opt = PBTXGBClassifier(population_size=12, generations=20, step_rounds=20,
                       metric="roc_auc", random_state=0)
opt.fit(X_tr, y_tr, X_va, y_va)          # fitness measured on the validation set
print(opt.best_score_, opt.best_params_)
```

## Examples

```bash
uv run python examples/generate_data.py        # create the canonical dataset
uv run python examples/run_classification.py    # classification demo
uv run python examples/run_regression.py        # regression demo
uv run python examples/plot_history.py          # plot results/history.csv
```

A runnable walkthrough is in [`examples/demo.ipynb`](examples/demo.ipynb).

## Key features

- Warm-start incremental boosting (no retraining from scratch each generation).
- Configurable **selection** (`top_k`, `n_replace`), **crossover** (`dominance`),
  **mutation** (`mutation_rate`), and **stopping** (`patience`, `target_score`,
  `max_time`).
- Pluggable fitness metric (built-in names or a custom callable).
- `save_results(dir)` writes `summary.json`, `history.csv`, and `best_model.json`.
- 100% test coverage, ruff-linted, CI on every push.

See [`pbt_xgb/README.md`](pbt_xgb/README.md) for the full API.

## Development

```bash
uv run ruff check pbt_xgb tests examples
uv run ruff format --check pbt_xgb tests examples
uv run pytest -q          # runs with a 100% coverage gate
```
