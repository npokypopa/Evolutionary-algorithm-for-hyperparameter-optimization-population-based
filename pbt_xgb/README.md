# pbt_xgb — Population Based Training for XGBoost

A small library that tunes XGBoost hyperparameters with **Population Based
Training (PBT)**. A population of models is trained *incrementally*; each
generation the weakest members **exploit** the strongest (copying their booster
state *and* hyperparameters) and then **explore** (perturbing those
hyperparameters), before training continues.

Because gradient boosting is additive, each PBT generation simply appends more
trees to a checkpointed `Booster` (via `xgb.train(..., xgb_model=...)`) instead
of retraining from scratch — which is what makes PBT efficient here.

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pbt_xgb import PBTXGBClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=10)

opt = PBTXGBClassifier(
    population_size=12,
    generations=20,
    step_rounds=20,
    metric="roc_auc",      # None -> task default (AUC binary / neg_log_loss multiclass)
    random_state=10,
)
opt.fit(X_tr, y_tr, X_va, y_va)   # explicit train + validation sets

print(opt.best_score_, opt.best_params_)
proba = opt.predict_proba(X_va)   # (n, n_classes)
preds = opt.predict(X_va)         # original class labels
```

Run the full demo: `uv run python examples/run_classification.py`.

## Regression

`PBTXGBRegressor` shares all of the evolutionary machinery (both classes
subclass an internal `_BasePBT`) and the same constructor signature. It targets
continuous values instead of classes:

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from pbt_xgb import PBTXGBRegressor

X, y = make_regression(n_samples=2000, n_features=20, noise=15.0, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=0)

opt = PBTXGBRegressor(
    population_size=12,
    generations=20,
    step_rounds=20,
    metric="r2",          # None -> r2 (maximize) by default
    random_state=0,
)
opt.fit(X_tr, y_tr, X_va, y_va)

print(opt.best_score_, opt.best_params_)
preds = opt.predict(X_va)   # continuous values (no predict_proba)
```

Differences from the classifier:

- Fixed params: `objective="reg:squarederror"`, `eval_metric="rmse"`, no
  `num_class`, no `LabelEncoder` (targets stay continuous floats).
- `predict(X)` returns continuous values; there is **no** `predict_proba` and no
  `classes_` attribute.
- Built-in metrics: `r2` (maximize, default), `neg_rmse`/`rmse`,
  `neg_mae`/`mae`, `neg_mean_squared_error`/`mse`. The scorer receives the raw
  1-D predictions.
- Default search space: `default_regression_space()` (identical objective-agnostic
  per-tree genes as the classification space).

Run the full demo: `uv run python examples/run_regression.py`.

## Design

| Aspect | Choice |
|--------|--------|
| Algorithm | Warm-start incremental PBT/GA (continue boosting each generation) |
| Tasks | Classification (binary + multiclass) and regression, sharing one `_BasePBT` engine |
| Fitness | Pluggable metric: `None` (default), built-in name, or `fn(y_true, y_pred)` + `maximize` |
| Data | Explicit `fit(X_train, y_train, X_val, y_val)`; fitness = held-out validation score |
| Selection | Keep `top_k` fittest; replace `n_replace` worst each generation |
| Crossover | Dominant/recessive: gene from fitter parent w.p. `dominance`, else recessive; child warm-starts from the dominant parent's booster |
| Mutation | Perturb a `mutation_rate` fraction of a child's hyperparameters |
| Stopping | First of: `generations`, `target_score`, `patience` (no improvement), `max_time` |
| Parallelism | Members trained sequentially; XGBoost uses all cores (`n_jobs=-1`) |

## Configurable controls

```python
PBTXGBClassifier(
    # population / training
    population_size=12, generations=20, step_rounds=20,
    # fitness
    metric="roc_auc", maximize=None,
    # selection
    top_k=4,                 # elite/parent pool (None -> selection_fraction)
    selection_fraction=0.25, # used only when top_k is None
    n_replace=4,             # worst members replaced per gen (None -> top_k)
    # crossover (dominant / recessive inheritance)
    crossover=True,
    dominance=0.8,           # P(gene from the fitter parent), in [0.5, 1.0]
    # mutation
    mutation_rate=0.4,       # fraction of a child's genes to perturb, [0, 1]
    perturb_factors=(0.8, 1.2),
    # terminal conditions
    target_score=0.999,      # stop when global-best reaches this
    patience=4,              # stop after N generations without improvement
    tol=1e-4,                # min change counted as improvement
    max_time=60,             # wall-clock budget (seconds)
    # runtime
    tree_method="hist", random_state=0, n_jobs=-1, verbose=True,
)
```

After `fit`, `stop_reason_` reports which terminal condition fired
(`"max_generations"`, `"target_score"`, `"patience"`, or `"max_time"`) and
`n_generations_` how many generations ran.

### Key attributes after `fit`
- `best_params_` — hyperparameters of the best member
- `best_score_` — best validation fitness
- `best_booster_` — the trained `xgboost.Booster` used for prediction
- `best_num_trees_` — number of boosting rounds in the best booster
- `history_` — per-generation `best`/`mean`/`worst`/`global_best` scores
- `classes_` — original class labels (classifier only)

## Search space

`default_classification_space()` and `default_regression_space()` evolve the same
objective-agnostic per-tree genes: `eta`, `max_depth`, `min_child_weight`,
`gamma`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`. Task-defining
parameters (`objective`, `num_class`, `eval_metric`, `tree_method`) are held
fixed across the run. Build a custom space with `SearchSpace` + `Hyperparameter`.

## Tests

`uv run pytest tests -q`
