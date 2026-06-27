"""Population Based / evolutionary training optimizers for XGBoost.

The evolutionary machinery (generation loop, truncation selection,
dominant/recessive crossover, partial mutation, terminal conditions,
result persistence) lives on :class:`_BasePBT`. Two thin subclasses specialize
it for the two supported tasks:

* :class:`PBTXGBClassifier` — classification (binary + multiclass);
* :class:`PBTXGBRegressor`  — regression (continuous targets).

Each generation grows every member incrementally and periodically applies an
evolutionary step:

* **selection** – keep the ``top_k`` fittest members as the parent/elite pool;
* **crossover** – build each replacement child from two parents, inheriting each
  hyperparameter from the *dominant* (fitter) parent with probability
  ``dominance`` and from the *recessive* parent otherwise; the child warm-starts
  from the dominant parent's booster;
* **mutation** – perturb a ``mutation_rate`` fraction of the child's
  hyperparameters.

Training stops on the first satisfied terminal condition: ``generations`` reached,
``target_score`` hit, no improvement for ``patience`` generations, or ``max_time``
seconds elapsed.
"""

from __future__ import annotations

import csv
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from .member import PopulationMember
from .scoring import Scorer
from .search_space import (
    SearchSpace,
    default_classification_space,
    default_regression_space,
)

__all__ = ["PBTXGBClassifier", "PBTXGBRegressor"]

ArrayLike = Any


class _BasePBT(ABC):
    """Shared evolutionary optimizer; subclassed per task.

    Population / training
    ---------------------
    population_size:
        Number of models evolved in parallel.
    step_rounds:
        Boosting rounds added per member per generation (warm-started).
    search_space:
        Hyperparameters to evolve. Defaults to the task-appropriate space.

    Fitness
    -------
    metric:
        ``None`` (task default), a built-in name, or a callable
        ``fn(y_true, y_pred) -> float`` (then ``maximize`` is required).
    maximize:
        Direction for a custom/overridden metric.

    Selection
    ---------
    top_k:
        Number of fittest members kept as the elite/parent pool each generation.
        If ``None``, derived as ``round(selection_fraction * population_size)``.
    selection_fraction:
        Used only when ``top_k`` is ``None``.
    n_replace:
        Number of worst (non-elite) members replaced by new children each
        generation. If ``None``, defaults to ``top_k`` (classic truncation).

    Crossover (dominant / recessive inheritance)
    --------------------------------------------
    crossover:
        If ``True``, children inherit genes from two parents; if ``False``, a
        child clones a single random elite member (pure PBT exploit).
    dominance:
        Probability in ``[0.5, 1.0]`` that a gene is taken from the dominant
        (fitter) parent rather than the recessive one.

    Mutation
    --------
    mutation_rate:
        Fraction of a child's hyperparameters to perturb after crossover
        (``0.0``–``1.0``).
    perturb_factors:
        Multiplicative factors applied when mutating a numeric hyperparameter.

    Terminal conditions
    -------------------
    generations:
        Maximum number of generations (always applies).
    target_score:
        Stop once the global-best fitness reaches this value.
    patience:
        Stop if the global-best fitness has not improved (by more than ``tol``)
        for this many consecutive generations. ``None`` disables it.
    tol:
        Minimum change counted as an improvement for ``patience``.
    max_time:
        Wall-clock budget in seconds. ``None`` disables it.

    Runtime
    -------
    tree_method, random_state, n_jobs, verbose:
        XGBoost tree algorithm (fixed), RNG seed, XGBoost threads (members run
        sequentially), and progress printing.

    Attributes set after :meth:`fit`
    --------------------------------
    best_params_, best_score_, best_booster_, best_num_trees_, history_,
    n_generations_, stop_reason_.
    """

    def __init__(
        self,
        population_size: int = 12,
        generations: int = 10,
        step_rounds: int = 25,
        search_space: SearchSpace | None = None,
        # fitness
        metric: str | Callable | None = None,
        maximize: bool | None = None,
        # selection
        top_k: int | None = None,
        selection_fraction: float = 0.25,
        n_replace: int | None = None,
        # crossover
        crossover: bool = True,
        dominance: float = 0.75,
        # mutation
        mutation_rate: float = 0.5,
        perturb_factors: tuple[float, float] = (0.8, 1.2),
        # terminal conditions
        target_score: float | None = None,
        patience: int | None = None,
        tol: float = 1e-4,
        max_time: float | None = None,
        # runtime
        tree_method: str = "hist",
        random_state: int | None = None,
        n_jobs: int = -1,
        verbose: bool = True,
    ):
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if not 0.0 < selection_fraction <= 1.0:
            raise ValueError("selection_fraction must be in (0, 1]")
        if top_k is not None and not 1 <= top_k <= population_size:
            raise ValueError("top_k must be in [1, population_size]")
        if n_replace is not None and n_replace < 0:
            raise ValueError("n_replace must be >= 0")
        if not 0.5 <= dominance <= 1.0:
            raise ValueError("dominance must be in [0.5, 1.0]")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0.0, 1.0]")
        if patience is not None and patience < 1:
            raise ValueError("patience must be >= 1")
        if max_time is not None and max_time <= 0:
            raise ValueError("max_time must be > 0")

        self.population_size = population_size
        self.generations = generations
        self.step_rounds = step_rounds
        self.search_space = search_space or self._default_search_space()
        self.metric = metric
        self.maximize = maximize
        self.top_k = top_k
        self.selection_fraction = selection_fraction
        self.n_replace = n_replace
        self.crossover = crossover
        self.dominance = dominance
        self.mutation_rate = mutation_rate
        self.perturb_factors = perturb_factors
        self.target_score = target_score
        self.patience = patience
        self.tol = tol
        self.max_time = max_time
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # -- task-specific hooks (implemented by subclasses) --------------------
    @abstractmethod
    def _default_search_space(self) -> SearchSpace:
        """The task's default search space when none is supplied."""

    @abstractmethod
    def _prepare_targets(
        self, y_train: ArrayLike, y_val: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode/validate targets and set any task-specific fitted attributes."""

    @abstractmethod
    def _make_scorer(self) -> Scorer:
        """Build the fitness scorer for the task."""

    @abstractmethod
    def _make_base_params(self) -> dict[str, Any]:
        """Fixed (non-evolved) XGBoost parameters defining the task."""

    def _runtime_params(self) -> dict[str, Any]:
        """Runtime params common to every task."""
        return {
            "tree_method": self.tree_method,
            "nthread": self.n_jobs,
            "verbosity": 0,
        }

    def _summary_extra(self) -> dict[str, Any]:
        """Task-specific keys added to ``summary.json``."""
        return {}

    # -- public API ---------------------------------------------------------
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
    ) -> _BasePBT:
        rng = np.random.default_rng(self.random_state)

        y_tr, y_va = self._prepare_targets(y_train, y_val)

        self._scorer = self._make_scorer()
        self._base_params = self._make_base_params()
        self._resolve_selection()

        dtrain = xgb.DMatrix(np.asarray(X_train), label=y_tr)
        dvalid = xgb.DMatrix(np.asarray(X_val))

        population = [
            PopulationMember(i, self.search_space.sample(rng)) for i in range(self.population_size)
        ]

        self.history_ = []
        self.best_score_ = self._scorer.worst
        self.best_params_ = None
        self.best_booster_ = None
        self.best_num_trees_ = 0
        self.stop_reason_ = "max_generations"

        start = time.perf_counter()
        no_improve = 0
        prev_best = self._scorer.worst
        generation = 0

        for generation in range(self.generations):
            for member in population:
                member.train_step(dtrain, self._base_params, self.step_rounds)
                member.evaluate(dvalid, y_va, self._scorer)
                self._update_best(member)

            if self._improved(prev_best):
                no_improve = 0
                prev_best = self.best_score_
            else:
                no_improve += 1

            self._log_generation(generation, population)

            reason = self._check_termination(start, no_improve)
            if reason is not None:
                self.stop_reason_ = reason
                break

            if generation < self.generations - 1:
                self._evolve(population, rng)

        self.n_generations_ = generation + 1
        if self.verbose:
            print(f"stopped after {self.n_generations_} generations: {self.stop_reason_}")
        return self

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict targets for ``X`` using the best booster."""

    def save_results(self, directory: str | Path) -> Path:
        """Persist the run's results to ``directory`` and return its path.

        Writes three files so a run can be reviewed later without re-training:

        * ``best_model.json``  – the best :class:`xgboost.Booster` (load with
          ``xgboost.Booster(model_file=...)``);
        * ``summary.json``     – best score, hyperparameters, stop reason, etc.;
        * ``history.csv``      – the per-generation best/mean/worst/global-best
          scores.
        """
        self._check_fitted()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        self.best_booster_.save_model(str(out / "best_model.json"))

        summary = {
            "metric": self._scorer.name,
            "maximize": self._scorer.maximize,
            "best_score": self.best_score_,
            "best_num_trees": self.best_num_trees_,
            "stop_reason": self.stop_reason_,
            "n_generations": self.n_generations_,
            "best_params": self.best_params_,
            **self._summary_extra(),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

        fields = ["generation", "best_score", "mean_score", "worst_score", "global_best_score"]
        with open(out / "history.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for record in self.history_:
                writer.writerow({k: record[k] for k in fields})

        return out

    # -- internals ----------------------------------------------------------
    def _predict_raw(self, X: ArrayLike) -> np.ndarray:
        """Raw best-booster output (probabilities or values)."""
        self._check_fitted()
        return np.asarray(self.best_booster_.predict(xgb.DMatrix(np.asarray(X))))

    def _resolve_selection(self) -> None:
        if self.top_k is not None:
            top_k = self.top_k
        else:
            top_k = max(1, round(self.selection_fraction * self.population_size))
        self._top_k = int(min(max(1, top_k), self.population_size))

        non_elite = self.population_size - self._top_k
        n_replace = self._top_k if self.n_replace is None else self.n_replace
        self._n_replace = int(min(n_replace, non_elite))

    def _update_best(self, member: PopulationMember) -> None:
        if self.best_params_ is None or self._scorer.is_better(member.score, self.best_score_):
            self.best_score_ = member.score
            self.best_params_ = dict(member.params)
            self.best_num_trees_ = member.num_trees
            raw = member.booster.save_raw()
            snapshot = xgb.Booster()
            snapshot.load_model(bytearray(raw))
            self.best_booster_ = snapshot

    def _evolve(
        self,
        population: list[PopulationMember],
        rng: np.random.Generator,
    ) -> None:
        ranked = sorted(
            population,
            key=lambda m: m.score,
            reverse=self._scorer.maximize,
        )
        elite = ranked[: self._top_k]
        non_elite = ranked[self._top_k :]
        if self._n_replace <= 0 or not non_elite:
            return
        to_replace = non_elite[-self._n_replace :]

        for member in to_replace:
            if self.crossover and len(elite) >= 2:
                i, j = rng.choice(len(elite), size=2, replace=False)
                p1, p2 = elite[int(i)], elite[int(j)]
                if self._scorer.is_better(p1.score, p2.score):
                    dominant, recessive = p1, p2
                else:
                    dominant, recessive = p2, p1
                member.clone_from(dominant)  # warm-start from fitter parent
                member.params = self._crossover(dominant.params, recessive.params, rng)
            else:
                donor = elite[int(rng.integers(len(elite)))]
                member.clone_from(donor)  # pure exploit
            member.params = self.search_space.mutate(
                member.params, rng, self.mutation_rate, self.perturb_factors
            )

    def _crossover(
        self,
        dominant_params: Mapping[str, Any],
        recessive_params: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        child: dict[str, Any] = {}
        for name in self.search_space.names:
            if name not in dominant_params:
                continue
            if rng.random() < self.dominance or name not in recessive_params:
                child[name] = dominant_params[name]
            else:
                child[name] = recessive_params[name]
        return child

    def _improved(self, prev_best: float) -> bool:
        if self.best_params_ is None:
            return False
        if self._scorer.maximize:
            return self.best_score_ > prev_best + self.tol
        return self.best_score_ < prev_best - self.tol

    def _check_termination(self, start: float, no_improve: int) -> str | None:
        if self.target_score is not None:
            reached = (
                self.best_score_ >= self.target_score
                if self._scorer.maximize
                else self.best_score_ <= self.target_score
            )
            if reached:
                return "target_score"
        if self.patience is not None and no_improve >= self.patience:
            return "patience"
        if self.max_time is not None and (time.perf_counter() - start) >= self.max_time:
            return "max_time"
        return None

    def _log_generation(
        self,
        generation: int,
        population: Sequence[PopulationMember],
    ) -> None:
        scores = np.array([m.score for m in population], dtype=float)
        best = np.nanmax(scores) if self._scorer.maximize else np.nanmin(scores)
        worst = np.nanmin(scores) if self._scorer.maximize else np.nanmax(scores)
        record = {
            "generation": generation,
            "best_score": float(best),
            "mean_score": float(np.nanmean(scores)),
            "worst_score": float(worst),
            "global_best_score": float(self.best_score_),
            "best_params": dict(self.best_params_) if self.best_params_ else None,
        }
        self.history_.append(record)
        if self.verbose:
            print(
                f"[gen {generation:>2}] {self._scorer.name}: "
                f"best={record['best_score']:.4f} "
                f"mean={record['mean_score']:.4f} "
                f"global_best={record['global_best_score']:.4f}"
            )

    def _check_fitted(self) -> None:
        if getattr(self, "best_booster_", None) is None:
            raise RuntimeError("Call fit() before predict().")


class PBTXGBClassifier(_BasePBT):
    """Tune and train an XGBoost classifier with an evolutionary population.

    See :class:`_BasePBT` for the full constructor signature. Classification
    targets are label-encoded; :meth:`predict_proba` returns class probabilities
    and :meth:`predict` returns the original class labels.

    Additional attribute set after :meth:`fit`: ``classes_``.
    """

    def _default_search_space(self) -> SearchSpace:
        return default_classification_space()

    def _prepare_targets(
        self, y_train: ArrayLike, y_val: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        self._label_encoder = LabelEncoder()
        y_tr = self._label_encoder.fit_transform(np.asarray(y_train))
        y_va = self._label_encoder.transform(np.asarray(y_val))
        self.classes_ = self._label_encoder.classes_
        if len(self.classes_) < 2:
            raise ValueError("Need at least two classes for classification.")
        return y_tr, y_va

    def _make_scorer(self) -> Scorer:
        return Scorer(self.metric, num_class=len(self.classes_), maximize=self.maximize)

    def _make_base_params(self) -> dict[str, Any]:
        num_class = len(self.classes_)
        if num_class == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
            extra: dict[str, Any] = {}
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
            extra = {"num_class": num_class}
        return {
            "objective": objective,
            "eval_metric": eval_metric,
            **self._runtime_params(),
            **extra,
        }

    def _summary_extra(self) -> dict[str, Any]:
        return {"classes": self.classes_.tolist()}

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        proba = self._predict_raw(X)
        if proba.ndim == 1:  # binary -> two columns
            return np.column_stack([1.0 - proba, proba])
        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self._label_encoder.inverse_transform(indices)


class PBTXGBRegressor(_BasePBT):
    """Tune and train an XGBoost regressor with an evolutionary population.

    See :class:`_BasePBT` for the full constructor signature. Targets are
    continuous floats (no label encoding); :meth:`predict` returns continuous
    values. There is no ``predict_proba``.
    """

    def _default_search_space(self) -> SearchSpace:
        return default_regression_space()

    def _prepare_targets(
        self, y_train: ArrayLike, y_val: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        y_tr = np.asarray(y_train, dtype=float)
        y_va = np.asarray(y_val, dtype=float)
        return y_tr, y_va

    def _make_scorer(self) -> Scorer:
        return Scorer(self.metric, maximize=self.maximize, regression=True)

    def _make_base_params(self) -> dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            **self._runtime_params(),
        }

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self._predict_raw(X)
