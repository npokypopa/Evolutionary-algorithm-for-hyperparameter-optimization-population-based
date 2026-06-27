"""Hyperparameter search space for Population Based Training of XGBoost.

A :class:`SearchSpace` is an ordered collection of :class:`Hyperparameter`
objects. Each hyperparameter knows how to *sample* an initial value and how to
*perturb* an existing value during the PBT "explore" step.

Only hyperparameters that are safe to vary while *continuing* to boost an
existing model belong here (eta, subsample, regularization, etc.). Structural /
task-defining parameters (objective, num_class, eval_metric, tree_method) are
held fixed by :class:`~pbt_xgb.pbt.PBTXGBClassifier` and are deliberately absent.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Hyperparameter",
    "SearchSpace",
    "default_classification_space",
    "default_regression_space",
]


@dataclass
class Hyperparameter:
    """A single tunable hyperparameter.

    Parameters
    ----------
    name:
        XGBoost parameter name (e.g. ``"eta"``).
    kind:
        One of ``"float"``, ``"int"`` or ``"categorical"``.
    low, high:
        Inclusive bounds for numeric (``float``/``int``) hyperparameters.
    choices:
        Allowed values for ``"categorical"`` hyperparameters.
    log:
        If ``True``, sampling is uniform in log-space (useful for learning rate
        and regularization terms that span several orders of magnitude).
    """

    name: str
    kind: str = "float"
    low: float | None = None
    high: float | None = None
    choices: Sequence[Any] | None = None
    log: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("float", "int", "categorical"):
            raise ValueError(f"Unknown hyperparameter kind: {self.kind!r}")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"Categorical hyperparameter {self.name!r} needs choices")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"Numeric hyperparameter {self.name!r} needs low and high")
            if self.high < self.low:
                raise ValueError(f"Hyperparameter {self.name!r} has high < low")
            if self.log and self.low <= 0:
                raise ValueError(f"Log-scaled hyperparameter {self.name!r} needs low > 0")

    # -- sampling -----------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> Any:
        """Draw a fresh value for population initialization."""
        if self.kind == "categorical":
            idx = rng.integers(len(self.choices))
            return self.choices[idx]
        if self.log:
            value = float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        else:
            value = float(rng.uniform(self.low, self.high))
        return self._coerce(value)

    # -- perturbation -------------------------------------------------------
    def perturb(
        self,
        value: Any,
        rng: np.random.Generator,
        factors: tuple[float, float] = (0.8, 1.2),
    ) -> Any:
        """Return a perturbed copy of ``value`` (the PBT "explore" step).

        Numeric values are multiplied by one of ``factors`` (chosen at random)
        and clipped to bounds. Categorical values are resampled.
        """
        if self.kind == "categorical":
            return self.sample(rng)
        factor = factors[int(rng.integers(len(factors)))]
        return self._coerce(self._clip(value * factor))

    # -- helpers ------------------------------------------------------------
    def _clip(self, value: float) -> float:
        return float(np.clip(value, self.low, self.high))

    def _coerce(self, value: float) -> Any:
        value = self._clip(value)
        if self.kind == "int":
            return int(round(value))
        return float(value)


class SearchSpace:
    """An ordered set of :class:`Hyperparameter` objects."""

    def __init__(self, hyperparameters: Sequence[Hyperparameter]):
        self._params: OrderedDict[str, Hyperparameter] = OrderedDict()
        for hp in hyperparameters:
            if hp.name in self._params:
                raise ValueError(f"Duplicate hyperparameter: {hp.name!r}")
            self._params[hp.name] = hp

    def __len__(self) -> int:
        return len(self._params)

    def __iter__(self):
        return iter(self._params.values())

    def __contains__(self, name: object) -> bool:
        return name in self._params

    @property
    def names(self) -> list[str]:
        return list(self._params.keys())

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample a full hyperparameter configuration."""
        return {name: hp.sample(rng) for name, hp in self._params.items()}

    def perturb(
        self,
        params: Mapping[str, Any],
        rng: np.random.Generator,
        factors: tuple[float, float] = (0.8, 1.2),
    ) -> dict[str, Any]:
        """Perturb every hyperparameter in ``params``.

        Keys not present in the search space are passed through unchanged.
        """
        out = dict(params)
        for name, hp in self._params.items():
            if name in out:
                out[name] = hp.perturb(out[name], rng, factors)
        return out

    def mutate(
        self,
        params: Mapping[str, Any],
        rng: np.random.Generator,
        mutation_rate: float = 1.0,
        factors: tuple[float, float] = (0.8, 1.2),
    ) -> dict[str, Any]:
        """Perturb a random subset of the hyperparameters.

        ``mutation_rate`` is the fraction of genes to mutate: the number of
        mutated hyperparameters is ``round(mutation_rate * n_genes)``, and that
        many are chosen uniformly at random. ``mutation_rate=1.0`` mutates all,
        ``0.0`` mutates none.
        """
        out = dict(params)
        names = [name for name in self._params if name in out]
        if not names or mutation_rate <= 0.0:
            return out
        k = int(round(mutation_rate * len(names)))
        k = max(0, min(k, len(names)))
        if k == 0:
            return out
        chosen = rng.choice(len(names), size=k, replace=False)
        for i in chosen:
            name = names[int(i)]
            out[name] = self._params[name].perturb(out[name], rng, factors)
        return out


def _default_tree_space() -> SearchSpace:
    """The shared per-tree search space used by both tasks.

    Every hyperparameter here is objective-agnostic — it controls tree growth
    and regularization, not the loss — so classification and regression evolve
    the same set of genes.
    """
    return SearchSpace(
        [
            Hyperparameter("eta", "float", low=1e-3, high=0.5, log=True),
            Hyperparameter("max_depth", "int", low=2, high=10),
            Hyperparameter("min_child_weight", "float", low=0.5, high=10.0),
            Hyperparameter("gamma", "float", low=0.0, high=5.0),
            Hyperparameter("subsample", "float", low=0.5, high=1.0),
            Hyperparameter("colsample_bytree", "float", low=0.5, high=1.0),
            Hyperparameter("reg_lambda", "float", low=1e-3, high=10.0, log=True),
            Hyperparameter("reg_alpha", "float", low=1e-3, high=10.0, log=True),
        ]
    )


def default_classification_space() -> SearchSpace:
    """Sensible default search space for XGBoost classification.

    Covers the hyperparameters with the largest effect on generalization that
    are also safe to vary during continued boosting.
    """
    return _default_tree_space()


def default_regression_space() -> SearchSpace:
    """Sensible default search space for XGBoost regression.

    Identical tunable per-tree parameters as
    :func:`default_classification_space` — they govern tree structure and
    regularization and are independent of the learning objective.
    """
    return _default_tree_space()
