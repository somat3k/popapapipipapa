"""Dynamic hyperparameter adjustment module.

Provides HyperparamSpace, HyperparamScheduler, and BayesianOptimiser.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter space definition
# ---------------------------------------------------------------------------

@dataclass
class HyperparamSpec:
    """Specification for a single hyperparameter."""

    name: str
    min_val: float
    max_val: float
    step: Optional[float] = None
    log_scale: bool = False
    param_type: str = "float"  # "float" | "int" | "categorical"
    choices: List[Any] = field(default_factory=list)

    def sample(self) -> Any:
        if self.param_type == "categorical":
            return random.choice(self.choices) if self.choices else None
        if self.log_scale:
            lo, hi = math.log(self.min_val), math.log(self.max_val)
            val = math.exp(random.uniform(lo, hi))
        else:
            val = random.uniform(self.min_val, self.max_val)
        if self.param_type == "int":
            val = int(round(val))
        if self.step is not None and self.param_type != "int":
            val = round(val / self.step) * self.step
        return val

    def clip(self, value: Any) -> Any:
        if self.param_type == "categorical":
            return value
        return max(self.min_val, min(self.max_val, value))


class HyperparamSpace:
    """Container for a collection of HyperparamSpec objects."""

    def __init__(self) -> None:
        self._specs: Dict[str, HyperparamSpec] = {}

    def add(self, spec: HyperparamSpec) -> None:
        self._specs[spec.name] = spec

    def sample(self) -> Dict[str, Any]:
        return {name: spec.sample() for name, spec in self._specs.items()}

    def clip(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            name: self._specs[name].clip(v) if name in self._specs else v
            for name, v in params.items()
        }

    def __len__(self) -> int:
        return len(self._specs)


# ---------------------------------------------------------------------------
# Learning rate schedules
# ---------------------------------------------------------------------------

class LRScheduler:
    """Learning rate scheduler with warm-up and cosine annealing."""

    def __init__(
        self,
        base_lr: float = 1e-3,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        min_lr: float = 1e-6,
    ) -> None:
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self._step = 0

    def step(self) -> float:
        self._step += 1
        lr = self._compute()
        return lr

    def _compute(self) -> float:
        s = self._step
        if s <= self.warmup_steps:
            return self.base_lr * (s / max(1, self.warmup_steps))
        progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def reset(self) -> None:
        self._step = 0

    @property
    def current_lr(self) -> float:
        return self._compute()


# ---------------------------------------------------------------------------
# Dynamic hyperparameter scheduler
# ---------------------------------------------------------------------------

class HyperparamScheduler:
    """Adjusts hyperparameters dynamically during training.

    Monitors a validation metric and adjusts learning rate, dropout,
    and batch size according to configurable policies.
    """

    def __init__(
        self,
        initial_params: Dict[str, Any],
        lr_spec: Optional[HyperparamSpec] = None,
        patience: int = 5,
        factor: float = 0.5,
    ) -> None:
        self._params = dict(initial_params)
        self._lr_spec = lr_spec
        self.patience = patience
        self.factor = factor
        self._best_metric: float = float("inf")
        self._wait: int = 0
        self._history: List[Dict[str, Any]] = []
        self._lr_scheduler: Optional[LRScheduler] = None

        if "learning_rate" in self._params:
            lr = self._params["learning_rate"]
            total = initial_params.get("epochs", 100)
            self._lr_scheduler = LRScheduler(
                base_lr=lr,
                warmup_steps=max(1, int(total * 0.05)),
                total_steps=total,
            )

    def on_epoch_end(self, epoch: int, val_metric: float) -> Dict[str, Any]:
        """Call at the end of every training epoch.

        Returns the (potentially updated) hyperparameter dict.
        """
        improved = val_metric < self._best_metric
        if improved:
            self._best_metric = val_metric
            self._wait = 0
        else:
            self._wait += 1

        # Reduce LR on plateau
        if self._wait >= self.patience and "learning_rate" in self._params:
            old_lr = self._params["learning_rate"]
            new_lr = old_lr * self.factor
            if self._lr_spec:
                new_lr = self._lr_spec.clip(new_lr)
            self._params["learning_rate"] = new_lr
            self._wait = 0
            logger.info(
                "HyperparamScheduler: LR reduced %.6f → %.6f (epoch %d)",
                old_lr,
                new_lr,
                epoch,
            )

        # Step LR scheduler
        if self._lr_scheduler is not None:
            scheduled_lr = self._lr_scheduler.step()
            # Use the smaller of plateau-reduced and scheduled LR
            self._params["learning_rate"] = min(
                self._params.get("learning_rate", scheduled_lr), scheduled_lr
            )

        snapshot = {"epoch": epoch, "val_metric": val_metric, **self._params}
        self._history.append(snapshot)
        return dict(self._params)

    def should_stop_early(self) -> bool:
        return self._wait >= self.patience * 2

    def get_params(self) -> Dict[str, Any]:
        return dict(self._params)

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Random / Grid Search
# ---------------------------------------------------------------------------

class RandomSearch:
    """Random hyperparameter search over a HyperparamSpace."""

    def __init__(self, space: HyperparamSpace, n_trials: int = 20) -> None:
        self.space = space
        self.n_trials = n_trials
        self._results: List[Tuple[Dict[str, Any], float]] = []

    def run(
        self,
        objective: Callable[[Dict[str, Any]], float],
    ) -> Tuple[Dict[str, Any], float]:
        """Run random search, returning best (params, score)."""
        best_params: Dict[str, Any] = {}
        best_score = float("inf")
        for i in range(self.n_trials):
            params = self.space.sample()
            try:
                score = objective(params)
            except Exception:
                logger.exception("Trial %d failed.", i)
                score = float("inf")
            self._results.append((params, score))
            if score < best_score:
                best_score = score
                best_params = params
            logger.debug("Trial %d: score=%.4f params=%s", i, score, params)
        return best_params, best_score

    def results(self) -> List[Tuple[Dict[str, Any], float]]:
        return list(self._results)


# ---------------------------------------------------------------------------
# Bayesian Optimisation (optuna-backed)
# ---------------------------------------------------------------------------

class BayesianOptimiser:
    """Bayesian hyperparameter optimisation using Optuna."""

    def __init__(self, space: HyperparamSpace, n_trials: int = 30) -> None:
        self.space = space
        self.n_trials = n_trials
        self._study: Any = None
        self._best_params: Dict[str, Any] = {}
        self._best_score: float = float("inf")

    def run(
        self, objective: Callable[[Dict[str, Any]], float]
    ) -> Tuple[Dict[str, Any], float]:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not available; falling back to random search.")
            rs = RandomSearch(self.space, n_trials=self.n_trials)
            best_params, best_score = rs.run(objective)
            self._best_params = best_params
            self._best_score = best_score
            return best_params, best_score

        specs = self.space._specs  # access internal dict

        def _optuna_objective(trial: Any) -> float:
            params: Dict[str, Any] = {}
            for name, spec in specs.items():
                if spec.param_type == "categorical":
                    params[name] = trial.suggest_categorical(name, spec.choices)
                elif spec.param_type == "int":
                    lo, hi = int(spec.min_val), int(spec.max_val)
                    params[name] = trial.suggest_int(name, lo, hi)
                elif spec.log_scale:
                    params[name] = trial.suggest_float(
                        name, spec.min_val, spec.max_val, log=True
                    )
                else:
                    params[name] = trial.suggest_float(name, spec.min_val, spec.max_val)
            return objective(params)

        study = optuna.create_study(direction="minimize")
        study.optimize(_optuna_objective, n_trials=self.n_trials)
        self._study = study
        self._best_params = study.best_params
        self._best_score = study.best_value
        return self._best_params, self._best_score

    @property
    def best_params(self) -> Dict[str, Any]:
        return self._best_params

    @property
    def best_score(self) -> float:
        return self._best_score
