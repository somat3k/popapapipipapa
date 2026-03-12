"""ML Models: BaseModel, LinearRegressionModel, RandomForestModel,
GradientBoostingModel, NeuralNetworkModel, EnsembleModel, EquityHealthEnsembleModel."""

from __future__ import annotations

import abc
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
_WEIGHT_EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# Base Model
# ---------------------------------------------------------------------------

class BaseModel(abc.ABC):
    """Abstract base for all supervised ML models."""

    def __init__(self, name: str = "BaseModel") -> None:
        self.name = name
        self._trained = False
        self._metrics: Dict[str, float] = {}

    @property
    def is_trained(self) -> bool:
        return self._trained

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseModel":
        """Train the model on X, y."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X."""

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute standard metrics on a held-out set."""
        preds = self.predict(X)
        metrics: Dict[str, float] = {}
        if y.ndim == 1:
            metrics["rmse"] = _rmse(y, preds)
            metrics["mae"] = _mae(y, preds)
            metrics["directional_accuracy"] = _directional_accuracy(y, preds)
        self._metrics = metrics
        return metrics

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded: %s", path)
        return model

    def __repr__(self) -> str:
        return f"<{self.name} trained={self._trained}>"


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

class LinearRegressionModel(BaseModel):
    """Ridge-regularised linear regression."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__("LinearRegression")
        self.alpha = alpha
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "LinearRegressionModel":
        n, d = X.shape
        # Closed-form ridge solution: (X^T X + alpha I)^{-1} X^T y
        A = X.T @ X + self.alpha * np.eye(d)
        b = X.T @ y
        self._weights = np.linalg.solve(A, b)
        self._bias = float(np.mean(y - X @ self._weights))
        self._trained = True
        logger.info("[LinearRegression] Fitted on %d samples, %d features.", n, d)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model not trained.")
        return X @ self._weights + self._bias


# ---------------------------------------------------------------------------
# Random Forest (sklearn wrapper)
# ---------------------------------------------------------------------------

class RandomForestModel(BaseModel):
    """Thin wrapper around sklearn RandomForestRegressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestRegressor  # lazy import

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        self._trained = True
        logger.info("[RandomForest] Fitted n_estimators=%d.", self.n_estimators)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        return self._model.predict(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        if self._model is not None:
            return self._model.feature_importances_
        return None


# ---------------------------------------------------------------------------
# Gradient Boosting (XGBoost wrapper)
# ---------------------------------------------------------------------------

class GradientBoostingModel(BaseModel):
    """Thin wrapper around xgboost.XGBRegressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
    ) -> None:
        super().__init__("GradientBoosting")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "GradientBoostingModel":
        from xgboost import XGBRegressor  # lazy import

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            objective="reg:squarederror",
            verbosity=0,
        )
        self._model.fit(X, y)
        self._trained = True
        logger.info("[GradientBoosting] Fitted n_estimators=%d.", self.n_estimators)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Neural Network (scikit-learn MLP)
# ---------------------------------------------------------------------------

class NeuralNetworkModel(BaseModel):
    """MLP-based regression model using scikit-learn."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        random_state: int = 42,
        early_stopping: bool = True,
    ) -> None:
        super().__init__("NeuralNetwork")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self._model: Any = None

    def _reshape_features(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return X
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        raise ValueError(
            "[NeuralNetwork] Expected 2D (samples, features) or 3D "
            "(samples, sequence_length, features) input. "
            f"Got {X.ndim}D array with shape {X.shape}."
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "NeuralNetworkModel":
        from sklearn.neural_network import MLPRegressor  # lazy import

        X_flat = self._reshape_features(X)
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
        )
        self._model.fit(X_flat, y)
        self._trained = True
        logger.info("[NeuralNetwork] Fitted layers=%s.", self.hidden_layer_sizes)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        X_flat = self._reshape_features(X)
        return self._model.predict(X_flat)


# ---------------------------------------------------------------------------
# Ensemble Model
# ---------------------------------------------------------------------------

class EnsembleModel(BaseModel):
    """Averages predictions from multiple base models."""

    def __init__(self, models: Optional[List[BaseModel]] = None) -> None:
        super().__init__("Ensemble")
        self._models: List[BaseModel] = models or []

    def add_model(self, model: BaseModel) -> None:
        self._models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "EnsembleModel":
        for model in self._models:
            model.fit(X, y, **kwargs)
        self._trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("No models in ensemble.")
        preds = np.array([m.predict(X) for m in self._models])
        return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# Equity + Health Weighted Ensemble
# ---------------------------------------------------------------------------

class EquityHealthEnsembleModel(BaseModel):
    """Ensemble that weights models by equity growth and health score."""

    DEFAULT_RETURNS = np.zeros(1)
    DEFAULT_HEALTH_FACTORS = np.full(1, 2.0)

    def __init__(self, models: Optional[List[BaseModel]] = None) -> None:
        super().__init__("EquityHealthEnsemble")
        self._models: List[BaseModel] = models or []
        self._weights: Optional[np.ndarray] = None

    def add_model(self, model: BaseModel) -> None:
        self._models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "EquityHealthEnsembleModel":
        for model in self._models:
            model.fit(X, y, **kwargs)
        self._trained = True
        return self

    def update_weights(
        self,
        returns: List[np.ndarray],
        health_factors: List[np.ndarray],
        periods_per_year: float = 252.0,
    ) -> np.ndarray:
        if len(returns) != len(self._models) or len(health_factors) != len(self._models):
            raise ValueError("Returns and health_factors must match number of models.")
        if not self._models:
            raise RuntimeError("No models in ensemble.")

        from app.evaluation.metrics import AgentEvaluationMetrics

        scores: List[float] = []
        for model_returns, model_hf in zip(returns, health_factors):
            resolved_returns = np.asarray(model_returns, dtype=float)
            resolved_hf = np.asarray(model_hf, dtype=float)
            aligned_len = min(len(resolved_returns), len(resolved_hf))
            if aligned_len > 0:
                resolved_returns = resolved_returns[:aligned_len]
                resolved_hf = resolved_hf[:aligned_len]
            if resolved_returns.size == 0:
                resolved_returns = self.DEFAULT_RETURNS.copy()
            if resolved_hf.size == 0:
                resolved_hf = self.DEFAULT_HEALTH_FACTORS.copy()
            metrics = AgentEvaluationMetrics(
                returns=resolved_returns,
                health_factors=resolved_hf,
            )
            growth = max(0.0, metrics.cagr(periods_per_year))
            health_score = max(0.0, metrics.health_factor_score())
            scores.append(growth * health_score)

        weights = self._normalise_weights(scores)
        self._weights = weights
        return weights

    def update_weights_from_equity(
        self,
        equity_curves: List[np.ndarray],
        health_factors: List[np.ndarray],
        periods_per_year: float = 252.0,
    ) -> np.ndarray:
        returns = []
        for equity in equity_curves:
            eq = np.asarray(equity, dtype=float)
            if len(eq) < 2:
                returns.append(np.array([]))
            else:
                denom = eq[:-1]
                # Avoid division-by-zero by returning 0.0 when the denominator is 0.
                returns.append(
                    np.divide(
                        np.diff(eq),
                        denom,
                        out=np.zeros_like(denom),
                        where=denom != 0,
                    )
                )
        return self.update_weights(returns, health_factors, periods_per_year)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("No models in ensemble.")
        preds = np.array([m.predict(X) for m in self._models])
        weights = self._resolve_weights()
        return np.average(preds, axis=0, weights=weights)

    def _resolve_weights(self) -> np.ndarray:
        if self._weights is None or len(self._weights) != len(self._models):
            return np.full(len(self._models), 1 / len(self._models))
        return self._weights

    @staticmethod
    def _normalise_weights(raw: List[float]) -> np.ndarray:
        values = np.array(raw, dtype=float)
        total = float(np.sum(values))
        if total <= _WEIGHT_EPSILON:
            return np.full(len(values), 1 / len(values))
        return values / total


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """In-memory registry mapping model names to saved model paths."""

    def __init__(self, base_dir: Path = Path("models")) -> None:
        self._base_dir = base_dir
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, model: BaseModel, version: str = "latest") -> Path:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        key = f"{model.name}_v{version}"
        path = self._base_dir / f"{key}.pkl"
        model.save(path)
        self._registry[key] = {"path": path, "metrics": dict(model._metrics)}
        logger.info("Model registered: %s → %s", key, path)
        return path

    def load(self, model_name: str, version: str = "latest") -> BaseModel:
        key = f"{model_name}_v{version}"
        if key not in self._registry:
            raise KeyError(f"Model '{key}' not found in registry.")
        return BaseModel.load(self._registry[key]["path"])

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"key": k, "path": str(v["path"]), "metrics": v["metrics"]}
            for k, v in self._registry.items()
        ]
