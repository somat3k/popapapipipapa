"""ML Models: BaseModel, LinearRegressionModel, RandomForestModel,
GradientBoostingModel, LSTMModel, EnsembleModel."""

from __future__ import annotations

import abc
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
# Gradient Boosting (sklearn wrapper)
# ---------------------------------------------------------------------------

class GradientBoostingModel(BaseModel):
    """Thin wrapper around sklearn GradientBoostingRegressor."""

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
        from sklearn.ensemble import GradientBoostingRegressor  # lazy import

        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
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
# LSTM Model (PyTorch)
# ---------------------------------------------------------------------------

class LSTMModel(BaseModel):
    """LSTM-based sequence model for time-series prediction."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        seq_len: int = 20,
        auto_adjust_input_size: bool = True,
    ) -> None:
        super().__init__("LSTM")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seq_len = seq_len
        self.auto_adjust_input_size = auto_adjust_input_size
        self._net: Any = None
        self._train_losses: List[float] = []

    def _build_net(self) -> Any:
        try:
            import torch
            import torch.nn as nn

            class _Net(nn.Module):
                def __init__(self, in_sz: int, hid: int, layers: int) -> None:
                    super().__init__()
                    self.lstm = nn.LSTM(in_sz, hid, layers, batch_first=True)
                    self.fc = nn.Linear(hid, 1)

                def forward(self, x: Any) -> Any:
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :]).squeeze(-1)

            return _Net(self.input_size, self.hidden_size, self.num_layers)
        except ImportError:
            return None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "LSTMModel":
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("[LSTM] PyTorch not available. Falling back to identity.")
            self._trained = True
            return self

        feature_dim = X.shape[-1] if X.ndim >= 2 else 1
        if feature_dim != self.input_size:
            if self.auto_adjust_input_size:
                logger.info(
                    "[LSTM] Adjusting input_size from %d to %d to match features.",
                    self.input_size,
                    feature_dim,
                )
                self.input_size = feature_dim
            else:
                raise ValueError(
                    "[LSTM] input_size mismatch: "
                    f"expected {self.input_size}, got {feature_dim}."
                )

        net = self._build_net()
        if net is None:
            self._trained = True
            return self

        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Reshape X to (N, seq_len, input_size) if 2-D
        if X.ndim == 2:
            # Treat each row as a sequence of length 1 × features
            X_t = torch.FloatTensor(X).unsqueeze(1)
        else:
            X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)

        net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = net(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            self._train_losses.append(float(loss.item()))
            if (epoch + 1) % 10 == 0:
                logger.debug("[LSTM] Epoch %d/%d  loss=%.4f", epoch + 1, self.epochs, loss.item())

        self._net = net
        self._trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        try:
            import torch
        except ImportError:
            return np.zeros(len(X))

        if self._net is None:
            return np.zeros(len(X))

        self._net.eval()
        if X.ndim == 2:
            X_t = torch.FloatTensor(X).unsqueeze(1)
        else:
            X_t = torch.FloatTensor(X)
        with torch.no_grad():
            return self._net(X_t).numpy()

    @property
    def train_losses(self) -> List[float]:
        return list(self._train_losses)


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
