"""ML Trainer: end-to-end supervised training with dynamic hyperparameter
adjustment, checkpointing, and evaluation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .hyperparams import HyperparamScheduler
from .models import BaseModel

logger = logging.getLogger(__name__)


class TrainingCallback:
    """Hook interface for training events."""

    def on_epoch_begin(self, epoch: int, params: Dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_train_begin(self) -> None:
        pass

    def on_train_end(self, metrics: Dict[str, float]) -> None:
        pass


class Trainer:
    """Orchestrates model training with dynamic hyperparameter adjustment.

    Features
    --------
    - Temporal train/val split (no look-ahead)
    - Per-epoch validation metric computation
    - Dynamic LR and hyperparameter adjustment via HyperparamScheduler
    - Early stopping (patience-based)
    - Checkpoint saving of best model
    - Live metric logging and callback support
    """

    def __init__(
        self,
        model: BaseModel,
        hyperparams: Dict[str, Any],
        val_fraction: float = 0.2,
        checkpoint_dir: Optional[Path] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> None:
        self.model = model
        self._initial_params = dict(hyperparams)
        self.val_fraction = val_fraction
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.callbacks = callbacks or []
        self.progress_callback = progress_callback
        self._history: List[Dict[str, Any]] = []
        self._best_model: Optional[BaseModel] = None
        self._best_val_loss: float = float("inf")
        self._training_time: float = 0.0

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Run training and return a summary dict."""
        epochs = int(self._initial_params.get("epochs", 50))
        scheduler = HyperparamScheduler(
            initial_params=self._initial_params,
            patience=int(self._initial_params.get("patience", 5)),
            factor=float(self._initial_params.get("lr_factor", 0.5)),
        )

        # Temporal split
        n = len(X)
        split = int(n * (1 - self.val_fraction))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        for cb in self.callbacks:
            cb.on_train_begin()

        start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            current_params = scheduler.get_params()
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch, current_params)

            # Fit on training split
            self.model.fit(X_train, y_train, **current_params)

            # Evaluate on validation split
            val_metrics = self.model.evaluate(X_val, y_val)
            val_loss = val_metrics.get("rmse", float("inf"))

            current_params = scheduler.on_epoch_end(epoch, val_loss)

            row: Dict[str, Any] = {"epoch": epoch, **val_metrics, **current_params}
            self._history.append(row)

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, val_metrics)

            if self.progress_callback:
                self.progress_callback(epoch, epochs, val_metrics)

            # Checkpoint best model
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._save_checkpoint(epoch)

            if scheduler.should_stop_early():
                logger.info("[Trainer] Early stopping at epoch %d.", epoch)
                break

        self._training_time = time.perf_counter() - start
        final_metrics = self.model.evaluate(X_val, y_val)

        for cb in self.callbacks:
            cb.on_train_end(final_metrics)

        summary = {
            "model": self.model.name,
            "epochs_trained": epoch,
            "training_time_s": round(self._training_time, 2),
            "best_val_loss": self._best_val_loss,
            **final_metrics,
        }
        logger.info("[Trainer] Training complete: %s", summary)
        return summary

    def _save_checkpoint(self, epoch: int) -> None:
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.checkpoint_dir / f"{self.model.name}_best.pkl"
            self.model.save(path)
        except Exception:
            logger.exception("[Trainer] Checkpoint save failed at epoch %d.", epoch)

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def best_val_loss(self) -> float:
        return self._best_val_loss

    def training_time(self) -> float:
        return self._training_time
