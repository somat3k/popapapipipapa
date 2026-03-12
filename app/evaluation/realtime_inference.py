"""Realtime inference runner for bar-by-bar ML model evaluation.

Simulates realtime inference conditions by iterating through a stream of bars
one at a time, running the trained model on each bar's features, and exchanging
inference payloads with agents via the :class:`~app.agents.base_agent.MessageBus`.

Usage
-----
::

    from app.evaluation.realtime_inference import RealtimeInferenceRunner
    from app.evaluation.data_loader import OHLCVLoader
    from app.ml.models import LinearRegressionModel
    import numpy as np

    loader = OHLCVLoader()
    bars = loader.generate_synthetic(n=100, symbol="ETH")

    model = LinearRegressionModel()
    X = np.random.randn(80, 8)
    y = np.sign(np.random.randn(80))
    model.fit(X, y)

    runner = RealtimeInferenceRunner(model=model, bars=bars, symbol="ETH")
    summary = runner.run()
    print(summary.buy_signals, summary.sell_signals)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.agents.base_agent import MESSAGE_BUS, MessageBus
from app.evaluation.rl_pipeline import RLEnvironment
from app.ml.models import BaseModel
from app.trading.algorithms import Bar

logger = logging.getLogger(__name__)

# Default message bus topic for per-bar inference payloads
INFERENCE_TOPIC = "ml.inference"


@dataclass
class InferencePayload:
    """Single-bar inference result exchanged between agents via the MessageBus."""

    bar_index: int
    symbol: str
    timestamp: float
    close: float
    prediction: float
    action: int          # +1 buy, 0 hold, -1 sell
    confidence: float    # capped at 1.0
    features: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for message bus publishing."""
        return {
            "bar_index": self.bar_index,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "close": self.close,
            "prediction": self.prediction,
            "action": self.action,
            "confidence": self.confidence,
        }


@dataclass
class InferenceSummary:
    """Aggregated summary of a realtime inference run."""

    symbol: str
    total_bars: int
    buy_signals: int
    hold_signals: int
    sell_signals: int
    mean_confidence: float
    elapsed_s: float
    payloads: List[InferencePayload] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation (without per-bar payloads)."""
        return {
            "symbol": self.symbol,
            "total_bars": self.total_bars,
            "buy_signals": self.buy_signals,
            "hold_signals": self.hold_signals,
            "sell_signals": self.sell_signals,
            "mean_confidence": self.mean_confidence,
            "elapsed_s": self.elapsed_s,
        }


class RealtimeInferenceRunner:
    """Runs bar-by-bar inference on a trained model, exchanging payloads via the MessageBus.

    The runner simulates realtime conditions by iterating through the provided
    bars one at a time, computing features, running inference, and publishing
    each result as an :class:`InferencePayload` to the message bus.

    Parameters
    ----------
    model:
        A trained :class:`~app.ml.models.BaseModel` instance.
    bars:
        Historical OHLCV bars used as the realtime stream.
    symbol:
        Asset symbol label (e.g. ``"ETH"``).
    topic:
        Message bus topic for inference payloads.  Defaults to ``"ml.inference"``.
    message_bus:
        Optional :class:`~app.agents.base_agent.MessageBus` instance.
        Defaults to the module-level ``MESSAGE_BUS``.
    on_payload:
        Optional callback invoked with each :class:`InferencePayload` as it
        is produced.  Useful for testing without a message bus subscription.
    """

    def __init__(
        self,
        model: BaseModel,
        bars: List[Bar],
        symbol: str = "",
        topic: str = INFERENCE_TOPIC,
        message_bus: Optional[MessageBus] = None,
        on_payload: Optional[Callable[[InferencePayload], None]] = None,
    ) -> None:
        self.model = model
        self.bars = bars
        self.symbol = symbol
        self.topic = topic
        self.bus = message_bus or MESSAGE_BUS
        self.on_payload = on_payload

    def run(self) -> InferenceSummary:
        """Run bar-by-bar inference and publish payloads.

        For each bar in the stream:

        1. Extract the feature vector pre-computed by :class:`~app.evaluation.rl_pipeline.RLEnvironment`.
        2. Run ``model.predict()`` on the single-bar feature vector.
        3. Map the scalar prediction to an action (``+1`` buy / ``0`` hold / ``-1`` sell).
        4. Publish an :class:`InferencePayload` dict to the configured message bus topic.

        Returns
        -------
        InferenceSummary
            Aggregated statistics for the full inference run.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self.model.is_trained:
            raise RuntimeError(
                f"[RealtimeInference] Model '{self.model.name}' must be trained "
                "before running realtime inference."
            )

        if not self.bars:
            return InferenceSummary(
                symbol=self.symbol,
                total_bars=0,
                buy_signals=0,
                hold_signals=0,
                sell_signals=0,
                mean_confidence=0.0,
                elapsed_s=0.0,
            )

        t0 = time.perf_counter()

        # Build the full feature matrix once using the shared RLEnvironment helper
        # so that feature construction is identical to training time.
        env = RLEnvironment(self.bars)
        features: np.ndarray = env._features  # shape (n_bars, n_features)

        payloads: List[InferencePayload] = []
        buy_count = hold_count = sell_count = 0
        confidence_sum = 0.0

        for i, bar in enumerate(self.bars):
            feat = features[i].reshape(1, -1)
            try:
                pred_arr = self.model.predict(feat)
                pred_val = float(pred_arr[0])
            except Exception:
                logger.warning(
                    "[RealtimeInference] Inference failed at bar %d; skipping.", i
                )
                continue

            action = 1 if pred_val > 0.01 else (-1 if pred_val < -0.01 else 0)
            confidence = min(1.0, abs(pred_val))

            payload = InferencePayload(
                bar_index=i,
                symbol=self.symbol,
                timestamp=bar.timestamp,
                close=bar.close,
                prediction=round(pred_val, 6),
                action=action,
                confidence=round(confidence, 4),
                features=feat.flatten().tolist(),
            )
            payloads.append(payload)

            # Publish lightweight dict to the message bus for broad compatibility
            self.bus.publish(self.topic, payload.to_dict())

            if self.on_payload:
                self.on_payload(payload)

            if action == 1:
                buy_count += 1
            elif action == -1:
                sell_count += 1
            else:
                hold_count += 1

            confidence_sum += confidence

        elapsed = time.perf_counter() - t0
        n = len(payloads)
        mean_conf = confidence_sum / n if n > 0 else 0.0

        logger.info(
            "[RealtimeInference] %s  bars=%d  buy=%d  hold=%d  sell=%d  "
            "mean_conf=%.4f  t=%.3fs",
            self.symbol or "?",
            n,
            buy_count,
            hold_count,
            sell_count,
            mean_conf,
            elapsed,
        )

        return InferenceSummary(
            symbol=self.symbol,
            total_bars=n,
            buy_signals=buy_count,
            hold_signals=hold_count,
            sell_signals=sell_count,
            mean_confidence=round(mean_conf, 4),
            elapsed_s=round(elapsed, 3),
            payloads=payloads,
        )
