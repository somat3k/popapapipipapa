"""Multi-timeframe signal fusion for unified decision making.

Aggregates trading signals produced by the same (or different) algorithms
run on different timeframe datasets into a single composite signal.

Philosophy
----------
- Higher timeframes (1d, 4h) carry more structural weight.
- The lower timeframe (1h) provides entry timing refinement.
- Signals are weighted by timeframe multiplier and algorithm confidence.
- The final fused direction is determined by weighted voting.

Usage
-----
::

    from app.evaluation.timeframe_fusion import MultiTimeframeAnalyzer, TimeframeFuser
    from app.trading.algorithms import MeanReversionAlgo, MomentumAlgo, TrendFollowingAlgo

    loader = OHLCVLoader()
    tf_bars = loader.fetch_multi_timeframe("ETH", days=90)

    analyzer = MultiTimeframeAnalyzer()
    analyzer.add_timeframe("1d", tf_bars["1d"], weight=3.0)
    analyzer.add_timeframe("4h", tf_bars["4h"], weight=2.0)
    analyzer.add_timeframe("1h", tf_bars["1h"], weight=1.0)

    fused_signal = analyzer.fuse()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.trading.algorithms import (
    Algorithm,
    Bar,
    MeanReversionAlgo,
    MomentumAlgo,
    Signal,
    SignalAggregator,
    TrendFollowingAlgo,
    _ema,
    _rsi,
)

logger = logging.getLogger(__name__)

# Small epsilon for avoiding division-by-zero in weighted normalisation
_EPSILON = 1e-12

# Default timeframe weights — higher timeframes carry more structural authority
DEFAULT_TF_WEIGHTS: Dict[str, float] = {
    "1w": 5.0,
    "3d": 4.0,
    "1d": 3.0,
    "12h": 2.5,
    "8h": 2.0,
    "4h": 2.0,
    "2h": 1.5,
    "1h": 1.0,
    "30m": 0.75,
    "15m": 0.5,
}


@dataclass
class TimeframeLayer:
    """One timeframe's contribution to the fused decision."""

    label: str
    bars: List[Bar]
    weight: float
    algorithms: List[Algorithm] = field(default_factory=list)
    last_signal: Optional[Signal] = None
    # Set to True once run() has processed this layer, distinguishing
    # "processed but no signal" from "not yet processed".
    processed: bool = False


@dataclass
class FusedDecision:
    """Result of multi-timeframe signal fusion."""

    direction: int          # +1 buy, -1 sell, 0 neutral
    confidence: float       # weighted confidence 0–1
    agreement_pct: float    # fraction of timeframes that agree with final direction
    bullish_weight: float   # total weight for long signals
    bearish_weight: float   # total weight for short signals
    neutral_weight: float   # total weight for neutral / no signal
    layers: Dict[str, Optional[Signal]] = field(default_factory=dict)

    def is_high_confidence(self, threshold: float = 0.65) -> bool:
        return self.confidence >= threshold and self.agreement_pct >= 0.5

    def summary(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "agreement_pct": round(self.agreement_pct, 4),
            "bullish_weight": round(self.bullish_weight, 4),
            "bearish_weight": round(self.bearish_weight, 4),
            "neutral_weight": round(self.neutral_weight, 4),
        }


class TimeframeFuser:
    """Combine signals from multiple :class:`TimeframeLayer` objects.

    Weights are normalised so that relative importance of timeframes is
    preserved regardless of how many timeframes are registered.
    """

    def fuse(self, layers: List[TimeframeLayer]) -> FusedDecision:
        """Fuse the last signal from each layer into one decision.

        Parameters
        ----------
        layers:
            Registered timeframe layers.  Each layer's ``last_signal``
            attribute is used.

        Returns
        -------
        FusedDecision
            The composite decision.
        """
        if not layers:
            return FusedDecision(
                direction=0, confidence=0.0, agreement_pct=0.0,
                bullish_weight=0.0, bearish_weight=0.0, neutral_weight=0.0,
            )

        total_weight = sum(l.weight for l in layers)
        if total_weight <= 0:
            return FusedDecision(
                direction=0, confidence=0.0, agreement_pct=0.0,
                bullish_weight=0.0, bearish_weight=0.0, neutral_weight=0.0,
            )

        bullish = 0.0
        bearish = 0.0
        neutral = 0.0
        layer_signals: Dict[str, Optional[Signal]] = {}

        for layer in layers:
            sig = layer.last_signal
            layer_signals[layer.label] = sig
            if sig is None:
                neutral += layer.weight
            elif sig.direction > 0:
                bullish += layer.weight * sig.confidence
            elif sig.direction < 0:
                bearish += layer.weight * sig.confidence
            else:
                neutral += layer.weight

        # Determine direction by comparing weighted sums
        if bullish > bearish and bullish > neutral:
            direction = 1
            top = bullish
        elif bearish > bullish and bearish > neutral:
            direction = -1
            top = bearish
        else:
            direction = 0
            top = neutral

        # Confidence: proportion of total weight going to winning side
        confidence = top / (total_weight + _EPSILON)

        # Agreement: fraction of layers whose signal matches the direction
        agreeing = sum(
            1 for l in layers
            if (l.last_signal is not None and l.last_signal.direction == direction)
            or (l.last_signal is None and direction == 0)
        )
        agreement_pct = agreeing / len(layers)

        return FusedDecision(
            direction=direction,
            confidence=float(confidence),
            agreement_pct=float(agreement_pct),
            bullish_weight=float(bullish),
            bearish_weight=float(bearish),
            neutral_weight=float(neutral),
            layers=layer_signals,
        )


class MultiTimeframeAnalyzer:
    """Run algorithms across multiple timeframes and fuse their signals.

    This is the primary entry-point for multi-timeframe analysis.

    Parameters
    ----------
    default_algorithms:
        If True, each timeframe layer is pre-populated with
        :class:`~app.trading.algorithms.MeanReversionAlgo`,
        :class:`~app.trading.algorithms.MomentumAlgo`, and
        :class:`~app.trading.algorithms.TrendFollowingAlgo`.

    Examples
    --------
    ::

        analyzer = MultiTimeframeAnalyzer()
        analyzer.add_timeframe("1d", daily_bars, weight=3.0)
        analyzer.add_timeframe("4h", four_hour_bars, weight=2.0)
        analyzer.add_timeframe("1h", hourly_bars, weight=1.0)
        decision = analyzer.fuse()
        print(decision.direction, decision.confidence)
    """

    def __init__(self, default_algorithms: bool = True) -> None:
        self._layers: List[TimeframeLayer] = []
        self._fuser = TimeframeFuser()
        self._default_algorithms = default_algorithms

    def add_timeframe(
        self,
        label: str,
        bars: List[Bar],
        weight: Optional[float] = None,
        algorithms: Optional[List[Algorithm]] = None,
    ) -> "MultiTimeframeAnalyzer":
        """Register a timeframe with its bar data.

        Parameters
        ----------
        label:
            Human-readable timeframe label such as ``"1d"``, ``"4h"``,
            ``"1h"``.
        bars:
            OHLCV bars for this timeframe.
        weight:
            Influence weight.  Looks up :data:`DEFAULT_TF_WEIGHTS` if not
            supplied.  Falls back to 1.0 when the label is not found.
        algorithms:
            Algorithms to run on this timeframe.  Uses default set when
            ``default_algorithms=True`` and this parameter is None.
        """
        if weight is None:
            weight = DEFAULT_TF_WEIGHTS.get(label, 1.0)
        if algorithms is None and self._default_algorithms:
            algorithms = [MeanReversionAlgo(), MomentumAlgo(), TrendFollowingAlgo()]
        layer = TimeframeLayer(
            label=label,
            bars=bars,
            weight=weight,
            algorithms=algorithms or [],
        )
        self._layers.append(layer)
        return self

    def run(self) -> "MultiTimeframeAnalyzer":
        """Process all bars through each layer's algorithms.

        Call this before :meth:`fuse` to populate ``last_signal``.
        """
        for layer in self._layers:
            if not layer.bars or not layer.algorithms:
                layer.last_signal = None
                continue

            aggregator = SignalAggregator(layer.algorithms)
            last: Optional[Signal] = None
            for bar in layer.bars:
                sig = aggregator.on_bar(bar)
                if sig is not None:
                    last = sig

            # Translate "AGG" symbol → actual timeframe label
            if last is not None:
                last = Signal(
                    timestamp=last.timestamp,
                    symbol=layer.label,
                    direction=last.direction,
                    confidence=last.confidence,
                    meta={**last.meta, "timeframe": layer.label},
                )
            layer.last_signal = last
            layer.processed = True
            logger.debug(
                "[MTF] Timeframe=%s  bars=%d  last_signal=%s",
                layer.label, len(layer.bars), last,
            )
        return self

    def fuse(self) -> FusedDecision:
        """Return the fused decision from all registered timeframe layers.

        Calls :meth:`run` automatically if any layer has not been processed
        yet.  Uses the ``processed`` flag to distinguish "not yet run" from
        "run but no signal produced", preventing re-runs on every call.
        """
        # Run only if there are unprocessed layers that have data+algorithms
        needs_run = any(
            not l.processed and l.bars and l.algorithms
            for l in self._layers
        )
        if needs_run:
            self.run()
        return self._fuser.fuse(self._layers)

    def layer_summary(self) -> List[Dict[str, Any]]:
        """Return a per-layer signal summary for inspection."""
        result = []
        for layer in self._layers:
            sig = layer.last_signal
            result.append({
                "timeframe": layer.label,
                "weight": layer.weight,
                "bars": len(layer.bars),
                "direction": sig.direction if sig else 0,
                "confidence": round(sig.confidence, 4) if sig else 0.0,
            })
        return result

    @property
    def layers(self) -> List[TimeframeLayer]:
        return list(self._layers)


# ---------------------------------------------------------------------------
# Regime detection helper
# ---------------------------------------------------------------------------

class MarketRegimeDetector:
    """Classify the current market regime for a given bar series.

    Regimes
    -------
    ``"bull"``   — price trending above long-term EMA, RSI > 50
    ``"bear"``   — price trending below long-term EMA, RSI < 50
    ``"ranging"``— price oscillating within ±1 std of mean

    This is used by the DeFi strategy to decide whether to enter a
    supply-collateral / borrow-USDC position (bearish ending / bull start)
    or to exit (extreme overbought).
    """

    def __init__(self, ema_period: int = 50, rsi_period: int = 14) -> None:
        self.ema_period = ema_period
        self.rsi_period = rsi_period

    def detect(self, bars: List[Bar]) -> str:
        """Return the detected regime string.

        Parameters
        ----------
        bars:
            OHLCV bars in ascending order.  Requires at least
            ``max(ema_period, rsi_period) + 2`` bars.
        """
        if len(bars) < max(self.ema_period, self.rsi_period) + 2:
            return "unknown"

        closes = np.array([b.close for b in bars])
        ema = _ema(closes, self.ema_period)
        rsi_vals = _rsi(closes, self.rsi_period)

        last_price = closes[-1]
        last_ema = ema[-1]
        last_rsi = rsi_vals[-1]

        # Ranging detection: close within 1 std of 20-bar mean
        window = closes[-20:]
        std = np.std(window)
        mean = np.mean(window)
        if abs(last_price - mean) < std:
            return "ranging"

        if last_price > last_ema and last_rsi > 50:
            return "bull"
        if last_price < last_ema and last_rsi < 50:
            return "bear"
        return "ranging"
