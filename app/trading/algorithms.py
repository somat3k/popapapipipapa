"""Trading algorithms, metrics, and the trading chain."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    """Single OHLCV bar."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    """Trading signal produced by an algorithm."""

    timestamp: float
    symbol: str
    direction: int  # +1 buy, -1 sell, 0 neutral
    confidence: float = 0.5
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """A trading order."""

    order_id: str
    symbol: str
    side: str       # "buy" | "sell"
    quantity: float
    price: float    # 0 for market order
    order_type: str = "market"  # "market" | "limit"
    status: str = "pending"


# ---------------------------------------------------------------------------
# Base Algorithm
# ---------------------------------------------------------------------------

class Algorithm:
    """Abstract trading algorithm base class."""

    def __init__(self, name: str = "Algorithm") -> None:
        self.name = name
        self._signals: List[Signal] = []

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        raise NotImplementedError

    def generate_signals(self, bars: List[Bar]) -> List[Signal]:
        self._signals = []
        for bar in bars:
            sig = self.on_bar(bar)
            if sig is not None:
                self._signals.append(sig)
        return self._signals

    def get_signals(self) -> List[Signal]:
        return list(self._signals)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def _sma(prices: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(prices, np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.zeros(len(prices))
    avg_loss = np.zeros(len(prices))
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-12))
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi


def _bollinger(
    prices: np.ndarray, period: int = 20, num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = _sma(prices, period)
    std = np.array([
        np.std(prices[max(0, i - period + 1) : i + 1]) if i >= period - 1 else np.nan
        for i in range(len(prices))
    ])
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ---------------------------------------------------------------------------
# Concrete Algorithms
# ---------------------------------------------------------------------------

class MeanReversionAlgo(Algorithm):
    """Bollinger Band mean-reversion strategy."""

    def __init__(self, symbol: str = "MATIC/USDC", period: int = 20, num_std: float = 2.0) -> None:
        super().__init__(f"MeanReversion[{symbol}]")
        self.symbol = symbol
        self.period = period
        self.num_std = num_std
        self._price_buffer: List[float] = []

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._price_buffer.append(bar.close)
        if len(self._price_buffer) < self.period:
            return None
        prices = np.array(self._price_buffer[-self.period :])
        upper, mid, lower = _bollinger(prices, self.period, self.num_std)
        price = bar.close
        if price <= lower[-1]:
            return Signal(bar.timestamp, self.symbol, +1, confidence=0.7)
        if price >= upper[-1]:
            return Signal(bar.timestamp, self.symbol, -1, confidence=0.7)
        return None


class MomentumAlgo(Algorithm):
    """RSI + MACD momentum strategy."""

    def __init__(
        self, symbol: str = "MATIC/USDC", rsi_period: int = 14, fast: int = 12, slow: int = 26
    ) -> None:
        super().__init__(f"Momentum[{symbol}]")
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.fast = fast
        self.slow = slow
        self._prices: List[float] = []

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._prices.append(bar.close)
        n = len(self._prices)
        if n < self.slow + 1:
            return None
        arr = np.array(self._prices)
        rsi = _rsi(arr, self.rsi_period)
        ema_fast = _ema(arr, self.fast)
        ema_slow = _ema(arr, self.slow)
        macd = ema_fast - ema_slow
        if rsi[-1] < 30 and macd[-1] > 0:
            return Signal(bar.timestamp, self.symbol, +1, confidence=0.65)
        if rsi[-1] > 70 and macd[-1] < 0:
            return Signal(bar.timestamp, self.symbol, -1, confidence=0.65)
        return None


class TrendFollowingAlgo(Algorithm):
    """EMA crossover trend following strategy."""

    def __init__(self, symbol: str = "MATIC/USDC", fast: int = 9, slow: int = 21) -> None:
        super().__init__(f"TrendFollowing[{symbol}]")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self._prices: List[float] = []
        self._prev_cross: int = 0

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._prices.append(bar.close)
        n = len(self._prices)
        if n < self.slow:
            return None
        arr = np.array(self._prices)
        ema_f = _ema(arr, self.fast)
        ema_s = _ema(arr, self.slow)
        cross = +1 if ema_f[-1] > ema_s[-1] else -1
        if cross != self._prev_cross:
            self._prev_cross = cross
            return Signal(bar.timestamp, self.symbol, cross, confidence=0.6)
        return None


class SignalAggregator:
    """Aggregates signals from multiple algorithms with configurable weights."""

    def __init__(self, algorithms: Optional[List[Algorithm]] = None) -> None:
        self._algos: List[Tuple[Algorithm, float]] = []
        for a in (algorithms or []):
            self.add(a)

    def add(self, algo: Algorithm, weight: float = 1.0) -> None:
        self._algos.append((algo, weight))

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        weighted_dir = 0.0
        total_weight = 0.0
        for algo, weight in self._algos:
            sig = algo.on_bar(bar)
            if sig is not None:
                weighted_dir += sig.direction * sig.confidence * weight
                total_weight += weight
        if total_weight == 0:
            return None
        final = weighted_dir / total_weight
        direction = int(np.sign(final))
        if direction == 0:
            return None
        return Signal(bar.timestamp, "AGG", direction, confidence=abs(final))


# ---------------------------------------------------------------------------
# Trading Metrics
# ---------------------------------------------------------------------------

class TradingMetrics:
    """Compute strategy performance metrics from a return series."""

    def __init__(self, returns: np.ndarray, risk_free_rate: float = 0.04) -> None:
        self.returns = returns
        self.risk_free_rate = risk_free_rate

    def sharpe(self, annualise: float = 252.0) -> float:
        excess = self.returns - self.risk_free_rate / annualise
        std = np.std(excess)
        if std < 1e-12:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(annualise))

    def sortino(self, annualise: float = 252.0) -> float:
        excess = self.returns - self.risk_free_rate / annualise
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")
        dd_std = np.std(downside)
        if dd_std < 1e-12:
            return float("inf")
        return float(np.mean(excess) / dd_std * np.sqrt(annualise))

    def max_drawdown(self) -> float:
        cum = np.cumprod(1 + self.returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-12)
        return float(np.min(dd))

    def win_rate(self) -> float:
        wins = np.sum(self.returns > 0)
        return float(wins / len(self.returns)) if len(self.returns) > 0 else 0.0

    def profit_factor(self) -> float:
        gains = np.sum(self.returns[self.returns > 0])
        losses = abs(np.sum(self.returns[self.returns < 0]))
        return float(gains / losses) if losses > 0 else float("inf")

    def calmar(self) -> float:
        ann_return = float(np.mean(self.returns) * 252)
        mdd = abs(self.max_drawdown())
        return ann_return / mdd if mdd > 1e-12 else float("inf")

    def summary(self) -> Dict[str, float]:
        return {
            "sharpe": self.sharpe(),
            "sortino": self.sortino(),
            "max_drawdown": self.max_drawdown(),
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "calmar": self.calmar(),
        }


# ---------------------------------------------------------------------------
# Trading Chain
# ---------------------------------------------------------------------------

class ChainStep:
    """Abstract chain step."""

    def __init__(self, name: str) -> None:
        self.name = name

    def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class DataIngestionStep(ChainStep):
    def __init__(self, bars: Optional[List[Bar]] = None) -> None:
        super().__init__("DataIngestion")
        self._bars = bars or []

    def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        ctx["bars"] = self._bars
        logger.debug("[%s] Ingested %d bars.", self.name, len(self._bars))
        return ctx


class FeatureEngineeringStep(ChainStep):
    def __init__(self) -> None:
        super().__init__("FeatureEngineering")

    def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        bars: List[Bar] = ctx.get("bars", [])
        if not bars:
            ctx["features"] = np.array([])
            return ctx
        closes = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars])
        n = len(closes)
        features = np.zeros((n, 5))
        features[:, 0] = closes
        features[:, 1] = volumes
        if n > 1:
            features[1:, 2] = np.diff(closes) / (closes[:-1] + 1e-12)
        ema9 = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        features[:, 3] = ema9
        features[:, 4] = ema21
        ctx["features"] = features
        ctx["closes"] = closes
        return ctx


class SignalGenerationStep(ChainStep):
    def __init__(self, aggregator: Optional[SignalAggregator] = None) -> None:
        super().__init__("SignalGeneration")
        self._aggregator = aggregator or SignalAggregator(
            [MeanReversionAlgo(), MomentumAlgo(), TrendFollowingAlgo()]
        )

    def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        bars: List[Bar] = ctx.get("bars", [])
        signals = []
        for bar in bars:
            sig = self._aggregator.on_bar(bar)
            if sig is not None:
                signals.append(sig)
        ctx["signals"] = signals
        logger.debug("[%s] Generated %d signals.", self.name, len(signals))
        return ctx


class RiskFilterStep(ChainStep):
    def __init__(self, max_position: float = 10_000.0) -> None:
        super().__init__("RiskFilter")
        self.max_position = max_position

    def process(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        portfolio_val = ctx.get("portfolio_value", 100_000.0)
        filtered = []
        for sig in ctx.get("signals", []):
            size = portfolio_val * 0.02  # 2% of portfolio per signal
            if size <= self.max_position:
                filtered.append(sig)
        ctx["filtered_signals"] = filtered
        return ctx


class TradingChain:
    """Ordered pipeline of ChainStep processors.

    Usage
    -----
    chain = TradingChain()
    chain.add_step(DataIngestionStep(bars))
    chain.add_step(FeatureEngineeringStep())
    chain.add_step(SignalGenerationStep())
    chain.add_step(RiskFilterStep())
    result = chain.run()
    """

    def __init__(self) -> None:
        self._steps: List[ChainStep] = []

    def add_step(self, step: ChainStep) -> "TradingChain":
        self._steps.append(step)
        return self

    def run(self, initial_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx: Dict[str, Any] = initial_ctx or {}
        for step in self._steps:
            t0 = time.perf_counter()
            try:
                ctx = step.process(ctx)
            except Exception:
                logger.exception("[TradingChain] Step '%s' failed.", step.name)
                ctx["chain_error"] = step.name
                break
            elapsed = time.perf_counter() - t0
            ctx.setdefault("step_timings", {})[step.name] = round(elapsed * 1000, 2)
        return ctx
