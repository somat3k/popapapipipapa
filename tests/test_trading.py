"""Tests for trading algorithms, metrics, and the trading chain."""

import time

import numpy as np
import pytest

from app.trading.algorithms import (
    Bar,
    DataIngestionStep,
    FeatureEngineeringStep,
    MeanReversionAlgo,
    MomentumAlgo,
    Order,
    RiskFilterStep,
    Signal,
    SignalAggregator,
    SignalGenerationStep,
    TradingChain,
    TradingMetrics,
    TrendFollowingAlgo,
    _bollinger,
    _ema,
    _rsi,
    _sma,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_bars(n: int = 50, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    prices = np.cumsum(rng.normal(0, 0.01, n)) + 1.0
    prices = np.exp(prices) * 0.85
    return [
        Bar(float(i), prices[i], prices[i] * 1.005, prices[i] * 0.995,
            prices[i], float(rng.integers(100_000, 1_000_000)))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Technical indicator tests
# ---------------------------------------------------------------------------

def test_ema_length():
    prices = np.ones(20)
    ema = _ema(prices, 5)
    assert len(ema) == 20


def test_ema_constant_series():
    prices = np.full(20, 2.0)
    ema = _ema(prices, 5)
    assert np.allclose(ema, 2.0)


def test_sma_length():
    prices = np.arange(1, 21, dtype=float)
    sma = _sma(prices, 5)
    assert len(sma) == 20


def test_sma_known_value():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sma = _sma(prices, 3)
    assert sma[2] == pytest.approx(2.0)
    assert sma[4] == pytest.approx(4.0)


def test_rsi_range():
    prices = np.cumsum(np.random.default_rng(0).normal(0, 0.01, 50)) + 1.0
    rsi = _rsi(prices, period=14)
    valid = rsi[~np.isnan(rsi)]
    assert np.all(valid >= 0) and np.all(valid <= 100)


def test_bollinger_upper_above_lower():
    prices = np.cumsum(np.random.default_rng(7).normal(0, 0.01, 50)) + 1.0
    upper, mid, lower = _bollinger(prices, 20, 2.0)
    valid = ~(np.isnan(upper) | np.isnan(lower))
    assert np.all(upper[valid] >= lower[valid])


# ---------------------------------------------------------------------------
# Bar and Signal dataclasses
# ---------------------------------------------------------------------------

def test_bar_fields():
    bar = Bar(1000.0, 1.0, 1.1, 0.9, 1.05, 50000.0)
    assert bar.close == 1.05
    assert bar.volume == 50000.0


def test_signal_direction():
    sig = Signal(timestamp=0.0, symbol="X", direction=1, confidence=0.8)
    assert sig.direction == 1
    assert sig.confidence == 0.8


# ---------------------------------------------------------------------------
# MeanReversionAlgo tests
# ---------------------------------------------------------------------------

def test_mean_reversion_returns_signals():
    bars = make_bars(60)
    algo = MeanReversionAlgo(period=10)
    signals = algo.generate_signals(bars)
    assert isinstance(signals, list)


def test_mean_reversion_no_signal_before_period():
    bars = make_bars(5)
    algo = MeanReversionAlgo(period=20)
    signals = algo.generate_signals(bars)
    assert signals == []


def test_mean_reversion_signal_direction():
    bars = make_bars(50)
    algo = MeanReversionAlgo(period=10)
    algo.generate_signals(bars)
    for sig in algo.get_signals():
        assert sig.direction in (-1, 0, 1)


# ---------------------------------------------------------------------------
# MomentumAlgo tests
# ---------------------------------------------------------------------------

def test_momentum_algo_runs():
    bars = make_bars(60)
    algo = MomentumAlgo()
    signals = algo.generate_signals(bars)
    assert isinstance(signals, list)


def test_momentum_no_signal_before_slow():
    bars = make_bars(10)
    algo = MomentumAlgo(slow=26)
    signals = algo.generate_signals(bars)
    assert signals == []


# ---------------------------------------------------------------------------
# TrendFollowingAlgo tests
# ---------------------------------------------------------------------------

def test_trend_following_runs():
    bars = make_bars(50)
    algo = TrendFollowingAlgo(fast=5, slow=10)
    signals = algo.generate_signals(bars)
    assert isinstance(signals, list)


def test_trend_following_at_crossover():
    # Manually craft a crossover: rising prices → golden cross
    prices = [1.0] * 10 + [i * 0.05 + 1.0 for i in range(30)]
    bars = [Bar(float(i), p, p, p, p, 1000.0) for i, p in enumerate(prices)]
    algo = TrendFollowingAlgo(fast=5, slow=10)
    signals = algo.generate_signals(bars)
    # At least one buy signal expected during the uptrend
    assert any(s.direction == 1 for s in signals)


# ---------------------------------------------------------------------------
# SignalAggregator tests
# ---------------------------------------------------------------------------

def test_aggregator_combines():
    bars = make_bars(80)
    agg = SignalAggregator([MeanReversionAlgo(period=10), MomentumAlgo()])
    signals = []
    for bar in bars:
        sig = agg.on_bar(bar)
        if sig:
            signals.append(sig)
    for s in signals:
        assert s.symbol == "AGG"
        assert s.direction in (-1, 1)


def test_aggregator_no_algos_returns_none():
    agg = SignalAggregator([])
    bar = Bar(0.0, 1.0, 1.0, 1.0, 1.0, 1000.0)
    assert agg.on_bar(bar) is None


# ---------------------------------------------------------------------------
# TradingMetrics tests
# ---------------------------------------------------------------------------

def test_sharpe_zero_variance():
    returns = np.zeros(100)
    metrics = TradingMetrics(returns)
    assert metrics.sharpe() == 0.0


def test_sharpe_nonzero():
    # Mix of positive and slightly negative returns with positive mean
    rng = np.random.default_rng(seed=99)
    returns = rng.normal(0.002, 0.005, 252)  # strong positive mean, low vol
    # Force the mean to be positive even if the seed gives negatives
    returns = np.abs(returns) * 0.001 + 0.001
    metrics = TradingMetrics(returns, risk_free_rate=0.0)
    assert metrics.sharpe() > 0


def test_max_drawdown_non_positive():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0, 0.01, 200)
    metrics = TradingMetrics(returns)
    assert metrics.max_drawdown() <= 0.0


def test_win_rate_range():
    returns = np.array([1.0, -1.0, 0.5, -0.5])
    metrics = TradingMetrics(returns)
    wr = metrics.win_rate()
    assert 0.0 <= wr <= 1.0


def test_win_rate_all_positive():
    returns = np.ones(10) * 0.01
    metrics = TradingMetrics(returns)
    assert metrics.win_rate() == pytest.approx(1.0)


def test_win_rate_all_negative():
    returns = -np.ones(10) * 0.01
    metrics = TradingMetrics(returns)
    assert metrics.win_rate() == pytest.approx(0.0)


def test_profit_factor():
    returns = np.array([0.1, -0.05, 0.2, -0.05])
    metrics = TradingMetrics(returns)
    pf = metrics.profit_factor()
    assert pf == pytest.approx(0.3 / 0.1)


def test_sortino_higher_than_sharpe_low_downside():
    returns = np.array([0.02] * 200 + [-0.001])
    metrics = TradingMetrics(returns)
    assert metrics.sortino() >= metrics.sharpe()


def test_summary_keys():
    returns = np.random.default_rng(3).normal(0, 0.01, 100)
    summary = TradingMetrics(returns).summary()
    assert set(summary.keys()) == {
        "sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor", "calmar"
    }


# ---------------------------------------------------------------------------
# TradingChain tests
# ---------------------------------------------------------------------------

def test_trading_chain_full():
    bars = make_bars(60)
    chain = (
        TradingChain()
        .add_step(DataIngestionStep(bars))
        .add_step(FeatureEngineeringStep())
        .add_step(SignalGenerationStep(
            SignalAggregator([MeanReversionAlgo(period=10), MomentumAlgo()])
        ))
        .add_step(RiskFilterStep(max_position=10_000.0))
    )
    ctx = chain.run({"portfolio_value": 100_000.0})
    assert "bars" in ctx
    assert "features" in ctx
    assert "signals" in ctx
    assert "filtered_signals" in ctx
    assert "step_timings" in ctx


def test_trading_chain_empty_bars():
    chain = (
        TradingChain()
        .add_step(DataIngestionStep([]))
        .add_step(FeatureEngineeringStep())
    )
    ctx = chain.run()
    assert ctx["features"].size == 0


def test_trading_chain_step_timings():
    bars = make_bars(30)
    chain = (
        TradingChain()
        .add_step(DataIngestionStep(bars))
        .add_step(FeatureEngineeringStep())
    )
    ctx = chain.run()
    timings = ctx["step_timings"]
    assert "DataIngestion" in timings
    assert "FeatureEngineering" in timings
    assert all(v >= 0 for v in timings.values())


def test_feature_engineering_shape():
    bars = make_bars(40)
    step = FeatureEngineeringStep()
    ctx = step.process({"bars": bars})
    features = ctx["features"]
    assert features.shape == (40, 5)


def test_risk_filter_blocks_oversized():
    sig = Signal(timestamp=0.0, symbol="X", direction=1, confidence=0.9)
    step = RiskFilterStep(max_position=0.0)  # zero threshold → all filtered
    ctx = step.process({"signals": [sig], "portfolio_value": 100_000.0})
    assert ctx["filtered_signals"] == []
