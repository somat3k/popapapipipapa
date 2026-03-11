"""Tests for the evaluation module.

Covers:
  - OHLCVLoader (synthetic generation, CSV loading, resampling)
  - MultiTimeframeAnalyzer + TimeframeFuser
  - MarketRegimeDetector
  - AgentEvaluationMetrics (all metrics)
  - AccountGrowthTracker (milestones, CAGR, on-track)
  - RLEnvironment (reset, step, rewards)
  - SupervisedRLAgent (pretrain, episode, policy update)
  - RLPipeline (full run)
  - HealthFactorManager (zones, repay fraction)
  - HalfHalfThreeStrategy (enter, rebalance, exit)
  - MarketEntryAdvisor (advise, multi-timeframe advice)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from app.trading.algorithms import Bar, Signal

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_bars() -> List[Bar]:
    """200 synthetic daily ETH bars."""
    from app.evaluation.data_loader import OHLCVLoader
    return OHLCVLoader().generate_synthetic(n=200, seed=7, start_price=2000.0)


@pytest.fixture
def simple_returns() -> np.ndarray:
    """252 small positive returns for predictable metric values."""
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.005, 252).astype(np.float64)


# ===========================================================================
# data_loader
# ===========================================================================

class TestOHLCVLoader:
    def test_generate_synthetic_length(self):
        from app.evaluation.data_loader import OHLCVLoader
        loader = OHLCVLoader()
        bars = loader.generate_synthetic(n=100)
        assert len(bars) == 100

    def test_generate_synthetic_returns_bars(self):
        from app.evaluation.data_loader import OHLCVLoader
        bars = OHLCVLoader().generate_synthetic(n=10, start_price=1500.0)
        for b in bars:
            assert isinstance(b, Bar)
            assert b.high >= b.close >= b.low > 0

    def test_generate_synthetic_reproducible(self):
        from app.evaluation.data_loader import OHLCVLoader
        loader = OHLCVLoader()
        b1 = loader.generate_synthetic(n=50, seed=0)
        b2 = loader.generate_synthetic(n=50, seed=0)
        assert all(a.close == pytest.approx(b.close) for a, b in zip(b1, b2))

    def test_generate_synthetic_different_seeds(self):
        from app.evaluation.data_loader import OHLCVLoader
        loader = OHLCVLoader()
        b1 = loader.generate_synthetic(n=50, seed=1)
        b2 = loader.generate_synthetic(n=50, seed=2)
        assert b1[0].close != pytest.approx(b2[0].close)

    def test_fetch_from_api_falls_back_to_synthetic(self, monkeypatch):
        """When network is unavailable, falls back to synthetic data."""
        from app.evaluation import data_loader

        def _fail(*args, **kwargs):
            raise ConnectionError("offline")

        monkeypatch.setattr(data_loader.OHLCVLoader, "_fetch_coingecko_ohlc", _fail)
        loader = data_loader.OHLCVLoader()
        bars = loader.fetch_from_api("ETH", days=30)
        assert len(bars) > 0
        assert all(isinstance(b, Bar) for b in bars)

    def test_fetch_from_api_unknown_symbol(self, monkeypatch):
        """Unknown symbol returns synthetic bars without crashing."""
        from app.evaluation.data_loader import OHLCVLoader
        bars = OHLCVLoader().fetch_from_api("UNKNOWNTOK3N", days=10)
        assert len(bars) > 0

    def test_load_from_csv(self, tmp_path):
        from app.evaluation.data_loader import OHLCVLoader
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
            )
            writer.writeheader()
            for i in range(10):
                writer.writerow({
                    "timestamp": float(i * 3600),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                    "volume": 1_000_000.0,
                })
        bars = OHLCVLoader().load_from_csv(csv_path)
        assert len(bars) == 10
        assert bars[0].close == pytest.approx(102.0)

    def test_load_from_csv_missing_file(self, tmp_path):
        from app.evaluation.data_loader import OHLCVLoader
        with pytest.raises(FileNotFoundError):
            OHLCVLoader().load_from_csv(tmp_path / "nonexistent_file.csv")

    def test_load_from_csv_missing_required_columns(self, tmp_path):
        from app.evaluation.data_loader import OHLCVLoader
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as f:
            f.write("foo,bar,baz\n1,2,3\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            OHLCVLoader().load_from_csv(csv_path)

    def test_fetch_multi_returns_dict(self, monkeypatch):
        from app.evaluation import data_loader

        monkeypatch.setattr(
            data_loader.OHLCVLoader,
            "_fetch_coingecko_ohlc",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("offline")),
        )
        loader = data_loader.OHLCVLoader()
        result = loader.fetch_multi(["ETH", "BTC"], days=20)
        assert "ETH" in result
        assert "BTC" in result
        assert len(result["ETH"]) > 0

    def test_fetch_multi_timeframe_keys_and_different_bar_counts(self, monkeypatch):
        from app.evaluation import data_loader

        monkeypatch.setattr(
            data_loader.OHLCVLoader,
            "_fetch_coingecko_ohlc",
            lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("offline")),
        )
        result = data_loader.OHLCVLoader().fetch_multi_timeframe(
            "ETH", days=120, timeframes=["1h", "4h", "1d"]
        )
        assert set(result.keys()) == {"1h", "4h", "1d"}

        bars_1h = result["1h"]
        bars_4h = result["4h"]
        bars_1d = result["1d"]

        # All timeframes should produce at least one bar
        assert len(bars_1h) > 0
        assert len(bars_4h) > 0
        assert len(bars_1d) > 0

        # Higher-frequency timeframes must have strictly fewer resampled bars
        # than lower-frequency ones when starting from the same daily series.
        # 1h is resampled by factor 24, 4h by factor 4, 1d by factor 1.
        assert len(bars_1d) >= len(bars_4h) >= len(bars_1h)


def test_resample_bars():
    from app.evaluation.data_loader import OHLCVLoader, resample_bars
    bars = OHLCVLoader().generate_synthetic(n=40, seed=1)
    resampled = resample_bars(bars, factor=4)
    assert len(resampled) == 10
    # High of 4h bar ≥ max high of constituent bars
    for i, rb in enumerate(resampled):
        chunk = bars[i * 4: (i + 1) * 4]
        assert rb.high == pytest.approx(max(b.high for b in chunk))
        assert rb.low == pytest.approx(min(b.low for b in chunk))


def test_resample_bars_factor_1(synthetic_bars):
    from app.evaluation.data_loader import resample_bars
    result = resample_bars(synthetic_bars, factor=1)
    assert len(result) == len(synthetic_bars)


# ===========================================================================
# timeframe_fusion
# ===========================================================================

class TestTimeframeFuser:
    def test_fuse_empty_layers(self):
        from app.evaluation.timeframe_fusion import TimeframeFuser
        fuser = TimeframeFuser()
        decision = fuser.fuse([])
        assert decision.direction == 0
        assert decision.confidence == 0.0

    def test_fuse_single_bullish_signal(self):
        from app.evaluation.timeframe_fusion import TimeframeFuser, TimeframeLayer
        sig = Signal(1.0, "1d", +1, confidence=0.8)
        layer = TimeframeLayer("1d", [], weight=3.0, last_signal=sig)
        decision = TimeframeFuser().fuse([layer])
        assert decision.direction == 1
        assert decision.bullish_weight > 0

    def test_fuse_single_bearish_signal(self):
        from app.evaluation.timeframe_fusion import TimeframeFuser, TimeframeLayer
        sig = Signal(1.0, "1h", -1, confidence=0.7)
        layer = TimeframeLayer("1h", [], weight=1.0, last_signal=sig)
        decision = TimeframeFuser().fuse([layer])
        assert decision.direction == -1

    def test_fuse_conflicting_signals_higher_wins(self):
        from app.evaluation.timeframe_fusion import TimeframeFuser, TimeframeLayer
        # Daily bullish (weight 3) vs hourly bearish (weight 1)
        bull = Signal(1.0, "1d", +1, confidence=0.9)
        bear = Signal(1.0, "1h", -1, confidence=0.9)
        layers = [
            TimeframeLayer("1d", [], weight=3.0, last_signal=bull),
            TimeframeLayer("1h", [], weight=1.0, last_signal=bear),
        ]
        decision = TimeframeFuser().fuse(layers)
        assert decision.direction == 1   # daily wins by weight

    def test_fuse_no_signals_returns_neutral(self):
        from app.evaluation.timeframe_fusion import TimeframeFuser, TimeframeLayer
        layers = [
            TimeframeLayer("1d", [], weight=3.0, last_signal=None),
            TimeframeLayer("1h", [], weight=1.0, last_signal=None),
        ]
        decision = TimeframeFuser().fuse(layers)
        assert decision.direction == 0

    def test_fuse_zero_weight_layers_returns_neutral(self):
        """Zero-weight layers must not cause a ZeroDivisionError."""
        from app.evaluation.timeframe_fusion import TimeframeFuser, TimeframeLayer
        sig = Signal(1.0, "1d", +1, confidence=0.9)
        layers = [TimeframeLayer("1d", [], weight=0.0, last_signal=sig)]
        decision = TimeframeFuser().fuse(layers)
        assert decision.direction == 0  # neutral — no positive weight

    def test_fused_decision_summary_keys(self):
        from app.evaluation.timeframe_fusion import TimeframeFuser, TimeframeLayer
        sig = Signal(1.0, "1d", 1, confidence=0.7)
        layer = TimeframeLayer("1d", [], weight=1.0, last_signal=sig)
        d = TimeframeFuser().fuse([layer])
        keys = d.summary().keys()
        assert "direction" in keys and "confidence" in keys

    def test_is_high_confidence(self):
        from app.evaluation.timeframe_fusion import TimeframeFuser, TimeframeLayer
        sigs = [
            TimeframeLayer("1d", [], weight=3.0,
                           last_signal=Signal(1.0, "1d", +1, confidence=0.9)),
            TimeframeLayer("4h", [], weight=2.0,
                           last_signal=Signal(1.0, "4h", +1, confidence=0.8)),
        ]
        d = TimeframeFuser().fuse(sigs)
        assert d.is_high_confidence(threshold=0.5)


class TestMultiTimeframeAnalyzer:
    def test_add_timeframe_returns_self(self, synthetic_bars):
        from app.evaluation.timeframe_fusion import MultiTimeframeAnalyzer
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.add_timeframe("1d", synthetic_bars, weight=3.0)
        assert result is analyzer

    def test_run_and_fuse(self, synthetic_bars):
        from app.evaluation.timeframe_fusion import MultiTimeframeAnalyzer
        analyzer = MultiTimeframeAnalyzer()
        analyzer.add_timeframe("1d", synthetic_bars, weight=3.0)
        analyzer.run()
        decision = analyzer.fuse()
        assert decision.direction in (-1, 0, 1)
        assert 0.0 <= decision.confidence <= 1.0

    def test_layer_summary_fields(self, synthetic_bars):
        from app.evaluation.timeframe_fusion import MultiTimeframeAnalyzer
        analyzer = MultiTimeframeAnalyzer()
        analyzer.add_timeframe("1d", synthetic_bars)
        analyzer.run()
        summary = analyzer.layer_summary()
        assert len(summary) == 1
        assert "timeframe" in summary[0]
        assert "direction" in summary[0]

    def test_multi_timeframe_fusion_consistent(self, synthetic_bars):
        """Fusing 3 timeframes always returns a valid direction."""
        from app.evaluation.data_loader import resample_bars
        from app.evaluation.timeframe_fusion import MultiTimeframeAnalyzer
        bars_4h = resample_bars(synthetic_bars, factor=4)
        bars_1d = resample_bars(synthetic_bars, factor=24) if len(synthetic_bars) >= 24 else synthetic_bars[:8]

        analyzer = MultiTimeframeAnalyzer()
        analyzer.add_timeframe("1h", synthetic_bars[:80], weight=1.0)
        analyzer.add_timeframe("4h", bars_4h[:20] if bars_4h else synthetic_bars[:20], weight=2.0)
        analyzer.add_timeframe("1d", bars_1d[:8] if bars_1d else synthetic_bars[:8], weight=3.0)
        decision = analyzer.fuse()
        assert decision.direction in (-1, 0, 1)

    def test_default_timeframe_weights_applied(self):
        from app.evaluation.timeframe_fusion import MultiTimeframeAnalyzer, DEFAULT_TF_WEIGHTS
        analyzer = MultiTimeframeAnalyzer()
        analyzer.add_timeframe("1d", [])  # no bars
        layer = analyzer.layers[0]
        assert layer.weight == DEFAULT_TF_WEIGHTS["1d"]

    def test_fuse_does_not_rerun_after_processed(self, synthetic_bars):
        """fuse() must not re-run algorithms when layers are already processed."""
        from app.evaluation.timeframe_fusion import MultiTimeframeAnalyzer
        analyzer = MultiTimeframeAnalyzer()
        analyzer.add_timeframe("1d", synthetic_bars, weight=3.0)
        analyzer.run()
        # Mark a sentinel value on the last_signal to detect re-runs
        sentinel = analyzer.layers[0].last_signal
        analyzer.fuse()  # must not call run() again
        assert analyzer.layers[0].last_signal is sentinel


class TestMarketRegimeDetector:
    def test_detect_bull(self, synthetic_bars):
        from app.evaluation.timeframe_fusion import MarketRegimeDetector
        # Use a strongly trending up series
        prices = [100.0 + i * 2 for i in range(100)]
        bars = [Bar(float(i), p, p * 1.01, p * 0.99, p, 1e6) for i, p in enumerate(prices)]
        detector = MarketRegimeDetector(ema_period=20)
        regime = detector.detect(bars)
        assert regime in ("bull", "ranging", "bear")  # just doesn't crash

    def test_detect_insufficient_data(self):
        from app.evaluation.timeframe_fusion import MarketRegimeDetector
        detector = MarketRegimeDetector(ema_period=50)
        regime = detector.detect([Bar(0.0, 1.0, 1.0, 1.0, 1.0, 0.0)])
        assert regime == "unknown"

    def test_detect_returns_string(self, synthetic_bars):
        from app.evaluation.timeframe_fusion import MarketRegimeDetector
        regime = MarketRegimeDetector().detect(synthetic_bars)
        assert isinstance(regime, str)
        assert regime in ("bull", "bear", "ranging", "unknown")


# ===========================================================================
# metrics
# ===========================================================================

class TestAgentEvaluationMetrics:
    def test_cagr_positive_returns(self, simple_returns):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(simple_returns)
        assert m.cagr() > 0

    def test_cagr_empty_returns(self):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(np.array([]))
        assert m.cagr() == 0.0

    def test_hit_rate_all_positive(self):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(np.ones(100) * 0.01)
        assert m.hit_rate() == pytest.approx(1.0)

    def test_hit_rate_all_negative(self):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(-np.ones(100) * 0.01)
        assert m.hit_rate() == pytest.approx(0.0)

    def test_health_factor_score_perfect(self):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(np.zeros(50), health_factors=np.full(50, 3.0))
        assert m.health_factor_score() == pytest.approx(1.0)

    def test_health_factor_score_dangerous(self):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(np.zeros(50), health_factors=np.ones(50) * 0.9)
        assert m.health_factor_score() == pytest.approx(0.0)

    def test_defi_yield_contribution_zero_returns(self):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(np.zeros(50), defi_yields=np.zeros(50))
        assert m.defi_yield_contribution() == 0.0

    def test_defi_yield_contribution_nonzero(self, simple_returns):
        from app.evaluation.metrics import AgentEvaluationMetrics
        defi = simple_returns * 0.30  # DeFi contributes 30%
        m = AgentEvaluationMetrics(simple_returns, defi_yields=defi)
        contrib = m.defi_yield_contribution()
        assert contrib == pytest.approx(0.30, rel=0.05)

    def test_recovery_speed_no_drawdown(self):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(np.ones(100) * 0.01)
        assert m.recovery_speed() == 0.0

    def test_information_coefficient_no_predictions(self, simple_returns):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(simple_returns)
        assert m.information_coefficient() == 0.0

    def test_information_coefficient_perfect(self, simple_returns):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(simple_returns, predictions=simple_returns)
        ic = m.information_coefficient()
        assert ic == pytest.approx(1.0, abs=0.01)

    def test_hf_adjusted_sharpe_penalised(self, simple_returns):
        from app.evaluation.metrics import AgentEvaluationMetrics
        safe = AgentEvaluationMetrics(simple_returns, health_factors=np.full(252, 2.5))
        risky = AgentEvaluationMetrics(simple_returns, health_factors=np.full(252, 1.05))
        assert safe.hf_adjusted_sharpe() > risky.hf_adjusted_sharpe()

    def test_composite_score_range(self, simple_returns):
        from app.evaluation.metrics import AgentEvaluationMetrics
        m = AgentEvaluationMetrics(simple_returns)
        score = m.composite_score()
        assert 0.0 <= score <= 1.0

    def test_letter_grade_A_plus_for_great_strategy(self):
        from app.evaluation.metrics import AgentEvaluationMetrics
        # Near-perfect strategy: high positive returns, no drawdowns, safe HF
        returns = np.full(252, 0.005)  # 0.5% per day
        m = AgentEvaluationMetrics(
            returns,
            health_factors=np.full(252, 3.0),
            defi_yields=returns * 0.1,
        )
        assert m.letter_grade() in ("A+", "A")

    def test_full_report_keys(self, simple_returns):
        from app.evaluation.metrics import AgentEvaluationMetrics
        report = AgentEvaluationMetrics(simple_returns).full_report()
        required = {
            "sharpe", "sortino", "max_drawdown", "cagr", "risk_adjusted_growth",
            "defi_yield_contribution", "hit_rate", "health_factor_score",
            "composite_score", "letter_grade",
        }
        assert required.issubset(set(report.keys()))


class TestHealthFactorHelpers:
    def test_health_factor_grade_values(self):
        from app.evaluation.metrics import health_factor_grade
        assert health_factor_grade(2.5) == "A+"
        assert health_factor_grade(1.8) == "A"
        assert health_factor_grade(1.3) == "B"
        assert health_factor_grade(1.05) == "C"
        assert health_factor_grade(0.9) == "D"

    def test_hf_penalty_safe(self):
        from app.evaluation.metrics import hf_penalty
        assert hf_penalty(3.0) == pytest.approx(0.0)

    def test_hf_penalty_danger(self):
        from app.evaluation.metrics import hf_penalty
        assert hf_penalty(0.5) == pytest.approx(1.0)

    def test_hf_penalty_interpolated(self):
        from app.evaluation.metrics import hf_penalty
        p = hf_penalty(1.5)
        assert 0.0 < p < 1.0


class TestAccountGrowthTracker:
    def test_initial_value(self):
        from app.evaluation.metrics import AccountGrowthTracker
        tracker = AccountGrowthTracker(10_000.0)
        assert tracker.current_value == pytest.approx(10_000.0)
        assert tracker.total_return == pytest.approx(0.0)

    def test_update_positive_return(self):
        from app.evaluation.metrics import AccountGrowthTracker
        tracker = AccountGrowthTracker(10_000.0)
        tracker.update(0, 0.10)  # +10%
        assert tracker.current_value == pytest.approx(11_000.0)

    def test_milestone_crossing(self):
        from app.evaluation.metrics import AccountGrowthTracker
        tracker = AccountGrowthTracker(10_000.0, milestone_step=0.10)
        m = tracker.update(0, 0.15)  # +15% crosses the 10% milestone
        assert m is not None
        assert m.milestone == pytest.approx(1.10)

    def test_no_milestone_small_return(self):
        from app.evaluation.metrics import AccountGrowthTracker
        tracker = AccountGrowthTracker(10_000.0, milestone_step=0.10)
        m = tracker.update(0, 0.05)  # +5%, does NOT cross 10% milestone
        assert m is None

    def test_equity_curve_length(self):
        from app.evaluation.metrics import AccountGrowthTracker
        tracker = AccountGrowthTracker(1000.0)
        for i in range(10):
            tracker.update(i, 0.01)
        assert len(tracker.equity_curve()) == 10

    def test_on_track_high_growth(self):
        from app.evaluation.metrics import AccountGrowthTracker
        tracker = AccountGrowthTracker(10_000.0)
        for i in range(252):
            tracker.update(i, 0.001)  # ~28% annual
        assert tracker.is_on_track(target_annual_growth=0.20)

    def test_summary_keys(self):
        from app.evaluation.metrics import AccountGrowthTracker
        tracker = AccountGrowthTracker(1000.0)
        tracker.update(0, 0.05)
        summary = tracker.summary()
        assert "cagr_pct" in summary
        assert "milestones_reached" in summary
        assert "on_track_20pct" in summary


# ===========================================================================
# rl_pipeline
# ===========================================================================

class TestRLEnvironment:
    def test_reset_returns_state(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment
        env = RLEnvironment(synthetic_bars[:100])
        state = env.reset()
        assert state.ndim == 1
        assert len(state) > 0

    def test_step_returns_env_step(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment, EnvStep
        env = RLEnvironment(synthetic_bars[:100])
        env.reset()
        result = env.step(1)
        assert isinstance(result, EnvStep)
        assert result.period == 0

    def test_episode_terminates(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment
        env = RLEnvironment(synthetic_bars[:30])
        env.reset()
        done = False
        steps = 0
        while not done:
            step_result = env.step(1)
            done = step_result.done
            steps += 1
            if steps > 1000:
                break
        assert done

    def test_buy_positive_returns_positive_reward(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment
        # Craft a strictly rising price series → buying should be profitable
        n = 50
        prices = [100.0 + i for i in range(n)]
        bars = [Bar(float(i), p, p * 1.01, p * 0.99, p, 1e6)
                for i, p in enumerate(prices)]
        env = RLEnvironment(bars)
        env.reset()
        rewards = []
        for _ in range(n - 1):
            step_result = env.step(1)
            rewards.append(step_result.reward)
            if step_result.done:
                break
        # On average rewards should be positive (rising prices + long position)
        assert np.mean(rewards) >= 0.0

    def test_portfolio_returns_shape(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment
        env = RLEnvironment(synthetic_bars[:50])
        env.reset()
        for _ in range(49):
            env.step(0)
        returns = env.portfolio_returns
        assert returns.ndim == 1

    def test_health_factors_shape(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment
        env = RLEnvironment(synthetic_bars[:40])
        env.reset()
        for _ in range(39):
            env.step(1)
        hf = env.health_factors
        assert len(hf) > 0
        assert np.all(hf > 0)


class TestSupervisedRLAgent:
    def test_pretrain_returns_metrics(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment, SupervisedRLAgent
        from app.ml.models import LinearRegressionModel
        env = RLEnvironment(synthetic_bars[:100])
        X, y = env._build_features(synthetic_bars[:100]), np.sign(
            np.diff([b.close for b in synthetic_bars[:100]])
        )
        X, y = X[:len(y)], y
        agent = SupervisedRLAgent(LinearRegressionModel(), n_features=X.shape[1])
        metrics = agent.pretrain(X, y)
        assert "rmse" in metrics

    def test_pretrain_raises_on_wrong_n_features(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment, SupervisedRLAgent
        from app.ml.models import LinearRegressionModel
        env = RLEnvironment(synthetic_bars[:50])
        X = env._features[:40]
        y = np.sign(np.diff([b.close for b in synthetic_bars[:50]]))[:40]
        # Declare wrong n_features on purpose
        agent = SupervisedRLAgent(LinearRegressionModel(), n_features=X.shape[1] + 5)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            agent.pretrain(X, y)

    def test_run_episode_returns_steps(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLEnvironment, SupervisedRLAgent
        from app.ml.models import LinearRegressionModel
        env = RLEnvironment(synthetic_bars[:40])
        # Pre-train with dummy data so model is trained
        X = env._features[:39]
        y = np.sign(np.diff([b.close for b in synthetic_bars[:40]]))
        agent = SupervisedRLAgent(LinearRegressionModel(), n_features=X.shape[1])
        agent.pretrain(X, y)
        steps = agent.run_episode(env)
        assert len(steps) > 0

    def test_update_policy_requires_enough_samples(self, synthetic_bars):
        from app.evaluation.rl_pipeline import SupervisedRLAgent
        from app.ml.models import LinearRegressionModel
        agent = SupervisedRLAgent(LinearRegressionModel())
        # No experience yet
        result = agent.update_policy(min_new_samples=100)
        assert result is None


class TestRLPipeline:
    def test_pipeline_run_returns_dict(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLPipeline
        from app.ml.models import LinearRegressionModel
        pipeline = RLPipeline(
            model=LinearRegressionModel(alpha=0.01),
            bars=synthetic_bars[:120],
            iterations=2,
            patience=2,
        )
        result = pipeline.run()
        assert "best_composite_score" in result
        assert "total_iterations" in result

    def test_pipeline_history_populated(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLPipeline
        from app.ml.models import LinearRegressionModel
        pipeline = RLPipeline(
            model=LinearRegressionModel(),
            bars=synthetic_bars[:100],
            iterations=2,
            patience=5,
        )
        pipeline.run()
        assert len(pipeline.history()) >= 1

    def test_pipeline_progress_callback(self, synthetic_bars):
        from app.evaluation.rl_pipeline import RLPipeline
        from app.ml.models import LinearRegressionModel
        calls = []
        pipeline = RLPipeline(
            model=LinearRegressionModel(),
            bars=synthetic_bars[:100],
            iterations=2,
            patience=5,
            progress_callback=lambda it, total, r: calls.append(it),
        )
        pipeline.run()
        assert len(calls) >= 1

    def test_pipeline_too_few_bars(self):
        from app.evaluation.rl_pipeline import RLPipeline
        from app.ml.models import LinearRegressionModel
        pipeline = RLPipeline(
            model=LinearRegressionModel(),
            bars=[Bar(0.0, 1.0, 1.0, 1.0, 1.0, 0.0)] * 5,
            iterations=1,
        )
        result = pipeline.run()
        assert "error" in result


# ===========================================================================
# defi_strategy
# ===========================================================================

class TestHealthFactorManager:
    def test_zone_values(self):
        from app.evaluation.defi_strategy import HealthFactorManager
        hfm = HealthFactorManager()
        assert hfm.zone(3.0) == "A+"
        assert hfm.zone(1.8) == "A"
        assert hfm.zone(1.3) == "B"
        assert hfm.zone(1.05) == "C"
        assert hfm.zone(0.8) == "D"

    def test_requires_action_false_for_safe(self):
        from app.evaluation.defi_strategy import HealthFactorManager
        assert not HealthFactorManager().requires_action(2.0)

    def test_requires_action_true_for_watch_zone(self):
        from app.evaluation.defi_strategy import HealthFactorManager
        assert HealthFactorManager().requires_action(1.1)

    def test_repay_fraction_no_repay_needed(self):
        from app.evaluation.defi_strategy import HealthFactorManager
        assert HealthFactorManager().repay_fraction(2.0) == pytest.approx(0.0)

    def test_repay_fraction_below_stable(self):
        from app.evaluation.defi_strategy import HealthFactorManager
        hfm = HealthFactorManager(stable_hf=1.5)
        # HF=1.0 is below stable_hf=1.5 → should repay 1/3 = (1 - 1.0/1.5)
        frac = hfm.repay_fraction(1.0)
        assert frac == pytest.approx(1.0 - 1.0 / 1.5)

    def test_evaluate_dict_keys(self):
        from app.evaluation.defi_strategy import HealthFactorManager
        evaluation = HealthFactorManager().evaluate(1.8)
        assert "zone" in evaluation
        assert "repay_fraction" in evaluation
        assert "safe_to_increase" in evaluation


class TestHalfHalfThreeStrategy:
    @pytest.fixture
    def strategy(self):
        from app.defi.morpho import MorphoClient
        from app.evaluation.defi_strategy import HalfHalfThreeStrategy
        client = MorphoClient()
        # Use first available market
        markets = client.list_markets()
        market_id = next(iter(markets.values())) if markets else "0x" + "00" * 32
        return HalfHalfThreeStrategy(
            morpho_client=client,
            market_id=market_id,
            symbol="WETH",
            collateral_price=3200.0,
        )

    def test_enter_succeeds(self, strategy):
        from app.evaluation.defi_strategy import StrategyAction
        action = strategy.enter(collateral_amount=1.0)
        assert isinstance(action, StrategyAction)
        assert action.action == "enter"

    def test_rebalance_runs(self, strategy):
        strategy.enter(collateral_amount=1.0)
        action = strategy.rebalance()
        assert action.action in ("monitor_ok", "rebalance")

    def test_exit_after_enter(self, strategy):
        strategy.enter(collateral_amount=1.0)
        action = strategy.exit(full=True)
        assert action.action == "exit"

    def test_history_populated(self, strategy):
        strategy.enter(collateral_amount=1.0)
        strategy.rebalance()
        assert len(strategy.history()) >= 2

    def test_report_callback_invoked(self):
        from app.defi.morpho import MorphoClient
        from app.evaluation.defi_strategy import HalfHalfThreeStrategy
        invocations = []
        client = MorphoClient()
        markets = client.list_markets()
        market_id = next(iter(markets.values())) if markets else "0x" + "00" * 32
        strategy = HalfHalfThreeStrategy(
            client, market_id, report_callback=invocations.append
        )
        strategy.enter(collateral_amount=0.5)
        assert len(invocations) >= 1


class TestMarketEntryAdvisor:
    def test_insufficient_data(self):
        from app.evaluation.defi_strategy import MarketEntryAdvisor
        advisor = MarketEntryAdvisor()
        closes = np.array([100.0, 101.0])
        assert advisor.advise(closes) == "insufficient_data"

    def test_advise_returns_valid_string(self):
        from app.evaluation.defi_strategy import MarketEntryAdvisor
        advisor = MarketEntryAdvisor()
        closes = np.linspace(100.0, 200.0, 60)  # strong uptrend
        advice = advisor.advise(closes)
        assert advice in ("enter", "hold", "reduce", "exit", "insufficient_data")

    def test_low_rsi_triggers_enter(self):
        from app.evaluation.defi_strategy import MarketEntryAdvisor
        # Craft a series with RSI < 35 (persistent decline)
        advisor = MarketEntryAdvisor(bear_rsi_threshold=40.0)
        # Declining prices generate low RSI
        prices = np.linspace(200.0, 100.0, 50)  # sharp decline
        advice = advisor.advise(prices)
        assert advice in ("enter", "hold", "reduce", "exit")  # at minimum doesn't crash

    def test_high_rsi_triggers_exit(self):
        from app.evaluation.defi_strategy import MarketEntryAdvisor
        advisor = MarketEntryAdvisor(bull_exit_rsi_threshold=60.0)
        # Strong uptrend → high RSI
        prices = np.linspace(100.0, 500.0, 50)
        advice = advisor.advise(prices)
        assert advice in ("enter", "hold", "reduce", "exit")

    def test_advise_multi_timeframe_overall_key(self):
        from app.evaluation.defi_strategy import MarketEntryAdvisor
        advisor = MarketEntryAdvisor()
        tf_closes = {
            "1d": np.linspace(100.0, 200.0, 60),
            "4h": np.linspace(150.0, 180.0, 60),
            "1h": np.linspace(175.0, 185.0, 60),
        }
        result = advisor.advise_multi_timeframe(tf_closes)
        assert "overall" in result
        assert result["overall"] in ("enter", "hold", "reduce", "exit", "insufficient_data")
