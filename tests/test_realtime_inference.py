"""Tests for realtime inference runner and agent payload exchange."""

from __future__ import annotations

import numpy as np
import pytest

from app.agents.agents import MLAgent, TradingAgent
from app.agents.base_agent import AgentContext, MessageBus
from app.evaluation.data_loader import OHLCVLoader
from app.evaluation.realtime_inference import (
    INFERENCE_TOPIC,
    InferencePayload,
    InferenceSummary,
    RealtimeInferenceRunner,
)
from app.evaluation.test_set_scoring import run_test_set_scoring
from app.ml.models import LinearRegressionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trained_model(n_features: int = 8) -> LinearRegressionModel:
    """Return a freshly trained LinearRegressionModel."""
    model = LinearRegressionModel(alpha=0.1)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((80, n_features))
    y = np.sign(rng.standard_normal(80))
    model.fit(X, y)
    return model


def _bars(n: int = 60, symbol: str = "ETH") -> list:
    loader = OHLCVLoader()
    return loader.generate_synthetic(n=n, seed=0, symbol=symbol)


# ---------------------------------------------------------------------------
# RealtimeInferenceRunner — basic operation
# ---------------------------------------------------------------------------

def test_runner_returns_summary():
    model = _trained_model()
    bars = _bars(60)
    bus = MessageBus()
    runner = RealtimeInferenceRunner(model=model, bars=bars, symbol="ETH", message_bus=bus)
    summary = runner.run()

    assert isinstance(summary, InferenceSummary)
    assert summary.total_bars == len(bars)
    assert summary.buy_signals + summary.hold_signals + summary.sell_signals == summary.total_bars
    assert 0.0 <= summary.mean_confidence <= 1.0
    assert summary.elapsed_s >= 0.0
    assert len(summary.payloads) == summary.total_bars


def test_runner_payload_fields():
    model = _trained_model()
    bars = _bars(10)
    bus = MessageBus()
    runner = RealtimeInferenceRunner(model=model, bars=bars, symbol="BTC", message_bus=bus)
    summary = runner.run()

    for i, p in enumerate(summary.payloads):
        assert isinstance(p, InferencePayload)
        assert p.bar_index == i
        assert p.symbol == "BTC"
        assert p.action in (-1, 0, 1)
        assert 0.0 <= p.confidence <= 1.0
        assert p.close > 0
        assert p.timestamp == bars[i].timestamp


def test_runner_publishes_to_message_bus():
    model = _trained_model()
    bars = _bars(20)
    bus = MessageBus()
    received = []
    bus.subscribe(INFERENCE_TOPIC, received.append)

    runner = RealtimeInferenceRunner(model=model, bars=bars, symbol="ETH", message_bus=bus)
    summary = runner.run()

    assert len(received) == summary.total_bars
    for msg in received:
        assert isinstance(msg, dict)
        assert "bar_index" in msg
        assert "action" in msg
        assert "prediction" in msg
        assert "confidence" in msg


def test_runner_on_payload_callback():
    model = _trained_model()
    bars = _bars(15)
    bus = MessageBus()
    collected = []

    runner = RealtimeInferenceRunner(
        model=model, bars=bars, symbol="MATIC", message_bus=bus, on_payload=collected.append
    )
    summary = runner.run()

    assert len(collected) == summary.total_bars
    assert all(isinstance(p, InferencePayload) for p in collected)


def test_runner_raises_if_model_not_trained():
    model = LinearRegressionModel()
    bars = _bars(20)
    bus = MessageBus()
    runner = RealtimeInferenceRunner(model=model, bars=bars, message_bus=bus)
    with pytest.raises(RuntimeError, match="must be trained"):
        runner.run()


def test_runner_empty_bars():
    model = _trained_model()
    bus = MessageBus()
    runner = RealtimeInferenceRunner(model=model, bars=[], symbol="ETH", message_bus=bus)
    summary = runner.run()

    assert summary.total_bars == 0
    assert summary.buy_signals == 0
    assert summary.sell_signals == 0
    assert summary.hold_signals == 0
    assert summary.payloads == []


def test_runner_to_dict():
    model = _trained_model()
    bars = _bars(20)
    bus = MessageBus()
    runner = RealtimeInferenceRunner(model=model, bars=bars, symbol="ETH", message_bus=bus)
    summary = runner.run()
    d = summary.to_dict()

    assert d["total_bars"] == summary.total_bars
    assert d["buy_signals"] == summary.buy_signals
    assert d["mean_confidence"] == summary.mean_confidence
    # payloads are not included in to_dict()
    assert "payloads" not in d


def test_inference_payload_to_dict():
    p = InferencePayload(
        bar_index=3,
        symbol="ETH",
        timestamp=1_000_000.0,
        close=2500.0,
        prediction=0.42,
        action=1,
        confidence=0.42,
    )
    d = p.to_dict()
    assert d["bar_index"] == 3
    assert d["action"] == 1
    assert "features" not in d


# ---------------------------------------------------------------------------
# MLAgent — run_inference_stream
# ---------------------------------------------------------------------------

def test_ml_agent_run_inference_stream():
    bus = MessageBus()
    ctx = AgentContext()
    agent = MLAgent(context=ctx, message_bus=bus)

    model = _trained_model()
    agent.set_model(model)

    bars = _bars(30)
    result = agent.run_inference_stream(bars, symbol="ETH")

    assert isinstance(result, dict)
    assert result["total_bars"] == len(bars)
    assert result["buy_signals"] + result["hold_signals"] + result["sell_signals"] == result["total_bars"]
    assert "mean_confidence" in result
    assert "elapsed_s" in result


def test_ml_agent_inference_stream_publishes_summary():
    bus = MessageBus()
    summaries = []
    bus.subscribe("ml.inference.summary", summaries.append)

    agent = MLAgent(message_bus=bus)
    model = _trained_model()
    agent.set_model(model)

    bars = _bars(25)
    agent.run_inference_stream(bars, symbol="BTC")

    assert len(summaries) == 1
    assert summaries[0]["agent"] == "MLAgent"
    assert summaries[0]["symbol"] == "BTC"
    assert summaries[0]["total_bars"] == len(bars)


def test_ml_agent_inference_stream_no_model_raises():
    bus = MessageBus()
    agent = MLAgent(message_bus=bus)
    with pytest.raises(RuntimeError, match="No active model"):
        agent.run_inference_stream(_bars(10))


def test_ml_agent_inference_stream_model_override():
    bus = MessageBus()
    agent = MLAgent(message_bus=bus)  # no active model set

    model = _trained_model()
    bars = _bars(20)
    # Pass model directly — should work without set_model()
    result = agent.run_inference_stream(bars, symbol="ETH", model=model)
    assert result["total_bars"] == len(bars)


# ---------------------------------------------------------------------------
# TradingAgent — inference payload subscription
# ---------------------------------------------------------------------------

def test_trading_agent_receives_inference_payloads():
    bus = MessageBus()
    trading = TradingAgent(message_bus=bus)

    # Publish a mock inference payload directly to the bus
    payload = {
        "bar_index": 0,
        "symbol": "ETH",
        "timestamp": 1_000_000.0,
        "close": 2500.0,
        "prediction": 0.5,
        "action": 1,
        "confidence": 0.5,
    }
    bus.publish(INFERENCE_TOPIC, payload)

    received = trading.get_inference_payloads()
    assert len(received) == 1
    assert received[0]["action"] == 1
    assert received[0]["symbol"] == "ETH"


def test_trading_agent_generates_signal_from_inference():
    bus = MessageBus()
    trading = TradingAgent(message_bus=bus)

    bus.publish(INFERENCE_TOPIC, {
        "bar_index": 0, "symbol": "ETH", "timestamp": 0.0,
        "close": 2000.0, "prediction": 0.7, "action": 1, "confidence": 0.7,
    })

    signals = trading.get_signals()
    assert any(s["source"] == "ml.inference" for s in signals)
    ml_signal = next(s for s in signals if s["source"] == "ml.inference")
    assert ml_signal["direction"] == 1
    assert ml_signal["symbol"] == "ETH"


def test_trading_agent_no_signal_for_hold_action():
    bus = MessageBus()
    trading = TradingAgent(message_bus=bus)

    bus.publish(INFERENCE_TOPIC, {
        "bar_index": 0, "symbol": "ETH", "timestamp": 0.0,
        "close": 2000.0, "prediction": 0.0, "action": 0, "confidence": 0.0,
    })

    # hold action (0) should NOT generate a signal
    signals = [s for s in trading.get_signals() if s.get("source") == "ml.inference"]
    assert len(signals) == 0


def test_trading_agent_sell_signal_from_inference():
    bus = MessageBus()
    trading = TradingAgent(message_bus=bus)

    bus.publish(INFERENCE_TOPIC, {
        "bar_index": 5, "symbol": "BTC", "timestamp": 0.0,
        "close": 40000.0, "prediction": -0.6, "action": -1, "confidence": 0.6,
    })

    signals = [s for s in trading.get_signals() if s.get("source") == "ml.inference"]
    assert len(signals) == 1
    assert signals[0]["direction"] == -1


def test_trading_agent_ignores_non_dict_payload():
    bus = MessageBus()
    trading = TradingAgent(message_bus=bus)

    # Publish a non-dict message on the inference topic
    bus.publish(INFERENCE_TOPIC, "not-a-dict")
    bus.publish(INFERENCE_TOPIC, 42)

    # Neither should be stored
    assert trading.get_inference_payloads() == []
    assert trading.get_signals() == []


def test_trading_agent_stop_unsubscribes():
    bus = MessageBus()
    trading = TradingAgent(message_bus=bus)

    # After stopping, new payloads on the topic should NOT be received
    trading.stop()

    bus.publish(INFERENCE_TOPIC, {
        "bar_index": 0, "symbol": "ETH", "timestamp": 0.0,
        "close": 2000.0, "prediction": 0.9, "action": 1, "confidence": 0.9,
    })

    assert trading.get_inference_payloads() == []


# ---------------------------------------------------------------------------
# Full agent payload exchange: MLAgent → TradingAgent via MessageBus
# ---------------------------------------------------------------------------

def test_ml_to_trading_agent_payload_exchange():
    """End-to-end: MLAgent streams inference → TradingAgent receives payloads."""
    bus = MessageBus()
    ml_agent = MLAgent(message_bus=bus)
    trading_agent = TradingAgent(message_bus=bus)

    model = _trained_model()
    ml_agent.set_model(model)

    bars = _bars(40, symbol="ETH")
    ml_agent.run_inference_stream(bars, symbol="ETH")

    payloads = trading_agent.get_inference_payloads()
    assert len(payloads) == len(bars)

    # Every payload should have the expected keys
    for p in payloads:
        assert "action" in p
        assert "confidence" in p
        assert p["symbol"] == "ETH"

    # At least some signals should have been generated (non-zero actions)
    ml_signals = [s for s in trading_agent.get_signals() if s.get("source") == "ml.inference"]
    non_hold = sum(1 for p in payloads if p["action"] != 0)
    assert len(ml_signals) == non_hold


# ---------------------------------------------------------------------------
# run_test_set_scoring — realtime_inference field
# ---------------------------------------------------------------------------

def test_run_test_set_scoring_includes_realtime_inference():
    result = run_test_set_scoring(
        model=LinearRegressionModel(alpha=0.1),
        symbol="ETH",
        data_source="synthetic",
        bars_count=80,
        seed=0,
        iterations=1,
        patience=1,
    )

    assert "realtime_inference" in result
    ri = result["realtime_inference"]
    assert isinstance(ri, dict)
    # Should not be an error or skipped dict
    assert "error" not in ri
    assert ri["total_bars"] > 0
    assert ri["buy_signals"] + ri["hold_signals"] + ri["sell_signals"] == ri["total_bars"]


def test_run_test_set_scoring_realtime_inference_with_morpho_client():
    class DummyMorphoClient:
        def list_markets(self):
            return {"TEST": "0x" + "00" * 32}

        def get_position(self, market_id: str):
            return {"market_id": market_id}

    result = run_test_set_scoring(
        model=LinearRegressionModel(alpha=0.1),
        symbol="BTC",
        data_source="synthetic",
        bars_count=80,
        seed=1,
        morpho_client=DummyMorphoClient(),
        iterations=1,
        patience=1,
    )

    assert result["connection"]["status"] in {"connected", "no_markets"}
    ri = result["realtime_inference"]
    assert ri["total_bars"] > 0
    assert "mean_confidence" in ri
