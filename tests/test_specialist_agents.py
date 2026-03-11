"""Tests for the specialist agents."""

import time

import pytest

from app.agents.agents import (
    AnalysisAgent,
    ChatAgent,
    DeFiAgent,
    MLAgent,
    OrchestratorAgent,
    RiskAgent,
    TradingAgent,
)
from app.agents.base_agent import AgentContext, AgentState


# ---------------------------------------------------------------------------
# TradingAgent
# ---------------------------------------------------------------------------

def test_trading_agent_runs():
    agent = TradingAgent()
    agent.run(background=True)
    time.sleep(0.05)
    assert agent.state == AgentState.RUNNING
    agent.stop()


def test_trading_agent_submit_signal():
    agent = TradingAgent()
    agent.submit_signal({"symbol": "MATIC/USDC", "direction": 1})
    sigs = agent.get_signals()
    assert len(sigs) == 1
    assert sigs[0]["direction"] == 1


def test_trading_agent_get_signals_empty():
    agent = TradingAgent()
    assert agent.get_signals() == []


# ---------------------------------------------------------------------------
# DeFiAgent
# ---------------------------------------------------------------------------

def test_defi_agent_runs():
    agent = DeFiAgent()
    agent.run(background=True)
    time.sleep(0.05)
    assert agent.state == AgentState.RUNNING
    agent.stop()


def test_defi_agent_update_and_get_positions():
    agent = DeFiAgent()
    agent.update_position("WMATIC/USDC", {"collateral": 500.0, "borrow": 100.0})
    positions = agent.get_positions()
    assert "WMATIC/USDC" in positions
    assert positions["WMATIC/USDC"]["collateral"] == 500.0


def test_defi_agent_position_published():
    received = []
    from app.agents.base_agent import MESSAGE_BUS
    MESSAGE_BUS.subscribe("defi.position.update", received.append)
    agent = DeFiAgent()
    agent.update_position("market1", {"collateral": 10.0})
    assert len(received) >= 1
    MESSAGE_BUS.unsubscribe("defi.position.update", received.append)


# ---------------------------------------------------------------------------
# MLAgent
# ---------------------------------------------------------------------------

def test_ml_agent_update_metrics():
    agent = MLAgent()
    agent.update_metrics({"rmse": 0.05, "mae": 0.03})
    m = agent.get_metrics()
    assert m["rmse"] == pytest.approx(0.05)
    assert m["mae"] == pytest.approx(0.03)


def test_ml_agent_predict_no_model():
    agent = MLAgent()
    with pytest.raises(RuntimeError, match="No active model"):
        agent.predict([[1, 2, 3]])


def test_ml_agent_predict_with_model():
    import numpy as np
    from app.ml.models import LinearRegressionModel

    agent = MLAgent()
    model = LinearRegressionModel()
    X = np.random.default_rng(0).standard_normal((50, 3)).astype(np.float64)
    y = X[:, 0] + X[:, 1]
    model.fit(X, y)
    agent.set_model(model)
    preds = agent.predict(X[:5])
    assert len(preds) == 5


# ---------------------------------------------------------------------------
# AnalysisAgent
# ---------------------------------------------------------------------------

def test_analysis_agent_runs():
    agent = AnalysisAgent()
    agent.run(background=True)
    time.sleep(0.05)
    assert agent.state == AgentState.RUNNING
    agent.stop()


def test_analysis_agent_get_latest():
    agent = AnalysisAgent()
    latest = agent.get_latest()
    assert isinstance(latest, dict)


# ---------------------------------------------------------------------------
# ChatAgent
# ---------------------------------------------------------------------------

def test_chat_agent_help():
    agent = ChatAgent()
    resp = agent.process_message("/help")
    assert "/defi" in resp
    assert "/trade" in resp


def test_chat_agent_status():
    agent = ChatAgent()
    resp = agent.process_message("/status")
    assert "Multiplex" in resp


def test_chat_agent_defi_status_no_positions():
    agent = ChatAgent()
    resp = agent.process_message("/defi status")
    assert "No active" in resp or "DeFi" in resp


def test_chat_agent_defi_status_with_positions():
    ctx = AgentContext()
    ctx.set("defi_positions", {"WMATIC/USDC": {"collateral": 500.0}})
    agent = ChatAgent(context=ctx)
    resp = agent.process_message("/defi status")
    assert "WMATIC/USDC" in resp


def test_chat_agent_trade_signal():
    agent = ChatAgent()
    resp = agent.process_message("/trade signal")
    assert "signal" in resp.lower()


def test_chat_agent_ml_metrics_empty():
    agent = ChatAgent()
    resp = agent.process_message("/ml metrics")
    assert "No ML metrics" in resp or "metrics" in resp.lower()


def test_chat_agent_ml_metrics_with_data():
    ctx = AgentContext()
    ctx.set("ml_metrics", {"rmse": 0.05, "mae": 0.03})
    agent = ChatAgent(context=ctx)
    resp = agent.process_message("/ml metrics")
    assert "rmse" in resp.lower()


def test_chat_agent_history():
    agent = ChatAgent()
    agent.process_message("hello")
    history = agent.get_history()
    assert len(history) == 2  # user + assistant
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_chat_agent_clear_history():
    agent = ChatAgent()
    agent.process_message("test")
    agent.clear_history()
    assert agent.get_history() == []


def test_chat_agent_unknown_command():
    agent = ChatAgent()
    resp = agent.process_message("/unknown")
    assert "Received" in resp or len(resp) > 0


def test_chat_agent_response_callback():
    received = []
    agent = ChatAgent(response_callback=received.append)
    agent.process_message("/help")
    assert len(received) == 1


# ---------------------------------------------------------------------------
# RiskAgent
# ---------------------------------------------------------------------------

def test_risk_agent_no_breach():
    ctx = AgentContext()
    ctx.set("current_drawdown", 0.05)
    agent = RiskAgent(max_drawdown=0.15, context=ctx)
    agent._evaluate_risk()
    assert not agent.breaker_tripped


def test_risk_agent_circuit_breaker_trips():
    ctx = AgentContext()
    ctx.set("current_drawdown", 0.20)
    agent = RiskAgent(max_drawdown=0.15, context=ctx)
    agent._evaluate_risk()
    assert agent.breaker_tripped


def test_risk_agent_reset_breaker():
    ctx = AgentContext()
    ctx.set("current_drawdown", 0.20)
    agent = RiskAgent(max_drawdown=0.15, context=ctx)
    agent._evaluate_risk()
    assert agent.breaker_tripped
    agent.reset_breaker()
    assert not agent.breaker_tripped


def test_risk_agent_breaker_trips_once():
    ctx = AgentContext()
    ctx.set("current_drawdown", 0.20)
    breaker_events = []
    from app.agents.base_agent import MESSAGE_BUS
    MESSAGE_BUS.subscribe("risk.circuit_breaker", breaker_events.append)
    agent = RiskAgent(max_drawdown=0.15, context=ctx)
    agent._evaluate_risk()
    agent._evaluate_risk()  # second call — already tripped
    assert len(breaker_events) == 1
    MESSAGE_BUS.unsubscribe("risk.circuit_breaker", breaker_events.append)


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

def test_orchestrator_gets_status():
    orch = OrchestratorAgent()
    sub = TradingAgent()
    orch.add_agent(sub)
    status = orch.get_status()
    assert "TradingAgent" in status


def test_orchestrator_starts_sub_agents():
    orch = OrchestratorAgent()
    trading = TradingAgent()
    defi = DeFiAgent()
    orch.add_agent(trading)
    orch.add_agent(defi)
    orch.run(background=True)
    time.sleep(0.2)
    assert trading.state == AgentState.RUNNING
    assert defi.state == AgentState.RUNNING
    orch.stop()
