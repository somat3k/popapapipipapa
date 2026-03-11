"""Specialist agents: TradingAgent, DeFiAgent, MLAgent, AnalysisAgent,
ChatAgent, RiskAgent, OrchestratorAgent."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .base_agent import AgentContext, AgentState, BaseAgent, MessageBus, ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trading Agent
# ---------------------------------------------------------------------------

class TradingAgent(BaseAgent):
    """Executes trading algorithms and manages order lifecycle."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="TradingAgent", **kwargs)
        self._signals: List[Dict[str, Any]] = []

    def _execute(self) -> None:
        logger.info("[TradingAgent] Starting trading loop.")
        while not self.should_stop():
            try:
                self._run_cycle()
            except Exception:
                logger.exception("[TradingAgent] Cycle error.")
            time.sleep(5)

    def _run_cycle(self) -> None:
        self.bus.publish("trading.cycle", {"agent": self.name, "ts": time.time()})
        logger.debug("[TradingAgent] Cycle complete.")

    def submit_signal(self, signal: Dict[str, Any]) -> None:
        self._signals.append(signal)
        self.bus.publish("trading.signal", signal)

    def get_signals(self) -> List[Dict[str, Any]]:
        return list(self._signals)


# ---------------------------------------------------------------------------
# DeFi Agent
# ---------------------------------------------------------------------------

class DeFiAgent(BaseAgent):
    """Manages DeFi operations via Morpho on Polygon."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="DeFiAgent", **kwargs)
        self._positions: Dict[str, Any] = {}

    def _execute(self) -> None:
        logger.info("[DeFiAgent] Starting DeFi monitoring loop.")
        while not self.should_stop():
            try:
                self._monitor_positions()
            except Exception:
                logger.exception("[DeFiAgent] Monitor error.")
            time.sleep(10)

    def _monitor_positions(self) -> None:
        self.bus.publish("defi.heartbeat", {"agent": self.name, "ts": time.time()})
        logger.debug("[DeFiAgent] Position check complete.")

    def update_position(self, market: str, data: Dict[str, Any]) -> None:
        self._positions[market] = data
        self.bus.publish("defi.position.update", {"market": market, **data})

    def get_positions(self) -> Dict[str, Any]:
        return dict(self._positions)


# ---------------------------------------------------------------------------
# ML Agent
# ---------------------------------------------------------------------------

class MLAgent(BaseAgent):
    """Trains, evaluates, and serves ML models."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="MLAgent", **kwargs)
        self._active_model: Optional[Any] = None
        self._metrics: Dict[str, float] = {}

    def _execute(self) -> None:
        logger.info("[MLAgent] ML Agent standing by.")
        while not self.should_stop():
            time.sleep(30)

    def set_model(self, model: Any) -> None:
        self._active_model = model
        self.bus.publish("ml.model.updated", {"agent": self.name})

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        self._metrics.update(metrics)
        self.bus.publish("ml.metrics", metrics)

    def get_metrics(self) -> Dict[str, float]:
        return dict(self._metrics)

    def predict(self, features: Any) -> Any:
        if self._active_model is None:
            raise RuntimeError("No active model loaded.")
        return self._active_model.predict(features)


# ---------------------------------------------------------------------------
# Analysis Agent
# ---------------------------------------------------------------------------

class AnalysisAgent(BaseAgent):
    """Market data analysis and signal generation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="AnalysisAgent", **kwargs)
        self._latest_analysis: Dict[str, Any] = {}

    def _execute(self) -> None:
        logger.info("[AnalysisAgent] Analysis loop started.")
        while not self.should_stop():
            try:
                self._analyse()
            except Exception:
                logger.exception("[AnalysisAgent] Analysis error.")
            time.sleep(15)

    def _analyse(self) -> None:
        result = {"ts": time.time(), "signal": "neutral"}
        self._latest_analysis = result
        self.bus.publish("analysis.result", result)

    def get_latest(self) -> Dict[str, Any]:
        return dict(self._latest_analysis)


# ---------------------------------------------------------------------------
# Chat Agent
# ---------------------------------------------------------------------------

class ChatAgent(BaseAgent):
    """Agentic chat interface with tool dispatch."""

    def __init__(
        self,
        response_callback: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="ChatAgent", **kwargs)
        self._history: List[Dict[str, str]] = []
        self._response_callback = response_callback

    def _execute(self) -> None:
        logger.info("[ChatAgent] Chat agent ready.")
        while not self.should_stop():
            time.sleep(1)

    def process_message(self, user_message: str) -> str:
        """Process a user message and return a response."""
        self._history.append({"role": "user", "content": user_message})
        response = self._generate_response(user_message)
        self._history.append({"role": "assistant", "content": response})
        if self._response_callback:
            self._response_callback(response)
        self.bus.publish("chat.response", {"response": response})
        return response

    def _generate_response(self, message: str) -> str:
        msg_lower = message.lower()
        if msg_lower.startswith("/help"):
            return self._help_text()
        if msg_lower.startswith("/defi"):
            return self._handle_defi_command(message)
        if msg_lower.startswith("/trade"):
            return self._handle_trade_command(message)
        if msg_lower.startswith("/ml"):
            return self._handle_ml_command(message)
        if msg_lower.startswith("/status"):
            return self._handle_status_command()
        return (
            f"Received: '{message}'. Use /help for available commands. "
            "I can assist with DeFi operations, trading signals, and ML model management."
        )

    def _help_text(self) -> str:
        return (
            "Available commands:\n"
            "  /help          — Show this help text\n"
            "  /defi status   — Show DeFi positions\n"
            "  /defi supply   — Supply assets to Morpho\n"
            "  /defi borrow   — Borrow against collateral\n"
            "  /defi repay    — Repay outstanding debt\n"
            "  /defi swap     — Execute collateral swap\n"
            "  /trade signal  — Get latest trading signal\n"
            "  /trade buy     — Place buy order\n"
            "  /trade sell    — Place sell order\n"
            "  /ml train      — Trigger model training\n"
            "  /ml metrics    — Show model metrics\n"
            "  /status        — System status overview\n"
        )

    def _handle_defi_command(self, message: str) -> str:
        parts = message.strip().split()
        sub = parts[1].lower() if len(parts) > 1 else "status"
        defi_pos = self.context.get("defi_positions", {})
        if sub == "status":
            if defi_pos:
                lines = [f"  {k}: {v}" for k, v in defi_pos.items()]
                return "DeFi Positions:\n" + "\n".join(lines)
            return "No active DeFi positions found."
        if sub == "supply":
            return "Supply: use the DeFi panel to supply assets to Morpho."
        if sub == "borrow":
            return "Borrow: use the DeFi panel to borrow against your collateral."
        if sub == "repay":
            return "Repay: use the DeFi panel to repay outstanding debt."
        if sub == "swap":
            return "Collateral Swap: use the DeFi panel wizard to initiate a swap."
        return f"Unknown DeFi sub-command: {sub}"

    def _handle_trade_command(self, message: str) -> str:
        parts = message.strip().split()
        sub = parts[1].lower() if len(parts) > 1 else "signal"
        if sub == "signal":
            sig = self.context.get("latest_signal", {"signal": "neutral", "confidence": 0.0})
            return f"Latest signal: {sig['signal']} (confidence: {sig.get('confidence', 0):.2f})"
        if sub in ("buy", "sell"):
            return f"Use the Trading panel to place a {sub} order."
        return f"Unknown trade sub-command: {sub}"

    def _handle_ml_command(self, message: str) -> str:
        parts = message.strip().split()
        sub = parts[1].lower() if len(parts) > 1 else "metrics"
        if sub == "train":
            return "Training: navigate to the ML panel to configure and start training."
        if sub == "metrics":
            metrics = self.context.get("ml_metrics", {})
            if metrics:
                lines = [f"  {k}: {v:.4f}" for k, v in metrics.items()]
                return "ML Metrics:\n" + "\n".join(lines)
            return "No ML metrics available yet."
        return f"Unknown ML sub-command: {sub}"

    def _handle_status_command(self) -> str:
        return (
            "System Status:\n"
            "  Platform: Multiplex Financials v0.1.0\n"
            "  Network: Polygon (Mainnet)\n"
            "  Agents: Active\n"
            "  Trading: Online\n"
            "  DeFi: Connected\n"
            "  ML: Ready\n"
        )

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------------------------

class RiskAgent(BaseAgent):
    """Portfolio risk scoring and circuit-breaker management."""

    def __init__(self, max_drawdown: float = 0.15, **kwargs: Any) -> None:
        super().__init__(name="RiskAgent", **kwargs)
        self.max_drawdown = max_drawdown
        self._breaker_tripped = False
        self._risk_score: float = 0.0

    def _execute(self) -> None:
        logger.info("[RiskAgent] Risk monitoring started.")
        while not self.should_stop():
            try:
                self._evaluate_risk()
            except Exception:
                logger.exception("[RiskAgent] Risk eval error.")
            time.sleep(5)

    def _evaluate_risk(self) -> None:
        drawdown = self.context.get("current_drawdown", 0.0)
        if drawdown >= self.max_drawdown and not self._breaker_tripped:
            self._trip_breaker(drawdown)
        self.bus.publish("risk.score", {"score": self._risk_score, "ts": time.time()})

    def _trip_breaker(self, drawdown: float) -> None:
        self._breaker_tripped = True
        logger.warning("[RiskAgent] Circuit breaker TRIPPED — drawdown=%.2f%%", drawdown * 100)
        self.bus.publish("risk.circuit_breaker", {"tripped": True, "drawdown": drawdown})

    def reset_breaker(self) -> None:
        self._breaker_tripped = False
        logger.info("[RiskAgent] Circuit breaker reset.")

    @property
    def breaker_tripped(self) -> bool:
        return self._breaker_tripped


# ---------------------------------------------------------------------------
# Orchestrator Agent
# ---------------------------------------------------------------------------

class OrchestratorAgent(BaseAgent):
    """Master coordinator routing tasks to specialist agents."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="OrchestratorAgent", **kwargs)
        self._sub_agents: List[BaseAgent] = []

    def add_agent(self, agent: BaseAgent) -> None:
        self._sub_agents.append(agent)

    def _execute(self) -> None:
        logger.info("[OrchestratorAgent] Starting all sub-agents.")
        for agent in self._sub_agents:
            agent.run(background=True)
        while not self.should_stop():
            self._health_check()
            time.sleep(10)
        logger.info("[OrchestratorAgent] Stopping all sub-agents.")
        for agent in self._sub_agents:
            agent.stop()

    def _health_check(self) -> None:
        for agent in self._sub_agents:
            if agent.state == AgentState.ERROR:
                logger.warning("[OrchestratorAgent] Agent %s in ERROR — resetting.", agent.name)
                agent.reset()
                agent.run(background=True)

    def get_status(self) -> Dict[str, str]:
        return {a.name: a.state.value for a in self._sub_agents}
