"""Specialist agents: TradingAgent, DeFiAgent, MLAgent, AnalysisAgent,
ChatAgent, RiskAgent, OrchestratorAgent.

All agents have tool-use capabilities via the shared ToolRegistry.
Available tool namespaces:
  polygon.*        — Polygon blockchain tools
  hyperliquid.*    — Hyperliquid exchange tools
  price.*          — Multi-source price feed tools
  morpho.*         — Morpho Blue DeFi tools
"""

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
    """Executes trading algorithms and manages order lifecycle.

    Uses the following tools when available:
      - ``hyperliquid.mid_price``     — fetch real-time price for a symbol
      - ``hyperliquid.position_summary`` — check open positions
      - ``hyperliquid.place_order``   — submit orders (requires private key)
      - ``price.get``                 — multi-source price lookup
      - ``polygon.gas_price``         — current Polygon gas price
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="TradingAgent", **kwargs)
        self._signals: List[Dict[str, Any]] = []
        self.symbols = symbols or ["ETH", "BTC", "MATIC"]

    def _execute(self) -> None:
        logger.info("[TradingAgent] Starting trading loop.")
        while not self.should_stop():
            try:
                self._run_cycle()
            except Exception:
                logger.exception("[TradingAgent] Cycle error.")
            time.sleep(5)

    def _run_cycle(self) -> None:
        prices = self._fetch_prices()
        payload: Dict[str, Any] = {
            "agent": self.name,
            "ts": time.time(),
            "prices": prices,
        }
        if prices:
            self.context.set("latest_prices", prices)
        self.bus.publish("trading.cycle", payload)
        logger.debug("[TradingAgent] Cycle complete. prices=%s", prices)

    def _fetch_prices(self) -> Dict[str, float]:
        """Fetch mid prices from Hyperliquid for all tracked symbols."""
        prices: Dict[str, float] = {}
        for sym in self.symbols:
            try:
                result = self.use_tool("hyperliquid.mid_price", coin=sym)
                if isinstance(result, dict) and "mid" in result:
                    prices[sym] = float(result["mid"])
                elif isinstance(result, dict) and "price" in result:
                    prices[sym] = float(result["price"])
            except Exception:
                logger.debug("[TradingAgent] Price fetch failed for %s", sym)
        return prices

    def get_position_summary(self, wallet_address: str) -> Dict[str, Any]:
        """Fetch open positions from Hyperliquid for *wallet_address*."""
        try:
            return self.use_tool("hyperliquid.position_summary", wallet_address=wallet_address)
        except Exception as exc:
            return {"error": str(exc)}

    def submit_signal(self, signal: Dict[str, Any]) -> None:
        self._signals.append(signal)
        self.bus.publish("trading.signal", signal)

    def get_signals(self) -> List[Dict[str, Any]]:
        return list(self._signals)


# ---------------------------------------------------------------------------
# DeFi Agent
# ---------------------------------------------------------------------------

class DeFiAgent(BaseAgent):
    """Manages DeFi operations via Morpho Blue on Polygon.

    Uses the following tools when available:
      - ``morpho.position``        — query on-chain position
      - ``morpho.health_factor``   — monitor health factor
      - ``morpho.market_apy``      — fetch current APYs
      - ``morpho.supply``          — supply tokens
      - ``morpho.borrow``          — borrow against collateral
      - ``morpho.repay``           — repay debt
      - ``morpho.monitor``         — run the growth engine monitor step
      - ``polygon.gas_price``      — check Polygon gas price before transacting
    """

    def __init__(
        self,
        monitored_markets: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="DeFiAgent", **kwargs)
        self._positions: Dict[str, Any] = {}
        self.monitored_markets = monitored_markets or ["WETH/USDC_E-86", "WBTC/USDC_E-86"]

    def _execute(self) -> None:
        logger.info("[DeFiAgent] Starting DeFi monitoring loop.")
        while not self.should_stop():
            try:
                self._monitor_positions()
            except Exception:
                logger.exception("[DeFiAgent] Monitor error.")
            time.sleep(10)

    def _monitor_positions(self) -> None:
        at_risk: List[str] = []
        for market in self.monitored_markets:
            hf = self._fetch_health_factor(market)
            if hf is not None and hf < 1.20:
                at_risk.append(market)
                logger.warning("[DeFiAgent] Market %s at risk: HF=%.3f", market, hf)
                self.bus.publish("defi.risk.alert", {"market": market, "health_factor": hf})
        self.bus.publish("defi.heartbeat", {
            "agent": self.name,
            "ts": time.time(),
            "at_risk_markets": at_risk,
        })
        logger.debug("[DeFiAgent] Position check complete.")

    def _fetch_health_factor(self, market: str) -> Optional[float]:
        """Fetch health factor via tool; returns None if unavailable."""
        try:
            result = self.use_tool("morpho.health_factor", market_name=market)
            if isinstance(result, (int, float)):
                return float(result)
        except Exception:
            pass
        return None

    def get_market_apy(self, market: str) -> Dict[str, float]:
        """Return current supply/borrow APY for *market*."""
        try:
            result = self.use_tool("morpho.market_apy", market_name=market)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return {}

    def supply(self, market: str, assets: int, *, dry_run: bool = True) -> Dict[str, Any]:
        """Supply *assets* (base units) to *market*.  Defaults to dry_run=True."""
        try:
            result = self.use_tool("morpho.supply", market_name=market, assets=assets, dry_run=dry_run)
            self.bus.publish("defi.supply", {"market": market, "assets": assets, "result": result})
            return result if isinstance(result, dict) else {"success": bool(result)}
        except Exception as exc:
            return {"error": str(exc)}

    def borrow(self, market: str, assets: int, *, dry_run: bool = True) -> Dict[str, Any]:
        """Borrow *assets* (base units) from *market*.  Defaults to dry_run=True."""
        try:
            result = self.use_tool("morpho.borrow", market_name=market, assets=assets, dry_run=dry_run)
            self.bus.publish("defi.borrow", {"market": market, "assets": assets, "result": result})
            return result if isinstance(result, dict) else {"success": bool(result)}
        except Exception as exc:
            return {"error": str(exc)}

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
    """Market data analysis and signal generation.

    Uses the following tools when available:
      - ``price.batch``              — fetch prices for all tracked symbols
      - ``hyperliquid.candles``      — fetch OHLCV candles
      - ``morpho.compare_markets``   — compare Morpho market yields
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="AnalysisAgent", **kwargs)
        self._latest_analysis: Dict[str, Any] = {}
        self.symbols = symbols or ["ETH", "BTC", "MATIC"]

    def _execute(self) -> None:
        logger.info("[AnalysisAgent] Analysis loop started.")
        while not self.should_stop():
            try:
                self._analyse()
            except Exception:
                logger.exception("[AnalysisAgent] Analysis error.")
            time.sleep(15)

    def _analyse(self) -> None:
        prices = self._fetch_prices()
        signal = self._compute_signal(prices)
        result: Dict[str, Any] = {
            "ts": time.time(),
            "signal": signal,
            "prices": prices,
        }
        self._latest_analysis = result
        self.context.set("latest_signal", {"signal": signal, "confidence": 0.5})
        self.bus.publish("analysis.result", result)

    def _fetch_prices(self) -> Dict[str, float]:
        """Fetch prices using the price.batch tool."""
        try:
            result = self.use_tool("price.batch", symbols=self.symbols)
            if isinstance(result, dict) and "prices" in result:
                return result["prices"]
        except Exception:
            pass
        return {}

    @staticmethod
    def _compute_signal(prices: Dict[str, float]) -> str:
        """Trivial signal: bullish if ETH price available, else neutral."""
        if prices.get("ETH", 0) > 0:
            return "bullish"
        return "neutral"

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
        if msg_lower.startswith("/polygon"):
            return self._handle_polygon_command(message)
        return (
            f"Received: '{message}'. Use /help for available commands. "
            "I can assist with DeFi operations, trading signals, and ML model management."
        )

    def _help_text(self) -> str:
        return (
            "Available commands:\n"
            "  /help               — Show this help text\n"
            "  /defi status        — Show DeFi positions\n"
            "  /defi supply        — Supply assets to Morpho\n"
            "  /defi borrow        — Borrow against collateral\n"
            "  /defi repay         — Repay outstanding debt\n"
            "  /defi swap          — Execute collateral swap\n"
            "  /defi apy <market>  — Show supply/borrow APY for a market\n"
            "  /defi markets       — List available Morpho markets\n"
            "  /trade signal       — Get latest trading signal\n"
            "  /trade buy          — Place buy order\n"
            "  /trade sell         — Place sell order\n"
            "  /trade price <coin> — Get Hyperliquid mid price\n"
            "  /trade positions    — Show Hyperliquid open positions\n"
            "  /ml train           — Trigger model training\n"
            "  /ml metrics         — Show model metrics\n"
            "  /polygon gas        — Current Polygon gas price\n"
            "  /polygon block      — Current Polygon block number\n"
            "  /status             — System status overview\n"
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
        if sub == "apy":
            market = parts[2] if len(parts) > 2 else "WETH/USDC_E-86"
            try:
                apy_data = self.tools.call("morpho.market_apy", market_name=market)
                if isinstance(apy_data, dict) and "supply_apy" in apy_data:
                    return (
                        f"Market {market} APY:\n"
                        f"  Supply APY: {apy_data['supply_apy'] * 100:.2f}%\n"
                        f"  Borrow APY: {apy_data['borrow_apy'] * 100:.2f}%\n"
                        f"  Utilisation: {apy_data.get('utilisation', 0) * 100:.1f}%"
                    )
            except Exception:
                pass
            return f"Could not fetch APY for {market}."
        if sub == "markets":
            try:
                markets = self.tools.call("morpho.list_markets")
                if markets:
                    lines = [f"  • {m.name} ({m.lltv_pct:.1f}% LLTV)" for m in markets]
                    return "Available Morpho markets:\n" + "\n".join(lines)
            except Exception:
                pass
            return "Could not list markets."
        return f"Unknown DeFi sub-command: {sub}"

    def _handle_trade_command(self, message: str) -> str:
        parts = message.strip().split()
        sub = parts[1].lower() if len(parts) > 1 else "signal"
        if sub == "signal":
            sig = self.context.get("latest_signal", {"signal": "neutral", "confidence": 0.0})
            return f"Latest signal: {sig['signal']} (confidence: {sig.get('confidence', 0):.2f})"
        if sub in ("buy", "sell"):
            return f"Use the Trading panel to place a {sub} order."
        if sub == "price":
            coin = parts[2].upper() if len(parts) > 2 else "ETH"
            try:
                result = self.tools.call("hyperliquid.mid_price", coin=coin)
                if isinstance(result, dict) and "mid" in result:
                    return f"{coin} mid price: ${result['mid']:,.2f} (Hyperliquid)"
                if isinstance(result, dict) and "error" in result:
                    # fallback to CoinGecko
                    cg = self.tools.call("price.get", symbol=coin)
                    if isinstance(cg, dict) and "price" in cg:
                        return f"{coin} price: ${cg['price']:,.2f} ({cg.get('source', 'price feed')})"
            except Exception:
                pass
            return f"Could not fetch price for {coin}."
        if sub == "positions":
            wallet = self.context.get("wallet_address", "")
            if not wallet:
                return "No wallet address configured. Set context key 'wallet_address'."
            try:
                summary = self.tools.call("hyperliquid.position_summary", wallet_address=wallet)
                if isinstance(summary, dict) and "positions" in summary:
                    pos_list = summary["positions"]
                    if not pos_list:
                        return "No open positions on Hyperliquid."
                    lines = [
                        f"  {p.get('position', {}).get('coin', '?')}: "
                        f"size={p.get('position', {}).get('szi', 0)} "
                        f"pnl={p.get('position', {}).get('unrealizedPnl', 0)}"
                        for p in pos_list
                    ]
                    return "Open positions:\n" + "\n".join(lines)
            except Exception:
                pass
            return "Could not fetch positions."
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
            "  Platform: Multiplex Financials v0.2.0\n"
            "  Network: Polygon (Mainnet) + Hyperliquid\n"
            "  Agents: Active\n"
            "  Trading: Online (Hyperliquid perps)\n"
            "  DeFi: Connected (Morpho Blue on Polygon)\n"
            "  ML: Ready\n"
            "  Tools: polygon.*, hyperliquid.*, price.*, morpho.*\n"
        )

    def _handle_polygon_command(self, message: str) -> str:
        parts = message.strip().split()
        sub = parts[1].lower() if len(parts) > 1 else "gas"
        if sub == "gas":
            try:
                result = self.tools.call("polygon.gas_price")
                if isinstance(result, dict) and "gas_price_gwei" in result:
                    return f"Polygon gas price: {result['gas_price_gwei']:.2f} Gwei"
            except Exception:
                pass
            return "Could not fetch Polygon gas price."
        if sub == "block":
            try:
                result = self.tools.call("polygon.block_number")
                if isinstance(result, dict) and "block_number" in result:
                    return f"Polygon block number: {result['block_number']:,}"
            except Exception:
                pass
            return "Could not fetch Polygon block number."
        return f"Unknown polygon sub-command: {sub}. Try /polygon gas or /polygon block."

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
