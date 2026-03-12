"""Tool definition registry — registers all platform tools with a ToolRegistry.

Usage
-----
Call :func:`register_all_tools` once at application startup to make all tools
available to agents via ``agent.use_tool("tool.name", **kwargs)``.

Tool namespace convention:
  ``polygon.<name>``      — Polygon blockchain tools
  ``hyperliquid.<name>``  — Hyperliquid exchange tools
  ``price.<name>``        — Price feed tools
  ``morpho.<name>``       — Morpho Blue DeFi tools
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agents.base_agent import ToolRegistry

logger = logging.getLogger(__name__)


def register_all_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """Register every platform tool into *registry*.

    If *registry* is None, the module-level ``TOOL_REGISTRY`` singleton is used
    and returned.

    Parameters
    ----------
    registry:
        A :class:`~app.agents.base_agent.ToolRegistry` instance.  Pass ``None``
        to register into the global shared registry.

    Returns
    -------
    The registry (for chaining / introspection).
    """
    from app.agents.base_agent import TOOL_REGISTRY

    reg = registry or TOOL_REGISTRY

    # ------------------------------------------------------------------
    # Polygon tools
    # ------------------------------------------------------------------
    from app.tools.polygon_tools import (
        estimate_gas,
        get_block_number,
        get_gas_price,
        get_matic_balance,
        get_token_balance,
        get_transaction_receipt,
    )

    _safe_register(reg, "polygon.block_number", get_block_number)
    _safe_register(reg, "polygon.gas_price", get_gas_price)
    _safe_register(reg, "polygon.token_balance", get_token_balance)
    _safe_register(reg, "polygon.matic_balance", get_matic_balance)
    _safe_register(reg, "polygon.tx_receipt", get_transaction_receipt)
    _safe_register(reg, "polygon.estimate_gas", estimate_gas)

    # ------------------------------------------------------------------
    # Hyperliquid tools
    # ------------------------------------------------------------------
    from app.tools.hyperliquid_tools import (
        cancel_all_orders,
        cancel_order,
        get_all_mids,
        get_candles,
        get_exchange_meta,
        get_funding_history,
        get_l2_book,
        get_mid_price,
        get_open_orders,
        get_order_status,
        get_position_summary,
        get_recent_trades,
        get_user_fills,
        get_user_state,
        place_order,
        set_leverage,
    )

    _safe_register(reg, "hyperliquid.meta", get_exchange_meta)
    _safe_register(reg, "hyperliquid.all_mids", get_all_mids)
    _safe_register(reg, "hyperliquid.mid_price", get_mid_price)
    _safe_register(reg, "hyperliquid.l2_book", get_l2_book)
    _safe_register(reg, "hyperliquid.recent_trades", get_recent_trades)
    _safe_register(reg, "hyperliquid.candles", get_candles)
    _safe_register(reg, "hyperliquid.user_state", get_user_state)
    _safe_register(reg, "hyperliquid.open_orders", get_open_orders)
    _safe_register(reg, "hyperliquid.order_status", get_order_status)
    _safe_register(reg, "hyperliquid.user_fills", get_user_fills)
    _safe_register(reg, "hyperliquid.funding_history", get_funding_history)
    _safe_register(reg, "hyperliquid.place_order", place_order)
    _safe_register(reg, "hyperliquid.cancel_order", cancel_order)
    _safe_register(reg, "hyperliquid.cancel_all", cancel_all_orders)
    _safe_register(reg, "hyperliquid.set_leverage", set_leverage)
    _safe_register(reg, "hyperliquid.position_summary", get_position_summary)

    # ------------------------------------------------------------------
    # Price feed tools
    # ------------------------------------------------------------------
    from app.tools.price_feed_tools import (
        get_price,
        get_price_coingecko,
        get_price_hyperliquid,
        get_prices_batch,
    )

    _safe_register(reg, "price.get", get_price)
    _safe_register(reg, "price.coingecko", get_price_coingecko)
    _safe_register(reg, "price.hyperliquid", get_price_hyperliquid)
    _safe_register(reg, "price.batch", get_prices_batch)

    # ------------------------------------------------------------------
    # Morpho tools
    # ------------------------------------------------------------------
    _register_morpho_tools(reg)

    logger.info("Registered %d tools in ToolRegistry.", len(reg.list_tools()))
    return reg


def _register_morpho_tools(reg: ToolRegistry) -> None:
    """Register Morpho Blue tools backed by a lazily-created mock client."""
    try:
        from morpho.client import MorphoBlueClient
        from morpho.growth import GrowthEngine
        from morpho.simulation import PositionSimulator

        def _client() -> MorphoBlueClient:
            """Return (or create) the shared mock client."""
            if not hasattr(_register_morpho_tools, "_shared_client"):
                _register_morpho_tools._shared_client = MorphoBlueClient()  # type: ignore[attr-defined]
            return _register_morpho_tools._shared_client  # type: ignore[attr-defined]

        def _engine() -> GrowthEngine:
            if not hasattr(_register_morpho_tools, "_shared_engine"):
                _register_morpho_tools._shared_engine = GrowthEngine(_client())  # type: ignore[attr-defined]
            return _register_morpho_tools._shared_engine  # type: ignore[attr-defined]

        def _sim() -> PositionSimulator:
            if not hasattr(_register_morpho_tools, "_shared_sim"):
                _register_morpho_tools._shared_sim = PositionSimulator(_client())  # type: ignore[attr-defined]
            return _register_morpho_tools._shared_sim  # type: ignore[attr-defined]

        _safe_register(reg, "morpho.list_markets", lambda: _client().list_markets())
        _safe_register(reg, "morpho.market_state",
                       lambda market_name: _client().get_market_state(market_name))
        _safe_register(reg, "morpho.position",
                       lambda market_name: _client().get_position(market_name))
        _safe_register(reg, "morpho.health_factor",
                       lambda market_name: _client().health_factor(market_name))
        _safe_register(reg, "morpho.liquidation_price",
                       lambda market_name: _client().liquidation_price(market_name))
        _safe_register(reg, "morpho.market_apy",
                       lambda market_name: _client().market_apy(market_name))
        _safe_register(reg, "morpho.supply",
                       lambda market_name, assets, dry_run=True: _client().supply(market_name, assets, dry_run=dry_run))
        _safe_register(reg, "morpho.borrow",
                       lambda market_name, assets, dry_run=True: _client().borrow(market_name, assets, dry_run=dry_run))
        _safe_register(reg, "morpho.repay",
                       lambda market_name, assets=None, dry_run=True: _client().repay(market_name, assets, dry_run=dry_run))
        _safe_register(reg, "morpho.withdraw",
                       lambda market_name, assets, dry_run=True: _client().withdraw(market_name, assets, dry_run=dry_run))
        _safe_register(reg, "morpho.supply_collateral",
                       lambda market_name, assets, dry_run=True: _client().supply_collateral(market_name, assets, dry_run=dry_run))
        _safe_register(reg, "morpho.simulate",
                       lambda market_name, horizon_days=30.0: _sim().project(market_name, horizon_days))
        _safe_register(reg, "morpho.compare_markets",
                       lambda horizon_days=30.0: _sim().compare_markets(horizon_days))
        _safe_register(reg, "morpho.growth_cycle",
                       lambda market_name, collateral_assets, dry_run=True:
                       _engine().run_growth_cycle(market_name, collateral_assets, dry_run=dry_run))
        _safe_register(reg, "morpho.monitor",
                       lambda market_name: _engine().monitor_and_rebalance(market_name))
        _safe_register(reg, "morpho.growth_grade",
                       lambda: _engine().growth_grade)

    except Exception:
        logger.exception("Failed to register Morpho tools — continuing without them.")


def _safe_register(reg: ToolRegistry, name: str, fn: object) -> None:
    """Register *fn* under *name*, skipping if already registered."""
    try:
        reg.register(name, fn)  # type: ignore[arg-type]
    except ValueError:
        pass  # already registered (e.g. called twice)
