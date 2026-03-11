"""
Multiplex Financials — DEFI AI Platform
Version 0.2.0

Environments
------------
  Blockchain: Polygon PoS (chain ID 137) via Web3.py
  DeFi:       Morpho Blue (https://morpho.org) — supply, borrow, repay, collateral swap
  Exchange:   Hyperliquid (https://hyperliquid.xyz) — perpetuals, spot

Tool namespaces registered at startup:
  polygon.*       — Polygon blockchain tools
  hyperliquid.*   — Hyperliquid exchange tools
  price.*         — Multi-source price feed (CoinGecko + Hyperliquid)
  morpho.*        — Morpho Blue DeFi tools
"""

import logging

VERSION = "0.2.0"
PLATFORM_NAME = "Multiplex Financials"

logger = logging.getLogger(__name__)


def _bootstrap_tools() -> None:
    """Register all platform tools into the global ToolRegistry on import."""
    try:
        from app.tools.tool_definitions import register_all_tools
        register_all_tools()
    except Exception:
        logger.exception("Tool registration failed — agents will run without tools.")


_bootstrap_tools()
