"""Agent tool definitions package.

Provides tool implementations for Polygon blockchain and Hyperliquid exchange
that can be registered in the ToolRegistry and used by any BaseAgent subclass.

Sub-modules
-----------
polygon_tools     — Web3/Polygon blockchain tools
hyperliquid_tools — Hyperliquid exchange tools (info + exchange API)
price_feed_tools  — Multi-source price feed tools
tool_definitions  — Convenience function to register all tools at once
"""

from .tool_definitions import register_all_tools

__all__ = ["register_all_tools"]
