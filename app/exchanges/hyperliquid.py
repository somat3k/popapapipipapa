"""Hyperliquid exchange client module.

Provides a high-level HyperliquidClient wrapping the Hyperliquid REST API
for use by TradingAgent and other specialist agents.

References
----------
- https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
- https://github.com/hyperliquid-dex/hyperliquid-python-sdk
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class HLOrder:
    """A Hyperliquid order."""

    coin: str
    is_buy: bool
    size: float
    price: float
    order_type: str = "limit"  # "limit" | "market"
    reduce_only: bool = False
    order_id: Optional[int] = None
    status: str = "pending"


@dataclass
class HLPosition:
    """An open Hyperliquid position."""

    coin: str
    side: str  # "long" | "short"
    size: float
    entry_price: float
    unrealized_pnl: float = 0.0
    leverage: int = 1


@dataclass
class HLMarketInfo:
    """Metadata about a Hyperliquid perpetual market."""

    name: str
    sz_decimals: int
    max_leverage: int = 50


class HyperliquidClient:
    """High-level Hyperliquid exchange client.

    Parameters
    ----------
    wallet_address:
        Hyperliquid wallet address (EVM-compatible).
    private_key:
        Private key for signing exchange transactions.  Leave empty for
        read-only mode.

    Examples
    --------
    >>> client = HyperliquidClient(wallet_address="0xYourAddress")
    >>> price = client.get_mid_price("ETH")
    >>> print(price)
    """

    def __init__(
        self,
        wallet_address: str = "",
        private_key: str = "",
    ) -> None:
        self.wallet = wallet_address
        self._private_key = private_key
        self._read_only = not private_key
        logger.info(
            "HyperliquidClient ready  wallet=%s  read_only=%s",
            wallet_address[:10] + "…" if wallet_address else "(none)",
            self._read_only,
        )

    # ------------------------------------------------------------------
    # Market data (no auth required)
    # ------------------------------------------------------------------

    def get_meta(self) -> dict[str, Any]:
        """Return exchange metadata (tradeable universe)."""
        from app.tools.hyperliquid_tools import get_exchange_meta
        return get_exchange_meta()

    def get_mid_price(self, coin: str) -> float:
        """Return the current mid price for *coin*.  Returns -1.0 on error."""
        from app.tools.hyperliquid_tools import get_mid_price
        result = get_mid_price(coin)
        if "error" in result:
            logger.warning("[HL] get_mid_price(%s): %s", coin, result["error"])
            return -1.0
        return float(result["mid"])

    def get_all_mids(self) -> dict[str, float]:
        """Return a dict mapping coin → mid price."""
        from app.tools.hyperliquid_tools import get_all_mids
        raw = get_all_mids()
        if "error" in raw:
            return {}
        return {coin: float(price) for coin, price in raw.items() if isinstance(price, str)}

    def get_order_book(self, coin: str, depth: int = 5) -> dict[str, Any]:
        """Return top *depth* levels of the L2 order book."""
        from app.tools.hyperliquid_tools import get_l2_book
        book = get_l2_book(coin)
        if "error" in book:
            return book
        levels = book.get("levels", [[], []])
        bids = [{"price": float(l["px"]), "size": float(l["sz"])} for l in levels[0][:depth]]
        asks = [{"price": float(l["px"]), "size": float(l["sz"])} for l in levels[1][:depth]]
        return {"coin": coin, "bids": bids, "asks": asks}

    def get_recent_trades(self, coin: str) -> list[dict]:
        """Return recent trades for *coin*."""
        from app.tools.hyperliquid_tools import get_recent_trades
        result = get_recent_trades(coin)
        return result if isinstance(result, list) else []

    def get_candles(
        self,
        coin: str,
        interval: str = "1h",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list[dict]:
        """Return OHLCV candles for *coin*."""
        from app.tools.hyperliquid_tools import get_candles
        result = get_candles(coin, interval, start_time, end_time)
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Account state (no auth required)
    # ------------------------------------------------------------------

    def get_positions(self) -> list[HLPosition]:
        """Return open positions for the wallet."""
        if not self.wallet:
            return []
        from app.tools.hyperliquid_tools import get_user_state
        state = get_user_state(self.wallet)
        positions = []
        for ap in state.get("assetPositions", []):
            p = ap.get("position", {})
            if float(p.get("szi", 0)) == 0:
                continue
            size = float(p.get("szi", 0))
            positions.append(
                HLPosition(
                    coin=p.get("coin", ""),
                    side="long" if size > 0 else "short",
                    size=abs(size),
                    entry_price=float(p.get("entryPx", 0)),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                    leverage=int(p.get("leverage", {}).get("value", 1)),
                )
            )
        return positions

    def get_open_orders(self) -> list[dict]:
        """Return open orders for the wallet."""
        if not self.wallet:
            return []
        from app.tools.hyperliquid_tools import get_open_orders
        result = get_open_orders(self.wallet)
        return result if isinstance(result, list) else []

    def get_account_value(self) -> float:
        """Return total account value in USD."""
        if not self.wallet:
            return 0.0
        from app.tools.hyperliquid_tools import get_user_state
        state = get_user_state(self.wallet)
        return float(state.get("marginSummary", {}).get("accountValue", 0))

    # ------------------------------------------------------------------
    # Order management (requires private key)
    # ------------------------------------------------------------------

    def place_limit_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        price: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place a limit order.

        Returns the raw API response dict.  In read-only mode, returns a
        simulated response.
        """
        if self._read_only:
            logger.warning("[HL] Read-only mode — order NOT placed.")
            return {"status": "read_only", "simulated": True, "coin": coin, "size": size, "price": price}
        from app.tools.hyperliquid_tools import place_order
        return place_order(
            self._private_key, self.wallet, coin, is_buy, size, price,
            order_type="limit", reduce_only=reduce_only,
        )

    def place_market_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Place a market order (IOC tif)."""
        if self._read_only:
            logger.warning("[HL] Read-only mode — order NOT placed.")
            return {"status": "read_only", "simulated": True, "coin": coin, "size": size}
        mid = self.get_mid_price(coin)
        slippage = 0.001  # 0.1% slippage for market order price
        price = mid * (1 + slippage) if is_buy else mid * (1 - slippage)
        from app.tools.hyperliquid_tools import place_order
        return place_order(
            self._private_key, self.wallet, coin, is_buy, size, price,
            order_type="market", reduce_only=reduce_only,
        )

    def cancel_order(self, coin: str, oid: int) -> dict[str, Any]:
        """Cancel an open order."""
        if self._read_only:
            return {"status": "read_only"}
        from app.tools.hyperliquid_tools import cancel_order
        return cancel_order(self._private_key, self.wallet, coin, oid)

    def cancel_all(self) -> dict[str, Any]:
        """Cancel all open orders."""
        if self._read_only:
            return {"status": "read_only"}
        from app.tools.hyperliquid_tools import cancel_all_orders
        return cancel_all_orders(self._private_key, self.wallet)

    def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> dict[str, Any]:
        """Set leverage for a coin."""
        if self._read_only:
            return {"status": "read_only"}
        from app.tools.hyperliquid_tools import set_leverage
        return set_leverage(self._private_key, self.wallet, coin, leverage, is_cross)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Return True if the Hyperliquid API is reachable."""
        meta = self.get_meta()
        return "error" not in meta

    def __repr__(self) -> str:
        return (
            f"<HyperliquidClient wallet={self.wallet[:8]}… "
            f"read_only={self._read_only}>"
        )
