"""Hyperliquid exchange tools for agent use.

Implements the Hyperliquid REST API (Info endpoint) and the Exchange endpoint
for order management.  Uses ``requests`` (sync) with a thin async wrapper when
``aiohttp`` is available.

Info API (read-only, no auth):  https://api.hyperliquid.xyz/info
Exchange API (signed trades):   https://api.hyperliquid.xyz/exchange

Reference: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api

All tools return structured dicts and never raise — errors are encoded in
``{"error": "..."}`` fields to keep agents fault-tolerant.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"
REQUEST_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict) -> dict[str, Any]:
    """POST JSON payload to *url* and return the parsed response."""
    try:
        import requests  # type: ignore

        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        return {"error": "requests library not installed"}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Info tools (no authentication required)
# ---------------------------------------------------------------------------

def get_exchange_meta() -> dict[str, Any]:
    """Fetch exchange metadata (universe of tradeable assets).

    Returns a dict with ``universe`` list of asset objects::

        {"universe": [{"name": "ETH", "szDecimals": 4, ...}, ...]}
    """
    return _post(HL_INFO_URL, {"type": "meta"})


def get_all_mids() -> dict[str, Any]:
    """Fetch mid prices for all assets.

    Returns
    -------
    {"BTC": "65000.5", "ETH": "3200.1", ...}
    """
    return _post(HL_INFO_URL, {"type": "allMids"})


def get_mid_price(coin: str) -> dict[str, Any]:
    """Fetch the current mid price for a single *coin* (e.g. ``"ETH"``).

    Returns
    -------
    {"coin": str, "mid": float} or {"error": str}
    """
    mids = get_all_mids()
    if "error" in mids:
        return mids
    price_str = mids.get(coin)
    if price_str is None:
        return {"error": f"Coin '{coin}' not found in Hyperliquid universe"}
    return {"coin": coin, "mid": float(price_str)}


def get_l2_book(coin: str, n_sig_figs: int = 5) -> dict[str, Any]:
    """Fetch the L2 order book for *coin*.

    Returns
    -------
    {"coin": str, "levels": [[{"px": str, "sz": str, "n": int}, ...], [...]]}
    where levels[0] = bids, levels[1] = asks.
    """
    return _post(HL_INFO_URL, {"type": "l2Book", "coin": coin, "nSigFigs": n_sig_figs})


def get_recent_trades(coin: str) -> dict[str, Any]:
    """Fetch recent trades for *coin*.

    Returns
    -------
    List of trade objects ``[{"coin": str, "side": str, "px": str, "sz": str, "time": int}, ...]``
    """
    return _post(HL_INFO_URL, {"type": "recentTrades", "coin": coin})


def get_candles(coin: str, interval: str = "1h", start_time: Optional[int] = None, end_time: Optional[int] = None) -> dict[str, Any]:
    """Fetch OHLCV candles for *coin*.

    Parameters
    ----------
    coin:       Asset symbol, e.g. ``"ETH"``.
    interval:   Candle interval — ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``, ``"4h"``, ``"1d"``.
    start_time: Unix timestamp in ms (optional).
    end_time:   Unix timestamp in ms (optional).
    """
    payload: dict[str, Any] = {"type": "candleSnapshot", "req": {"coin": coin, "interval": interval}}
    if start_time:
        payload["req"]["startTime"] = start_time
    if end_time:
        payload["req"]["endTime"] = end_time
    return _post(HL_INFO_URL, payload)


def get_user_state(wallet_address: str) -> dict[str, Any]:
    """Fetch user positions, balances, and open orders.

    Returns a dict with ``marginSummary``, ``crossMaintenanceMarginUsed``,
    ``assetPositions``, etc.
    """
    return _post(HL_INFO_URL, {"type": "clearinghouseState", "user": wallet_address})


def get_open_orders(wallet_address: str) -> dict[str, Any]:
    """Fetch open orders for a user.

    Returns a list of order objects.
    """
    return _post(HL_INFO_URL, {"type": "openOrders", "user": wallet_address})


def get_order_status(wallet_address: str, oid: int) -> dict[str, Any]:
    """Fetch status of a specific order by its order ID."""
    return _post(HL_INFO_URL, {"type": "orderStatus", "user": wallet_address, "oid": oid})


def get_user_fills(wallet_address: str) -> dict[str, Any]:
    """Fetch historical fills (executed trades) for a user."""
    return _post(HL_INFO_URL, {"type": "userFills", "user": wallet_address})


def get_funding_history(coin: str, start_time: int, end_time: Optional[int] = None) -> dict[str, Any]:
    """Fetch historical funding rate data for *coin*."""
    payload: dict[str, Any] = {"type": "fundingHistory", "coin": coin, "startTime": start_time}
    if end_time:
        payload["endTime"] = end_time
    return _post(HL_INFO_URL, payload)


# ---------------------------------------------------------------------------
# Exchange tools (authenticated — require private key)
# ---------------------------------------------------------------------------

def _sign_l1_action(private_key: str, action: dict, vault_address: Optional[str], nonce: int) -> dict[str, Any]:
    """Sign a Hyperliquid L1 action using the agent's private key.

    .. warning::
        This is a **structural placeholder** using a simple message hash.
        Hyperliquid requires EIP-712 structured-data signing with a specific
        domain separator.  For production use, integrate the official
        ``hyperliquid-python-sdk`` (``pip install hyperliquid-python-sdk``),
        which implements the correct signing scheme.  Signatures produced by
        this function **will be rejected** by the live exchange.

    See also: https://github.com/hyperliquid-dex/hyperliquid-python-sdk
    """
    try:
        from eth_account import Account  # type: ignore
        from eth_account.messages import encode_defunct  # type: ignore
    except ImportError:
        return {"error": "eth_account not installed — pip install eth-account"}

    # Simplified signing: hash action JSON + nonce (not full EIP-712, see SDK for prod)
    msg = json.dumps({"action": action, "nonce": nonce}, sort_keys=True)
    message = encode_defunct(text=msg)
    signed = Account.sign_message(message, private_key=private_key)
    return {
        "r": hex(signed.r),
        "s": hex(signed.s),
        "v": signed.v,
    }


def place_order(
    private_key: str,
    wallet_address: str,
    coin: str,
    is_buy: bool,
    size: float,
    price: float,
    order_type: str = "limit",
    reduce_only: bool = False,
    vault_address: Optional[str] = None,
) -> dict[str, Any]:
    """Place an order on Hyperliquid.

    Parameters
    ----------
    private_key:    Hex private key for signing.
    wallet_address: Your Hyperliquid wallet address.
    coin:           Asset symbol, e.g. ``"ETH"``.
    is_buy:         True for buy/long, False for sell/short.
    size:           Order size in base units.
    price:          Limit price (ignored for market orders).
    order_type:     ``"limit"`` or ``"market"``.
    reduce_only:    If True, only reduce existing position.
    vault_address:  Optional vault address for sub-accounts.

    Returns
    -------
    {"status": "ok", "response": {...}} or {"error": str}
    """
    nonce = int(time.time() * 1000)

    if order_type == "market":
        order_type_obj = {"limit": {"tif": "Ioc"}}
    else:
        order_type_obj = {"limit": {"tif": "Gtc"}}

    action = {
        "type": "order",
        "orders": [
            {
                "a": coin,
                "b": is_buy,
                "p": str(price),
                "s": str(size),
                "r": reduce_only,
                "t": order_type_obj,
            }
        ],
        "grouping": "na",
    }

    signature = _sign_l1_action(private_key, action, vault_address, nonce)
    if "error" in signature:
        return signature

    payload = {
        "action": action,
        "nonce": nonce,
        "signature": signature,
    }
    if vault_address:
        payload["vaultAddress"] = vault_address

    return _post(HL_EXCHANGE_URL, payload)


def cancel_order(
    private_key: str,
    wallet_address: str,
    coin: str,
    oid: int,
    vault_address: Optional[str] = None,
) -> dict[str, Any]:
    """Cancel an open order by its order ID.

    Returns
    -------
    {"status": "ok"} or {"error": str}
    """
    nonce = int(time.time() * 1000)
    action = {
        "type": "cancel",
        "cancels": [{"a": coin, "o": oid}],
    }
    signature = _sign_l1_action(private_key, action, vault_address, nonce)
    if "error" in signature:
        return signature

    payload = {"action": action, "nonce": nonce, "signature": signature}
    if vault_address:
        payload["vaultAddress"] = vault_address
    return _post(HL_EXCHANGE_URL, payload)


def cancel_all_orders(
    private_key: str,
    wallet_address: str,
) -> dict[str, Any]:
    """Cancel all open orders for the wallet.

    Fetches open orders first, then cancels each one.
    """
    open_orders = get_open_orders(wallet_address)
    if "error" in open_orders:
        return open_orders
    if not isinstance(open_orders, list):
        return {"cancelled": 0, "errors": []}

    results = []
    for order in open_orders:
        coin = order.get("coin", "")
        oid = order.get("oid", 0)
        res = cancel_order(private_key, wallet_address, coin, oid)
        results.append({"oid": oid, "coin": coin, "result": res})

    cancelled = sum(1 for r in results if r["result"].get("status") == "ok")
    return {"cancelled": cancelled, "total": len(results), "results": results}


def set_leverage(
    private_key: str,
    wallet_address: str,
    coin: str,
    leverage: int,
    is_cross: bool = True,
) -> dict[str, Any]:
    """Set leverage for a coin position.

    Parameters
    ----------
    leverage:  Integer leverage level (1–50).
    is_cross:  True for cross margin, False for isolated.
    """
    nonce = int(time.time() * 1000)
    action = {
        "type": "updateLeverage",
        "asset": coin,
        "isCross": is_cross,
        "leverage": leverage,
    }
    signature = _sign_l1_action(private_key, action, None, nonce)
    if "error" in signature:
        return signature
    return _post(HL_EXCHANGE_URL, {"action": action, "nonce": nonce, "signature": signature})


def get_position_summary(wallet_address: str, coin: Optional[str] = None) -> dict[str, Any]:
    """High-level position summary for the wallet, optionally filtered by coin.

    Returns
    -------
    {"positions": [...], "total_unrealized_pnl": float, "account_value": float}
    """
    state = get_user_state(wallet_address)
    if "error" in state:
        return state

    positions = state.get("assetPositions", [])
    if coin:
        positions = [p for p in positions if p.get("position", {}).get("coin") == coin]

    total_pnl = sum(
        float(p.get("position", {}).get("unrealizedPnl", 0)) for p in positions
    )
    account_value = float(state.get("marginSummary", {}).get("accountValue", 0))

    return {
        "positions": positions,
        "total_unrealized_pnl": round(total_pnl, 4),
        "account_value": round(account_value, 4),
    }
