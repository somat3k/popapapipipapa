"""Multi-source price feed tools for agent use.

Provides token price lookups from:
  1. CoinGecko public API (no API key required)
  2. Hyperliquid allMids (for perp-listed assets)
  3. Local cache with configurable TTL

All functions return structured dicts and are safe to call repeatedly.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Thread-safe TTL cache
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()
CACHE_TTL = 30.0  # seconds

COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"

# CoinGecko coin IDs for common tokens
COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "WBTC": "wrapped-bitcoin",
    "ETH": "ethereum",
    "WETH": "weth",
    "MATIC": "matic-network",
    "POL": "matic-network",
    "WPOL": "matic-network",
    "USDC": "usd-coin",
    "USDT": "tether",
    "DAI": "dai",
    "stMATIC": "lido-staked-matic",
    "MaticX": "stader-maticx",
    "SOL": "solana",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "ARB": "arbitrum",
    "OP": "optimism",
}


def _cache_get(key: str) -> Optional[Any]:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
    if entry and (time.time() - entry[0]) < CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: Any) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


def _http_get(url: str, params: Optional[dict] = None) -> dict[str, Any]:
    try:
        import requests  # type: ignore

        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        return {"error": "requests not installed"}
    except Exception as exc:
        return {"error": str(exc)}


def get_price_coingecko(symbol: str, vs_currency: str = "usd") -> dict[str, Any]:
    """Fetch current price from CoinGecko.

    Parameters
    ----------
    symbol:      Token symbol, e.g. ``"ETH"``, ``"WBTC"``.
    vs_currency: Quote currency (default ``"usd"``).

    Returns
    -------
    {"symbol": str, "price": float, "source": "coingecko"}
    """
    cache_key = f"cg:{symbol}:{vs_currency}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    coin_id = COINGECKO_IDS.get(symbol.upper())
    if not coin_id:
        return {"error": f"No CoinGecko ID for symbol '{symbol}'"}

    data = _http_get(COINGECKO_PRICE_URL, {"ids": coin_id, "vs_currencies": vs_currency})
    if "error" in data:
        return {**data, "symbol": symbol}

    price = data.get(coin_id, {}).get(vs_currency)
    if price is None:
        return {"error": f"Price not found for {coin_id}", "symbol": symbol}

    result = {"symbol": symbol, "price": float(price), "source": "coingecko"}
    _cache_set(cache_key, result)
    return result


def get_price_hyperliquid(symbol: str) -> dict[str, Any]:
    """Fetch current mid price from Hyperliquid.

    Parameters
    ----------
    symbol:  Coin symbol as listed on Hyperliquid, e.g. ``"ETH"``, ``"BTC"``.

    Returns
    -------
    {"symbol": str, "price": float, "source": "hyperliquid"}
    """
    cache_key = f"hl:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    from .hyperliquid_tools import get_mid_price

    result = get_mid_price(symbol)
    if "error" in result:
        return {**result, "symbol": symbol}

    out = {"symbol": symbol, "price": result["mid"], "source": "hyperliquid"}
    _cache_set(cache_key, out)
    return out


def get_price(symbol: str, prefer: str = "coingecko") -> dict[str, Any]:
    """Fetch price from the preferred source, with fallback.

    Parameters
    ----------
    symbol:  Token symbol.
    prefer:  ``"coingecko"`` or ``"hyperliquid"``.

    Returns
    -------
    {"symbol": str, "price": float, "source": str}
    """
    if prefer == "hyperliquid":
        result = get_price_hyperliquid(symbol)
        if "error" not in result:
            return result
        return get_price_coingecko(symbol)

    result = get_price_coingecko(symbol)
    if "error" not in result:
        return result
    return get_price_hyperliquid(symbol)


def get_prices_batch(symbols: list[str], vs_currency: str = "usd") -> dict[str, Any]:
    """Fetch prices for multiple symbols in one CoinGecko call.

    Returns
    -------
    {"prices": {"ETH": 3200.0, "BTC": 65000.0, ...}, "source": "coingecko"}
    """
    coin_ids = {s: COINGECKO_IDS.get(s.upper()) for s in symbols}
    missing = [s for s, cid in coin_ids.items() if cid is None]
    valid_ids = [cid for cid in coin_ids.values() if cid]
    id_to_symbol = {cid: sym for sym, cid in coin_ids.items() if cid}

    if not valid_ids:
        return {"error": f"No CoinGecko IDs for {symbols}", "prices": {}}

    data = _http_get(
        COINGECKO_PRICE_URL,
        {"ids": ",".join(valid_ids), "vs_currencies": vs_currency},
    )
    if "error" in data:
        return {**data, "prices": {}}

    prices = {}
    for coin_id, price_info in data.items():
        symbol = id_to_symbol.get(coin_id)
        if symbol:
            prices[symbol] = float(price_info.get(vs_currency, 0))

    return {"prices": prices, "missing": missing, "source": "coingecko"}
