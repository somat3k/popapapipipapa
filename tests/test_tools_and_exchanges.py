"""Tests for app/tools and app/exchanges."""

from __future__ import annotations

import pytest

from app.agents.base_agent import ToolRegistry
from app.tools.tool_definitions import register_all_tools, _safe_register


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def test_register_all_tools_returns_registry():
    reg = ToolRegistry()
    result = register_all_tools(reg)
    assert result is reg


def test_all_polygon_tools_registered():
    reg = ToolRegistry()
    register_all_tools(reg)
    names = reg.list_tools()
    for expected in (
        "polygon.block_number",
        "polygon.gas_price",
        "polygon.token_balance",
        "polygon.matic_balance",
        "polygon.tx_receipt",
        "polygon.estimate_gas",
    ):
        assert expected in names, f"{expected} not registered"


def test_all_hyperliquid_tools_registered():
    reg = ToolRegistry()
    register_all_tools(reg)
    names = reg.list_tools()
    for expected in (
        "hyperliquid.meta",
        "hyperliquid.all_mids",
        "hyperliquid.mid_price",
        "hyperliquid.l2_book",
        "hyperliquid.recent_trades",
        "hyperliquid.candles",
        "hyperliquid.user_state",
        "hyperliquid.open_orders",
        "hyperliquid.place_order",
        "hyperliquid.cancel_order",
        "hyperliquid.cancel_all",
        "hyperliquid.set_leverage",
        "hyperliquid.position_summary",
    ):
        assert expected in names, f"{expected} not registered"


def test_all_price_tools_registered():
    reg = ToolRegistry()
    register_all_tools(reg)
    names = reg.list_tools()
    for expected in ("price.get", "price.coingecko", "price.hyperliquid", "price.batch"):
        assert expected in names, f"{expected} not registered"


def test_all_morpho_tools_registered():
    reg = ToolRegistry()
    register_all_tools(reg)
    names = reg.list_tools()
    for expected in (
        "morpho.list_markets",
        "morpho.market_state",
        "morpho.position",
        "morpho.health_factor",
        "morpho.liquidation_price",
        "morpho.market_apy",
        "morpho.supply",
        "morpho.borrow",
        "morpho.repay",
        "morpho.withdraw",
        "morpho.supply_collateral",
        "morpho.simulate",
        "morpho.compare_markets",
        "morpho.growth_cycle",
        "morpho.monitor",
        "morpho.growth_grade",
    ):
        assert expected in names, f"{expected} not registered"


def test_safe_register_idempotent():
    reg = ToolRegistry()
    _safe_register(reg, "test.tool", lambda: 42)
    _safe_register(reg, "test.tool", lambda: 99)  # should not raise
    assert reg.call("test.tool") == 42  # original still registered


def test_double_register_all_tools_safe():
    reg = ToolRegistry()
    register_all_tools(reg)
    count_first = len(reg.list_tools())
    register_all_tools(reg)  # second call should be idempotent
    assert len(reg.list_tools()) == count_first


# ---------------------------------------------------------------------------
# Polygon tools (offline / mock behaviour)
# ---------------------------------------------------------------------------

def test_polygon_block_number_offline():
    from app.tools.polygon_tools import get_block_number
    result = get_block_number(rpc_url="http://localhost:9999")  # unreachable
    assert isinstance(result, dict)
    # Should return an error or -1 gracefully
    assert "block_number" in result or "error" in result


def test_polygon_gas_price_offline():
    from app.tools.polygon_tools import get_gas_price
    result = get_gas_price(rpc_url="http://localhost:9999")
    assert isinstance(result, dict)
    assert "gas_price_gwei" in result or "error" in result


def test_polygon_token_balance_offline():
    from app.tools.polygon_tools import get_token_balance
    result = get_token_balance(
        token_address="0x" + "aa" * 20,
        wallet_address="0x" + "bb" * 20,
        rpc_url="http://localhost:9999",
    )
    assert isinstance(result, dict)


def test_polygon_matic_balance_offline():
    from app.tools.polygon_tools import get_matic_balance
    result = get_matic_balance(
        wallet_address="0x" + "cc" * 20,
        rpc_url="http://localhost:9999",
    )
    assert isinstance(result, dict)


def test_polygon_estimate_gas_offline():
    from app.tools.polygon_tools import estimate_gas
    result = estimate_gas(
        from_address="0x" + "aa" * 20,
        to_address="0x" + "bb" * 20,
        rpc_url="http://localhost:9999",
    )
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Hyperliquid tools (offline — no real API)
# ---------------------------------------------------------------------------

def test_hyperliquid_get_exchange_meta_returns_dict():
    from app.tools.hyperliquid_tools import get_exchange_meta
    result = get_exchange_meta()
    # Will be either the real response or an error dict when offline
    assert isinstance(result, dict)


def test_hyperliquid_get_all_mids_returns_dict():
    from app.tools.hyperliquid_tools import get_all_mids
    result = get_all_mids()
    assert isinstance(result, dict)


def test_hyperliquid_get_mid_price_offline_graceful():
    from app.tools.hyperliquid_tools import get_mid_price
    # When offline, should return error dict, not raise
    result = get_mid_price("ETH")
    assert isinstance(result, dict)


def test_hyperliquid_get_l2_book_offline():
    from app.tools.hyperliquid_tools import get_l2_book
    result = get_l2_book("ETH")
    assert isinstance(result, dict)


def test_hyperliquid_get_user_state_offline():
    from app.tools.hyperliquid_tools import get_user_state
    result = get_user_state("0x" + "00" * 20)
    assert isinstance(result, dict)


def test_hyperliquid_get_open_orders_offline():
    from app.tools.hyperliquid_tools import get_open_orders
    result = get_open_orders("0x" + "00" * 20)
    assert isinstance(result, (dict, list))


# ---------------------------------------------------------------------------
# Price feed tools
# ---------------------------------------------------------------------------

def test_price_cache():
    from app.tools.price_feed_tools import _cache_set, _cache_get
    _cache_set("test_key", {"price": 100.0})
    val = _cache_get("test_key")
    assert val is not None
    assert val["price"] == 100.0


def test_price_coingecko_unknown_symbol():
    from app.tools.price_feed_tools import get_price_coingecko
    result = get_price_coingecko("UNKNOWN_XYZ")
    assert "error" in result


def test_price_batch_no_ids():
    from app.tools.price_feed_tools import get_prices_batch
    result = get_prices_batch(["UNKNOWN_XYZ_123"])
    assert isinstance(result, dict)


def test_price_get_graceful():
    from app.tools.price_feed_tools import get_price
    result = get_price("ETH")
    # Online: returns price dict; offline: returns error dict — both are dicts
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Hyperliquid client (app/exchanges)
# ---------------------------------------------------------------------------

def test_hyperliquid_client_init():
    from app.exchanges.hyperliquid import HyperliquidClient
    client = HyperliquidClient(wallet_address="0x" + "aa" * 20)
    assert client.wallet.startswith("0x")
    assert client._read_only is True


def test_hyperliquid_client_read_only_order():
    from app.exchanges.hyperliquid import HyperliquidClient
    client = HyperliquidClient(wallet_address="0x" + "aa" * 20)
    result = client.place_limit_order("ETH", True, 0.1, 3200.0)
    assert result.get("status") == "read_only"


def test_hyperliquid_client_read_only_market_order():
    from app.exchanges.hyperliquid import HyperliquidClient
    client = HyperliquidClient(wallet_address="0x" + "aa" * 20)
    result = client.place_market_order("BTC", True, 0.001)
    assert result.get("status") == "read_only"


def test_hyperliquid_client_read_only_cancel():
    from app.exchanges.hyperliquid import HyperliquidClient
    client = HyperliquidClient(wallet_address="0x" + "aa" * 20)
    result = client.cancel_order("ETH", 12345)
    assert result.get("status") == "read_only"


def test_hyperliquid_client_get_all_mids():
    from app.exchanges.hyperliquid import HyperliquidClient
    client = HyperliquidClient()
    mids = client.get_all_mids()
    # Either a dict of prices or empty dict when offline
    assert isinstance(mids, dict)


def test_hyperliquid_client_repr():
    from app.exchanges.hyperliquid import HyperliquidClient
    client = HyperliquidClient(wallet_address="0x" + "ab" * 20)
    r = repr(client)
    assert "HyperliquidClient" in r
