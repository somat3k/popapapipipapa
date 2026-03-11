"""Tests for the morpho package: contracts, markets, client, growth, simulation."""

from __future__ import annotations

import pytest

from morpho.contracts import (
    MORPHO_BLUE_POLYGON,
    POLYGON_CHAIN_ID,
    POLYGON_RPC_URLS,
    TOKEN_ADDRESSES,
    TOKEN_DECIMALS,
    IRM_ADDRESS,
)
from morpho.markets import (
    MarketConfig,
    MarketRegistry,
    build_market_id,
    WAD,
)
from morpho.client import (
    MorphoBlueClient,
    MarketState,
    TxResult,
    UserPosition,
)
from morpho.growth import GrowthEngine, GrowthReport
from morpho.simulation import PositionSimulator


# ---------------------------------------------------------------------------
# contracts.py
# ---------------------------------------------------------------------------

def test_morpho_blue_address_format():
    assert MORPHO_BLUE_POLYGON.startswith("0x")
    assert len(MORPHO_BLUE_POLYGON) == 42


def test_polygon_chain_id():
    assert POLYGON_CHAIN_ID == 137


def test_polygon_rpc_urls_non_empty():
    assert len(POLYGON_RPC_URLS) > 0
    for url in POLYGON_RPC_URLS:
        assert url.startswith("https://")


def test_token_addresses_present():
    for symbol in ("USDC_E", "USDC", "WETH", "WBTC", "WPOL"):
        assert symbol in TOKEN_ADDRESSES
        assert TOKEN_ADDRESSES[symbol].startswith("0x")


def test_token_decimals_present():
    for symbol in ("USDC_E", "USDC", "WETH", "WBTC"):
        assert symbol in TOKEN_DECIMALS


def test_irm_address_format():
    assert IRM_ADDRESS.startswith("0x")
    assert len(IRM_ADDRESS) == 42


# ---------------------------------------------------------------------------
# markets.py
# ---------------------------------------------------------------------------

def test_build_market_id_returns_hex():
    mid = build_market_id(
        TOKEN_ADDRESSES["USDC_E"],
        TOKEN_ADDRESSES["WETH"],
        "0x" + "00" * 20,
        IRM_ADDRESS,
        int(0.86 * WAD),
    )
    assert mid.startswith("0x")
    assert len(mid) == 66  # 0x + 64 hex chars


def test_build_market_id_deterministic():
    args = (
        TOKEN_ADDRESSES["USDC_E"],
        TOKEN_ADDRESSES["WETH"],
        "0x" + "00" * 20,
        IRM_ADDRESS,
        int(0.86 * WAD),
    )
    assert build_market_id(*args) == build_market_id(*args)


def test_build_market_id_different_for_different_params():
    mid1 = build_market_id(TOKEN_ADDRESSES["USDC_E"], TOKEN_ADDRESSES["WETH"],
                            "0x" + "00" * 20, IRM_ADDRESS, int(0.86 * WAD))
    mid2 = build_market_id(TOKEN_ADDRESSES["USDC_E"], TOKEN_ADDRESSES["WBTC"],
                            "0x" + "00" * 20, IRM_ADDRESS, int(0.86 * WAD))
    assert mid1 != mid2


def test_market_config_lltv_pct():
    mc = MarketConfig(
        name="test",
        loan_token_symbol="USDC_E",
        collateral_token_symbol="WETH",
        loan_token=TOKEN_ADDRESSES["USDC_E"],
        collateral_token=TOKEN_ADDRESSES["WETH"],
        oracle="0x" + "00" * 20,
        irm=IRM_ADDRESS,
        lltv=int(0.86 * WAD),
    )
    assert abs(mc.lltv_pct - 86.0) < 0.01


def test_market_config_params_tuple():
    mc = MarketConfig(
        name="test2",
        loan_token_symbol="USDC",
        collateral_token_symbol="WBTC",
        loan_token=TOKEN_ADDRESSES["USDC"],
        collateral_token=TOKEN_ADDRESSES["WBTC"],
        oracle="0x" + "00" * 20,
        irm=IRM_ADDRESS,
        lltv=int(0.77 * WAD),
    )
    tup = mc.to_params_tuple()
    assert len(tup) == 5
    assert tup[0] == TOKEN_ADDRESSES["USDC"]


def test_market_registry_default_markets():
    reg = MarketRegistry()
    markets = reg.list_markets()
    assert len(markets) >= 4
    names = [m.name for m in markets]
    assert "WETH/USDC_E-86" in names


def test_market_registry_get():
    reg = MarketRegistry()
    m = reg.get("WETH/USDC_E-86")
    assert m is not None
    assert m.loan_token_symbol == "USDC_E"
    assert m.collateral_token_symbol == "WETH"


def test_market_registry_get_missing():
    reg = MarketRegistry()
    assert reg.get("NONEXISTENT") is None


def test_market_registry_filter_by_tag():
    reg = MarketRegistry()
    stable = reg.filter_by_tag("stable-borrow")
    assert len(stable) >= 2


def test_market_registry_len():
    reg = MarketRegistry()
    assert len(reg) >= 4


# ---------------------------------------------------------------------------
# client.py
# ---------------------------------------------------------------------------

def test_morpho_client_mock_mode():
    client = MorphoBlueClient()
    assert client._is_mock


def test_morpho_client_supply():
    client = MorphoBlueClient()
    result = client.supply("WETH/USDC_E-86", assets=1_000_000)
    assert result.success


def test_morpho_client_supply_zero_fails():
    client = MorphoBlueClient()
    result = client.supply("WETH/USDC_E-86", assets=0)
    assert not result.success
    assert "positive" in result.error


def test_morpho_client_supply_collateral_then_borrow():
    client = MorphoBlueClient()
    col_result = client.supply_collateral("WETH/USDC_E-86", assets=1_000_000_000_000_000_000)
    assert col_result.success
    borrow_result = client.borrow("WETH/USDC_E-86", assets=500_000)
    assert borrow_result.success


def test_morpho_client_borrow_without_collateral_fails():
    client = MorphoBlueClient()
    result = client.borrow("WETH/USDC_E-86", assets=500_000)
    assert not result.success


def test_morpho_client_repay():
    client = MorphoBlueClient()
    client.supply_collateral("WETH/USDC_E-86", assets=10**18)
    client.borrow("WETH/USDC_E-86", assets=500_000)
    result = client.repay("WETH/USDC_E-86", assets=250_000)
    assert result.success


def test_morpho_client_repay_full():
    client = MorphoBlueClient()
    client.supply_collateral("WETH/USDC_E-86", assets=10**18)
    client.borrow("WETH/USDC_E-86", assets=500_000)
    result = client.repay("WETH/USDC_E-86")  # repay full
    assert result.success


def test_morpho_client_withdraw():
    client = MorphoBlueClient()
    client.supply("WETH/USDC_E-86", assets=1_000_000)
    result = client.withdraw("WETH/USDC_E-86", assets=500_000)
    assert result.success


def test_morpho_client_dry_run():
    client = MorphoBlueClient()
    result = client.supply("WETH/USDC_E-86", assets=1_000_000, dry_run=True)
    assert result.success
    assert result.details.get("dry_run") is True


def test_morpho_client_get_position():
    client = MorphoBlueClient()
    pos = client.get_position("WETH/USDC_E-86")
    assert isinstance(pos, UserPosition)
    assert pos.market_id != ""


def test_morpho_client_health_factor_no_borrow():
    client = MorphoBlueClient()
    hf = client.health_factor("WETH/USDC_E-86")
    assert hf == float("inf")


def test_morpho_client_health_factor_with_borrow():
    client = MorphoBlueClient()
    client.supply_collateral("WETH/USDC_E-86", assets=10**18)
    client.borrow("WETH/USDC_E-86", assets=1_000_000)
    hf = client.health_factor("WETH/USDC_E-86")
    assert hf > 0


def test_morpho_client_market_apy():
    client = MorphoBlueClient()
    apy = client.market_apy("WETH/USDC_E-86")
    assert "supply_apy" in apy
    assert "borrow_apy" in apy
    assert "utilisation" in apy


def test_morpho_client_liquidation_price_no_collateral():
    client = MorphoBlueClient()
    price = client.liquidation_price("WETH/USDC_E-86")
    assert price == 0.0


def test_morpho_client_approve():
    client = MorphoBlueClient()
    result = client.approve("WETH", 10**18)
    assert result.success


def test_morpho_client_unknown_market_raises():
    client = MorphoBlueClient()
    with pytest.raises(KeyError):
        client.get_position("NONEXISTENT/MARKET")


def test_morpho_client_list_markets():
    client = MorphoBlueClient()
    markets = client.list_markets()
    assert len(markets) >= 4


def test_morpho_client_get_market_state():
    client = MorphoBlueClient()
    state = client.get_market_state("WETH/USDC_E-86")
    assert isinstance(state, MarketState)


def test_tx_result_bool():
    assert bool(TxResult(True))
    assert not bool(TxResult(False))


# ---------------------------------------------------------------------------
# growth.py
# ---------------------------------------------------------------------------

def test_growth_engine_basic_cycle():
    client = MorphoBlueClient()
    engine = GrowthEngine(client, target_ltv=0.40)
    report = engine.run_growth_cycle(
        "WETH/USDC_E-86",
        collateral_assets=10**18,
        dry_run=True,
    )
    assert isinstance(report, GrowthReport)
    assert len(report.cycles) > 0


def test_growth_engine_dry_run_no_supply():
    client = MorphoBlueClient()
    engine = GrowthEngine(client)
    # Dry run should not actually change balances in a way that matters
    engine.run_growth_cycle("WETH/USDC_E-86", collateral_assets=10**18, dry_run=True)
    # Check nothing catastrophic happened
    assert len(engine.report.cycles) > 0


def test_growth_engine_unknown_market():
    client = MorphoBlueClient()
    engine = GrowthEngine(client)
    report = engine.run_growth_cycle("UNKNOWN/MARKET", collateral_assets=10**18)
    assert not report.cycles[-1].success


def test_growth_engine_report_callback():
    called = []
    client = MorphoBlueClient()
    engine = GrowthEngine(client, report_callback=called.append)
    engine.run_growth_cycle("WETH/USDC_E-86", collateral_assets=10**18, dry_run=True)
    assert len(called) > 0


def test_growth_engine_monitor_ok():
    client = MorphoBlueClient()
    engine = GrowthEngine(client)
    result = engine.monitor_and_rebalance("WETH/USDC_E-86")
    assert result.success


def test_growth_engine_grade_no_positions():
    client = MorphoBlueClient()
    engine = GrowthEngine(client)
    grade = engine.growth_grade
    assert grade in ("A+", "A", "B", "C", "D", "N/A")


def test_growth_report_summary():
    client = MorphoBlueClient()
    engine = GrowthEngine(client)
    engine.run_growth_cycle("WETH/USDC_E-86", collateral_assets=10**18, dry_run=True)
    summary = engine.report.summary()
    assert "total_cycles" in summary
    assert "success_rate" in summary
    assert 0.0 <= summary["success_rate"] <= 1.0


# ---------------------------------------------------------------------------
# simulation.py
# ---------------------------------------------------------------------------

def test_position_simulator_project():
    client = MorphoBlueClient()
    sim = PositionSimulator(client)
    result = sim.project("WETH/USDC_E-86", horizon_days=30.0)
    assert result.market_name == "WETH/USDC_E-86"
    assert result.horizon_days == 30.0
    assert "supply_usd" in result.initial
    assert "supply_usd" in result.projected


def test_position_simulator_compare_markets():
    client = MorphoBlueClient()
    sim = PositionSimulator(client)
    results = sim.compare_markets(horizon_days=30.0, supply_amount_usd=10_000.0)
    assert len(results) > 0
    # Should be sorted by net yield descending
    yields = [r["net_yield_usd"] for r in results]
    assert yields == sorted(yields, reverse=True)


def test_position_simulator_what_if_price_drop_no_borrow():
    client = MorphoBlueClient()
    sim = PositionSimulator(client)
    result = sim.what_if_price_drop("WETH/USDC_E-86", price_drop_pct=30.0)
    # No borrow → HF should be inf
    assert result["new_health_factor"] == float("inf")


def test_position_simulator_what_if_price_drop_with_borrow():
    client = MorphoBlueClient()
    client.supply_collateral("WETH/USDC_E-86", assets=10**18)
    client.borrow("WETH/USDC_E-86", assets=2_000_000)  # 2 USDC
    sim = PositionSimulator(client)
    result = sim.what_if_price_drop("WETH/USDC_E-86", price_drop_pct=50.0)
    assert "new_health_factor" in result
    assert "liquidatable" in result
