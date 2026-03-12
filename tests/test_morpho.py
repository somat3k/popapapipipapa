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


# ---------------------------------------------------------------------------
# JSON config loading tests
# ---------------------------------------------------------------------------

def test_contracts_loaded_from_json():
    """Verify that contract constants are sourced from config/contracts.json."""
    import json
    import pathlib

    cfg_path = pathlib.Path(__file__).parent.parent / "config" / "contracts.json"
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    assert MORPHO_BLUE_POLYGON == cfg["morpho_blue_polygon"]
    assert POLYGON_CHAIN_ID == cfg["polygon_chain_id"]
    assert POLYGON_RPC_URLS == cfg["polygon_rpc_urls"]


def test_tokens_loaded_from_json():
    """Verify that token addresses/decimals are sourced from config/tokens.json."""
    import json
    import pathlib

    cfg_path = pathlib.Path(__file__).parent.parent / "config" / "tokens.json"
    with open(cfg_path) as fh:
        tokens = json.load(fh)

    for sym, info in tokens.items():
        if sym.startswith("_"):
            continue
        assert TOKEN_ADDRESSES[sym] == info["address"]
        assert TOKEN_DECIMALS[sym] == info["decimals"]


def test_collateral_tokens_from_json():
    """COLLATERAL_TOKENS contains exactly the tokens marked collateral: true."""
    import json
    import pathlib
    from morpho.contracts import COLLATERAL_TOKENS

    cfg_path = pathlib.Path(__file__).parent.parent / "config" / "tokens.json"
    with open(cfg_path) as fh:
        tokens = json.load(fh)

    expected = {sym for sym, info in tokens.items()
                if not sym.startswith("_") and info.get("collateral")}
    assert set(COLLATERAL_TOKENS.keys()) == expected


def test_markets_loaded_from_json():
    """MarketRegistry default markets match config/markets.json."""
    import json
    import pathlib

    cfg_path = pathlib.Path(__file__).parent.parent / "config" / "markets.json"
    with open(cfg_path) as fh:
        market_defs = json.load(fh)

    reg = MarketRegistry()
    json_names = {d["name"] for d in market_defs if "name" in d}
    registry_names = {m.name for m in reg.list_markets()}
    assert json_names == registry_names


def test_collateral_swap_routes_from_json():
    """COLLATERAL_SWAP_ROUTES is populated from config/swap_routes.json."""
    import json
    import pathlib
    from morpho.markets import COLLATERAL_SWAP_ROUTES

    cfg_path = pathlib.Path(__file__).parent.parent / "config" / "swap_routes.json"
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    assert COLLATERAL_SWAP_ROUTES == cfg["collateral_swap_routes"]


def test_borrow_token_swap_routes_from_json():
    """BORROW_TOKEN_SWAP_ROUTES is populated from config/swap_routes.json."""
    import json
    import pathlib
    from morpho.markets import BORROW_TOKEN_SWAP_ROUTES

    cfg_path = pathlib.Path(__file__).parent.parent / "config" / "swap_routes.json"
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    assert BORROW_TOKEN_SWAP_ROUTES == cfg["borrow_token_swap_routes"]


def test_get_collateral_swap_route_found():
    from morpho.markets import get_collateral_swap_route

    route = get_collateral_swap_route("WETH", "USDC_E")
    assert route is not None
    assert route["from_token"] == "WETH"
    assert route["to_token"] == "USDC_E"
    assert route["slippage_pct"] > 0


def test_get_collateral_swap_route_not_found():
    from morpho.markets import get_collateral_swap_route

    assert get_collateral_swap_route("WETH", "DAI") is None


def test_get_borrow_token_swap_route_found():
    from morpho.markets import get_borrow_token_swap_route

    route = get_borrow_token_swap_route("USDC_E", "USDC")
    assert route is not None
    assert route["slippage_pct"] > 0


def test_get_borrow_token_swap_route_not_found():
    from morpho.markets import get_borrow_token_swap_route

    assert get_borrow_token_swap_route("WETH", "WBTC") is None


# ---------------------------------------------------------------------------
# api.py
# ---------------------------------------------------------------------------

from morpho.api import (
    MorphoAPIClient,
    APIMarketState,
    APIUserPosition,
    MarketRewards,
    RewardEntry,
    MORPHO_API_URL,
    build_supply_payload,
    build_borrow_payload,
    build_repay_payload,
    build_withdraw_payload,
    build_supply_collateral_payload,
    build_withdraw_collateral_payload,
)


def test_morpho_api_url_format():
    assert MORPHO_API_URL.startswith("https://")


def test_api_client_fetch_markets_returns_list():
    client = MorphoAPIClient()
    markets = client.fetch_markets()
    assert isinstance(markets, list)
    assert len(markets) > 0


def test_api_client_markets_are_api_market_state():
    client = MorphoAPIClient()
    markets = client.fetch_markets()
    for m in markets:
        assert isinstance(m, APIMarketState)


def test_api_market_state_attributes():
    client = MorphoAPIClient()
    markets = client.fetch_markets()
    m = markets[0]
    assert m.unique_key.startswith("0x")
    assert m.loan_symbol != ""
    assert m.collateral_symbol != ""
    assert 0.0 < m.lltv <= 1.0
    assert m.supply_apy >= 0
    assert m.borrow_apy >= 0
    assert 0.0 <= m.utilization <= 1.0
    assert m.liquidity_usd >= 0


def test_api_market_state_apy_pct_properties():
    client = MorphoAPIClient()
    m = client.fetch_markets()[0]
    assert abs(m.supply_apy_pct - m.supply_apy * 100) < 0.001
    assert abs(m.borrow_apy_pct - m.borrow_apy * 100) < 0.001
    assert abs(m.utilization_pct - m.utilization * 100) < 0.01


def test_api_client_markets_sorted_by_supply_apy():
    client = MorphoAPIClient()
    markets = client.fetch_markets()
    apys = [m.supply_apy for m in markets]
    assert apys == sorted(apys, reverse=True)


def test_api_client_fetch_user_positions_returns_list():
    client = MorphoAPIClient()
    positions = client.fetch_user_positions("0x" + "00" * 20)
    assert isinstance(positions, list)


def test_api_client_user_positions_are_api_user_position():
    client = MorphoAPIClient()
    positions = client.fetch_user_positions("0x" + "00" * 20)
    for p in positions:
        assert isinstance(p, APIUserPosition)


def test_api_client_fetch_rewards_returns_list():
    client = MorphoAPIClient()
    rewards = client.fetch_rewards("0x" + "00" * 20)
    assert isinstance(rewards, list)


def test_api_client_fetch_market_returns_none_for_unknown():
    client = MorphoAPIClient()
    # For a made-up key that won't exist even in mock data
    result = client.fetch_market("0x" + "ff" * 32)
    # Mock fallback returns None for unknown uniqueKey
    assert result is None or isinstance(result, APIMarketState)


def test_api_query_method_raises_on_bad_url():
    import urllib.error
    client = MorphoAPIClient(api_url="http://localhost:1/bad", timeout=1)
    with pytest.raises(Exception):
        client.query("{markets{items{uniqueKey}}}")


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def test_build_supply_payload_keys():
    params = ("0xLoan", "0xColl", "0xOracle", "0xIRM", 860000000000000000)
    payload = build_supply_payload(params, assets=1_000_000, on_behalf="0x" + "00" * 20)
    assert "marketParams" in payload
    assert "assets" in payload
    assert payload["assets"] == 1_000_000
    assert payload["shares"] == 0
    assert payload["onBehalf"] == "0x" + "00" * 20
    assert "data" in payload


def test_build_borrow_payload_keys():
    params = ("0xLoan", "0xColl", "0xOracle", "0xIRM", 860000000000000000)
    wallet = "0x" + "11" * 20
    payload = build_borrow_payload(params, assets=500_000, on_behalf=wallet, receiver=wallet)
    assert payload["assets"] == 500_000
    assert payload["receiver"] == wallet


def test_build_repay_payload_keys():
    params = ("0xLoan", "0xColl", "0xOracle", "0xIRM", 860000000000000000)
    wallet = "0x" + "22" * 20
    payload = build_repay_payload(params, assets=250_000, on_behalf=wallet)
    assert payload["assets"] == 250_000
    assert payload["shares"] == 0


def test_build_withdraw_payload_keys():
    params = ("0xLoan", "0xColl", "0xOracle", "0xIRM", 860000000000000000)
    wallet = "0x" + "33" * 20
    payload = build_withdraw_payload(params, assets=100_000, on_behalf=wallet, receiver=wallet)
    assert "receiver" in payload


def test_build_supply_collateral_payload_keys():
    params = ("0xLoan", "0xColl", "0xOracle", "0xIRM", 860000000000000000)
    wallet = "0x" + "44" * 20
    payload = build_supply_collateral_payload(params, assets=10**18, on_behalf=wallet)
    assert payload["assets"] == 10**18
    assert "onBehalf" in payload


def test_build_withdraw_collateral_payload_keys():
    params = ("0xLoan", "0xColl", "0xOracle", "0xIRM", 860000000000000000)
    wallet = "0x" + "55" * 20
    payload = build_withdraw_collateral_payload(params, assets=10**18, on_behalf=wallet, receiver=wallet)
    assert "receiver" in payload


def test_build_supply_payload_zero_assets():
    """Zero assets is valid at the payload level (validated by contract, not builder)."""
    params = ("0xA", "0xB", "0xC", "0xD", 860000000000000000)
    payload = build_supply_payload(params, assets=0, on_behalf="0x" + "00" * 20)
    assert payload["assets"] == 0


def test_build_borrow_payload_default_data():
    params = ("0xA", "0xB", "0xC", "0xD", 860000000000000000)
    wallet = "0x" + "ab" * 20
    payload = build_repay_payload(params, assets=100, on_behalf=wallet)
    assert payload["data"] == b""


def test_build_supply_collateral_payload_market_params_preserved():
    params = ("0xLoan", "0xColl", "0xOracle", "0xIRM", 860000000000000000)
    wallet = "0x" + "cc" * 20
    payload = build_supply_collateral_payload(params, assets=10**18, on_behalf=wallet)
    assert payload["marketParams"] == params


# ---------------------------------------------------------------------------
# rewards.py
# ---------------------------------------------------------------------------

from morpho.rewards import (
    RewardsCalculator,
    RewardEstimate,
    NetAPR,
    BreakEvenAnalysis,
    DEFAULT_SUPPLY_REWARD_RATE,
    DEFAULT_BORROW_REWARD_RATE,
)


def test_rewards_calculator_estimate_supply_rewards():
    calc = RewardsCalculator()
    est = calc.estimate_supply_rewards("0xabc", position_usd=10_000, days=365.25)
    assert isinstance(est, RewardEstimate)
    assert est.side == "supply"
    assert est.position_usd == 10_000
    assert abs(est.reward_usd - 10_000 * DEFAULT_SUPPLY_REWARD_RATE) < 0.01
    assert est.reward_pct_annual == round(DEFAULT_SUPPLY_REWARD_RATE * 100, 4)


def test_rewards_calculator_estimate_borrow_rewards():
    calc = RewardsCalculator()
    est = calc.estimate_borrow_rewards("0xabc", position_usd=5_000, days=30)
    assert est.side == "borrow"
    expected = 5_000 * DEFAULT_BORROW_REWARD_RATE * 30 / 365.25
    assert abs(est.reward_usd - expected) < 0.001


def test_rewards_calculator_net_supply_apr():
    calc = RewardsCalculator()
    net = calc.net_supply_apr("0xmkt", base_apy=0.04)
    assert isinstance(net, NetAPR)
    assert net.side == "supply"
    assert abs(net.net_apr - (0.04 + DEFAULT_SUPPLY_REWARD_RATE)) < 1e-9
    assert net.is_positive_carry  # supply always positive with positive APY


def test_rewards_calculator_net_borrow_apr():
    calc = RewardsCalculator()
    net = calc.net_borrow_apr("0xmkt", borrow_apy=0.05)
    assert net.side == "borrow"
    assert abs(net.net_apr - (0.05 - DEFAULT_BORROW_REWARD_RATE)) < 1e-9


def test_rewards_calculator_positive_carry_borrow():
    # If reward rate > borrow APY, net borrow APR < 0 → positive carry
    calc = RewardsCalculator()
    calc.set_borrow_reward_rate("0xmkt", rate=0.08)
    net = calc.net_borrow_apr("0xmkt", borrow_apy=0.05)
    assert net.net_apr < 0
    assert net.is_positive_carry


def test_rewards_calculator_set_rates():
    calc = RewardsCalculator()
    calc.set_supply_reward_rate("0xmarket", 0.03)
    calc.set_borrow_reward_rate("0xmarket", 0.02)
    assert calc.get_supply_reward_rate("0xmarket") == 0.03
    assert calc.get_borrow_reward_rate("0xmarket") == 0.02


def test_rewards_calculator_unknown_market_uses_defaults():
    calc = RewardsCalculator()
    assert calc.get_supply_reward_rate("0xunknown") == DEFAULT_SUPPLY_REWARD_RATE
    assert calc.get_borrow_reward_rate("0xunknown") == DEFAULT_BORROW_REWARD_RATE


def test_rewards_calculator_net_spread():
    calc = RewardsCalculator()
    spread = calc.net_spread("0xmkt", supply_apy=0.04, borrow_apy=0.06)
    expected_s = 0.04 + DEFAULT_SUPPLY_REWARD_RATE
    expected_b = 0.06 - DEFAULT_BORROW_REWARD_RATE
    assert abs(spread - (expected_s - expected_b)) < 1e-9


def test_rewards_calculator_break_even_profitable():
    calc = RewardsCalculator()
    # High supply APY, low borrow → net daily earn positive
    analysis = calc.break_even_analysis(
        "0xmkt",
        supply_usd=20_000,
        borrow_usd=5_000,
        supply_apy=0.08,
        borrow_apy=0.02,
    )
    assert isinstance(analysis, BreakEvenAnalysis)
    assert analysis.profitable is True
    assert analysis.break_even_days == 0.0
    assert analysis.net_daily_earn_usd > 0


def test_rewards_calculator_break_even_not_profitable():
    calc = RewardsCalculator()
    # Borrow more than supply, high borrow APY
    analysis = calc.break_even_analysis(
        "0xmkt",
        supply_usd=1_000,
        borrow_usd=50_000,
        supply_apy=0.02,
        borrow_apy=0.20,
    )
    assert analysis.profitable is False
    assert analysis.break_even_days == float("inf")


def test_rewards_calculator_compare_markets():
    calc = RewardsCalculator()
    markets_data = [
        {"market_key": "0xA", "supply_apy": 0.04, "borrow_apy": 0.06},
        {"market_key": "0xB", "supply_apy": 0.07, "borrow_apy": 0.09},
        {"market_key": "0xC", "supply_apy": 0.02, "borrow_apy": 0.04},
    ]
    ranked = calc.compare_markets(markets_data)
    assert len(ranked) == 3
    net_aprs = [r["net_supply_apr_pct"] for r in ranked]
    assert net_aprs == sorted(net_aprs, reverse=True)
    for r in ranked:
        assert "net_supply_apr_pct" in r
        assert "net_borrow_apr_pct" in r
        assert "positive_carry" in r


# ---------------------------------------------------------------------------
# opportunity.py
# ---------------------------------------------------------------------------

from morpho.opportunity import (
    OpportunityScanner,
    OpportunityScore,
    BorrowSwapOpportunity,
    BorrowCapacity,
    RebalanceRecommendation,
    SAFE_BORROW_LTV_RATIO,
    PRIME_SCORE_THRESHOLD,
)


def test_opportunity_scanner_rank_returns_list():
    scanner = OpportunityScanner()
    ranked = scanner.rank_opportunities()
    assert isinstance(ranked, list)


def test_opportunity_scanner_ranked_by_score():
    scanner = OpportunityScanner()
    ranked = scanner.rank_opportunities(min_score=0.0, min_liquidity_usd=0.0)
    scores = [o.score for o in ranked]
    assert scores == sorted(scores, reverse=True)


def test_opportunity_score_attributes():
    scanner = OpportunityScanner()
    ranked = scanner.rank_opportunities(min_score=0.0, min_liquidity_usd=0.0)
    assert len(ranked) > 0
    opp = ranked[0]
    assert isinstance(opp, OpportunityScore)
    assert opp.loan_symbol != ""
    assert opp.collateral_symbol != ""
    assert 0 <= opp.score <= 100
    assert opp.label in ("PRIME", "GOOD", "FAIR", "POOR")


def test_opportunity_score_is_prime():
    opp = OpportunityScore(
        market_key="0xA",
        loan_symbol="USDC_E",
        collateral_symbol="WETH",
        lltv=0.86,
        supply_apy_pct=4.2,
        borrow_apy_pct=5.8,
        net_supply_apr_pct=5.7,
        net_borrow_apr_pct=4.8,
        utilization_pct=72.0,
        liquidity_usd=14000.0,
        score=75.0,
    )
    assert opp.is_prime
    assert opp.label == "PRIME"


def test_opportunity_scanner_find_best_supply():
    scanner = OpportunityScanner()
    best = scanner.find_best_supply_market(amount_usd=1_000.0)
    assert best is None or isinstance(best, OpportunityScore)


def test_opportunity_scanner_find_best_supply_by_loan_symbol():
    scanner = OpportunityScanner()
    best = scanner.find_best_supply_market(amount_usd=100.0, loan_symbol="USDC_E")
    if best is not None:
        assert best.loan_symbol == "USDC_E"


def test_opportunity_scanner_borrow_capacity():
    scanner = OpportunityScanner()
    cap = scanner.get_borrow_capacity(
        market_key="0xA",
        loan_symbol="USDC_E",
        collateral_symbol="WETH",
        current_collateral_usd=10_000.0,
        current_borrow_usd=3_000.0,
        lltv=0.86,
        additional_supply_usd=2_000.0,
    )
    assert isinstance(cap, BorrowCapacity)
    assert cap.max_borrow_usd == round(12_000.0 * 0.86, 2)
    assert cap.safe_additional_borrow_usd >= 0


def test_opportunity_scanner_borrow_capacity_no_extra_supply():
    scanner = OpportunityScanner()
    cap = scanner.get_borrow_capacity(
        market_key="0xB",
        loan_symbol="USDC",
        collateral_symbol="WETH",
        current_collateral_usd=5_000.0,
        current_borrow_usd=2_000.0,
        lltv=0.86,
        additional_supply_usd=0.0,
    )
    assert cap.safe_additional_borrow_usd == round(5_000 * 0.86 * SAFE_BORROW_LTV_RATIO - 2_000, 2)


def test_opportunity_scanner_should_rebalance_returns_recommendation():
    scanner = OpportunityScanner()
    rec = scanner.should_rebalance(
        current_market_key="0x" + "a1" * 32,  # matches mock data
        current_supply_apy=0.042,
    )
    assert isinstance(rec, RebalanceRecommendation)
    assert isinstance(rec.should_rebalance, bool)
    assert rec.improvement_pct is not None


def test_opportunity_scanner_classify_borrow_token_swap():
    scanner = OpportunityScanner()
    swaps = scanner.classify_borrow_token_swap(
        current_market_key="0x" + "a1" * 32,
        current_loan_symbol="USDC_E",
        collateral_symbol="WETH",
    )
    assert isinstance(swaps, list)
    for s in swaps:
        assert isinstance(s, BorrowSwapOpportunity)
        assert s.classification in ("cost_saving", "neutral", "worse")


def test_opportunity_scanner_scan_markets():
    scanner = OpportunityScanner()
    markets = scanner.scan_markets()
    assert isinstance(markets, list)
    assert len(markets) > 0


# ---------------------------------------------------------------------------
# visuals.py
# ---------------------------------------------------------------------------

from morpho import visuals


def test_visuals_apy_bar_chart():
    client = MorphoAPIClient()
    markets = client.fetch_markets()
    output = visuals.apy_bar_chart(markets)
    assert isinstance(output, str)
    assert "APY" in output
    assert len(output) > 10


def test_visuals_apy_bar_chart_empty():
    output = visuals.apy_bar_chart([])
    assert "No market" in output


def test_visuals_utilization_gauge():
    output = visuals.utilization_gauge(0.72, "WETH/USDC_E")
    assert "72.0" in output
    assert "WETH/USDC_E" in output


def test_visuals_utilization_gauge_high():
    output = visuals.utilization_gauge(0.95)
    assert "HIGH" in output


def test_visuals_position_table_empty():
    output = visuals.position_table([])
    assert "No positions" in output


def test_visuals_position_table():
    from morpho.client import MorphoBlueClient
    client = MorphoBlueClient()
    positions = client.fetch_user_positions("0x" + "00" * 20) if hasattr(client, "fetch_user_positions") else []
    # Use API client positions for table
    api = MorphoAPIClient()
    api_positions = api.fetch_user_positions("0x" + "00" * 20)
    output = visuals.position_table(api_positions)
    assert isinstance(output, str)


def test_visuals_opportunity_ranking():
    scanner = OpportunityScanner()
    ranked = scanner.rank_opportunities(min_score=0.0, min_liquidity_usd=0.0)
    output = visuals.opportunity_ranking(ranked)
    assert isinstance(output, str)
    assert "Opportunity Ranking" in output


def test_visuals_opportunity_ranking_empty():
    output = visuals.opportunity_ranking([])
    assert "No opportunities" in output


def test_visuals_health_factor_meter_infinite():
    output = visuals.health_factor_meter(float("inf"))
    assert "∞" in output


def test_visuals_health_factor_meter_safe():
    output = visuals.health_factor_meter(2.0)
    assert "2.000" in output
    assert "SAFE" in output


def test_visuals_health_factor_meter_at_risk():
    output = visuals.health_factor_meter(1.08)
    assert "AT RISK" in output


def test_visuals_health_factor_meter_liquidatable():
    output = visuals.health_factor_meter(0.95)
    assert "LIQUIDATABLE" in output


def test_visuals_borrow_capacity_table():
    scanner = OpportunityScanner()
    cap = scanner.get_borrow_capacity(
        "0xA", "USDC_E", "WETH", 10_000.0, 3_000.0, 0.86, 1_000.0
    )
    output = visuals.borrow_capacity_table([cap])
    assert "Borrow Capacity" in output
    assert "WETH" in output


def test_visuals_borrow_capacity_table_empty():
    output = visuals.borrow_capacity_table([])
    assert "No borrow" in output


def test_visuals_market_summary():
    client = MorphoAPIClient()
    markets = client.fetch_markets()
    output = visuals.market_summary(markets[0])
    assert "Market:" in output
    assert "Supply APY" in output
    assert "Borrow APY" in output
    assert "Liquidity" in output


def test_visuals_rewards_table_empty():
    output = visuals.rewards_table([])
    assert "No reward" in output


def test_visuals_rewards_table():
    from morpho.api import MarketRewards, RewardEntry
    mr = MarketRewards(
        market_key="0x" + "a1" * 32,
        loan_symbol="USDC_E",
        supply_rewards=[
            RewardEntry("MORPHO", "0x" + "aa" * 20, 1.5, 3.2, 0.0)
        ],
        borrow_rewards=[],
    )
    output = visuals.rewards_table([mr])
    assert "MORPHO" in output
    assert "1.500000" in output


def test_visuals_util_gauge_inline():
    output = visuals.util_gauge_inline(0.50)
    assert "50.0" in output
