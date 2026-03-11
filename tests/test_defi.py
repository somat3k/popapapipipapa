"""Tests for DeFi / Morpho integration."""

import pytest

from app.defi.morpho import (
    KNOWN_MARKETS,
    MARKET_WMATIC_USDC,
    MockWeb3Provider,
    MorphoClient,
    Position,
    TxResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return MorphoClient(wallet_address="0x" + "ab" * 20)


@pytest.fixture
def funded_client():
    """Client with collateral already deposited."""
    c = MorphoClient(wallet_address="0x" + "cd" * 20)
    c.deposit_collateral(MARKET_WMATIC_USDC, 1000.0)
    return c


# ---------------------------------------------------------------------------
# MockWeb3Provider tests
# ---------------------------------------------------------------------------

def test_mock_supply():
    prov = MockWeb3Provider()
    result = prov.supply(MARKET_WMATIC_USDC, "0xwallet", 100.0)
    assert result.success
    assert result.tx_hash.startswith("0x")


def test_mock_borrow():
    prov = MockWeb3Provider()
    result = prov.borrow(MARKET_WMATIC_USDC, "0xwallet", 50.0)
    assert result.success


def test_mock_repay_partial():
    prov = MockWeb3Provider()
    wallet = "0xwallet"
    prov.borrow(MARKET_WMATIC_USDC, wallet, 100.0)
    result = prov.repay(MARKET_WMATIC_USDC, wallet, 40.0)
    assert result.success
    assert result.details["repaid"] == pytest.approx(40.0)
    pos = prov.get_position(MARKET_WMATIC_USDC, wallet)
    assert pos.borrow_shares == pytest.approx(60.0)


def test_mock_repay_more_than_owed():
    prov = MockWeb3Provider()
    wallet = "0xwallet"
    prov.borrow(MARKET_WMATIC_USDC, wallet, 50.0)
    result = prov.repay(MARKET_WMATIC_USDC, wallet, 200.0)
    assert result.details["repaid"] == pytest.approx(50.0)
    pos = prov.get_position(MARKET_WMATIC_USDC, wallet)
    assert pos.borrow_shares == pytest.approx(0.0)


def test_mock_withdraw():
    prov = MockWeb3Provider()
    wallet = "0xwallet"
    prov.supply(MARKET_WMATIC_USDC, wallet, 200.0)
    result = prov.withdraw(MARKET_WMATIC_USDC, wallet, 80.0)
    assert result.success
    assert result.details["withdrawn"] == pytest.approx(80.0)
    pos = prov.get_position(MARKET_WMATIC_USDC, wallet)
    assert pos.supply_shares == pytest.approx(120.0)


def test_mock_health_factor():
    prov = MockWeb3Provider()
    wallet = "0xwallet"
    prov.deposit_collateral(MARKET_WMATIC_USDC, wallet, 500.0)
    prov.borrow(MARKET_WMATIC_USDC, wallet, 100.0)
    pos = prov.get_position(MARKET_WMATIC_USDC, wallet)
    assert pos.health_factor == pytest.approx(5.0)


def test_mock_get_price():
    prov = MockWeb3Provider()
    assert prov.get_price("WMATIC") == pytest.approx(0.85)
    assert prov.get_price("USDC") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MorphoClient tests
# ---------------------------------------------------------------------------

def test_client_supply(client):
    result = client.supply(MARKET_WMATIC_USDC, 100.0)
    assert result.success


def test_client_supply_zero_fails(client):
    result = client.supply(MARKET_WMATIC_USDC, 0.0)
    assert not result.success
    assert "positive" in result.error.lower()


def test_client_supply_negative_fails(client):
    result = client.supply(MARKET_WMATIC_USDC, -50.0)
    assert not result.success


def test_client_borrow_no_collateral_fails(client):
    result = client.borrow(MARKET_WMATIC_USDC, 50.0)
    assert not result.success
    assert "collateral" in result.error.lower()


def test_client_borrow_with_collateral(funded_client):
    result = funded_client.borrow(MARKET_WMATIC_USDC, 200.0)
    assert result.success


def test_client_repay_full(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 300.0)
    result = funded_client.repay(MARKET_WMATIC_USDC)
    assert result.success
    pos = funded_client.get_position(MARKET_WMATIC_USDC)
    assert pos.borrow_shares == pytest.approx(0.0)


def test_client_repay_partial(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 300.0)
    result = funded_client.repay(MARKET_WMATIC_USDC, 100.0)
    assert result.success
    pos = funded_client.get_position(MARKET_WMATIC_USDC)
    assert pos.borrow_shares == pytest.approx(200.0)


def test_client_withdraw(funded_client):
    funded_client.supply(MARKET_WMATIC_USDC, 500.0)
    result = funded_client.withdraw(MARKET_WMATIC_USDC, 200.0)
    assert result.success


def test_client_withdraw_zero_fails(funded_client):
    result = funded_client.withdraw(MARKET_WMATIC_USDC, 0.0)
    assert not result.success


def test_client_get_position(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 100.0)
    pos = funded_client.get_position(MARKET_WMATIC_USDC)
    assert isinstance(pos, Position)
    assert pos.collateral == pytest.approx(1000.0)
    assert pos.borrow_shares == pytest.approx(100.0)


def test_client_health_factor(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 200.0)
    hf = funded_client.health_factor(MARKET_WMATIC_USDC)
    assert hf == pytest.approx(5.0)


def test_client_liquidation_price(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 200.0)
    liq = funded_client.liquidation_price(MARKET_WMATIC_USDC)
    assert liq > 0.0


def test_client_list_markets(client):
    markets = client.list_markets()
    assert "WMATIC/USDC" in markets
    assert "WETH/USDC" in markets


def test_client_market_apy(funded_client):
    apy = funded_client.market_apy(MARKET_WMATIC_USDC)
    assert "supply_apy" in apy
    assert "borrow_apy" in apy
    assert apy["supply_apy"] > 0


# ---------------------------------------------------------------------------
# Collateral swap tests
# ---------------------------------------------------------------------------

def test_collateral_swap_dry_run(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 200.0)
    result = funded_client.collateral_swap(MARKET_WMATIC_USDC, 100.0, dry_run=True)
    assert result["success"]
    assert result["dry_run"]
    assert result["swap_amount"] == pytest.approx(100.0)
    assert result["received"] > 0


def test_collateral_swap_reduces_borrow(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 200.0)
    pos_before = funded_client.get_position(MARKET_WMATIC_USDC)
    result = funded_client.collateral_swap(MARKET_WMATIC_USDC, 50.0, dry_run=False)
    assert result["success"]
    pos_after = funded_client.get_position(MARKET_WMATIC_USDC)
    assert pos_after.borrow_shares < pos_before.borrow_shares


def test_collateral_swap_insufficient_collateral(client):
    result = client.collateral_swap(MARKET_WMATIC_USDC, 9999.0)
    assert not result["success"]
    assert "collateral" in result["error"].lower()


def test_collateral_swap_slippage_guard(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 200.0)
    result = funded_client.collateral_swap(
        MARKET_WMATIC_USDC, 100.0, min_received=9999.0
    )
    assert not result["success"]
    assert "slippage" in result["error"].lower()


def test_position_is_not_liquidatable_healthy(funded_client):
    funded_client.borrow(MARKET_WMATIC_USDC, 100.0)
    pos = funded_client.get_position(MARKET_WMATIC_USDC)
    assert not pos.is_liquidatable()


def test_position_is_liquidatable_under_hf1():
    prov = MockWeb3Provider()
    wallet = "0xtest"
    prov.deposit_collateral(MARKET_WMATIC_USDC, wallet, 10.0)
    prov.borrow(MARKET_WMATIC_USDC, wallet, 100.0)
    pos = prov.get_position(MARKET_WMATIC_USDC, wallet)
    assert pos.is_liquidatable()


# ---------------------------------------------------------------------------
# Swap route validation tests
# ---------------------------------------------------------------------------

def test_collateral_swap_valid_route(funded_client):
    """Swap succeeds when a valid route is specified in swap_routes.json."""
    funded_client.borrow(MARKET_WMATIC_USDC, 200.0)
    result = funded_client.collateral_swap(
        MARKET_WMATIC_USDC, 50.0,
        from_token="WPOL", to_token="USDC_E",
        dry_run=True,
    )
    assert result["success"]
    assert result["dry_run"]
    assert "slippage_pct" in result


def test_collateral_swap_invalid_route_rejected(funded_client):
    """Swap is rejected when no matching route exists in swap_routes.json."""
    funded_client.borrow(MARKET_WMATIC_USDC, 200.0)
    result = funded_client.collateral_swap(
        MARKET_WMATIC_USDC, 50.0,
        from_token="WETH", to_token="DAI",  # not in swap_routes.json
    )
    assert not result["success"]
    assert "No collateral swap route" in result["error"]


def test_known_markets_populated_from_json():
    """KNOWN_MARKETS contains real market IDs loaded from config/markets.json."""
    from app.defi.morpho import KNOWN_MARKETS

    # All IDs sourced from the morpho registry should be proper hex strings
    for name, market_id in KNOWN_MARKETS.items():
        assert market_id.startswith("0x"), f"Market {name!r} ID not a hex string"
        # keccak256 hash = 32 bytes = 64 hex chars + "0x" prefix → 66 chars
        assert len(market_id) == 66, f"Market {name!r} ID wrong length"
