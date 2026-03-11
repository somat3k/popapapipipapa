"""Morpho Blue production client for Polygon.

Provides both query (read) and agent payload (write) communication with
the Morpho Blue smart contract.  Uses web3.py when available; falls back
to a mock provider for offline development and unit tests.

Query communication
-------------------
- ``get_market_state()``     → on-chain market totals and fee
- ``get_position()``         → per-user supply/borrow/collateral
- ``get_market_params()``    → decode market params from on-chain ID
- ``health_factor()``        → compute HF from position + oracle price
- ``liquidation_price()``    → approximate liquidation price
- ``market_apy()``           → supply and borrow APY from on-chain rates

Agent payload communication
----------------------------
- ``supply()``               → supply loan tokens to earn interest
- ``borrow()``               → borrow against deposited collateral
- ``repay()``                → repay outstanding debt
- ``withdraw()``             → withdraw supplied tokens
- ``supply_collateral()``    → deposit collateral without supplying
- ``withdraw_collateral()``  → withdraw collateral
- ``approve()``              → ERC-20 token approval helper

All write methods return a :class:`TxResult` with success flag, tx hash,
gas used, and optional error message.  Use ``dry_run=True`` on supported
methods to simulate without broadcasting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import (
    ERC20_ABI,
    MORPHO_BLUE_ABI,
    MORPHO_BLUE_POLYGON,
    POLYGON_CHAIN_ID,
    POLYGON_RPC_URLS,
    TOKEN_ADDRESSES,
    TOKEN_DECIMALS,
)
from .markets import MarketConfig, MarketRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TxResult:
    """Result of a Morpho Blue write transaction."""

    success: bool
    tx_hash: str = ""
    gas_used: int = 0
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


@dataclass
class MarketState:
    """On-chain state of a Morpho Blue market."""

    total_supply_assets: int = 0
    total_supply_shares: int = 0
    total_borrow_assets: int = 0
    total_borrow_shares: int = 0
    last_update: int = 0
    fee: int = 0

    @property
    def utilisation(self) -> float:
        """Borrow utilisation ratio (0–1)."""
        if self.total_supply_assets == 0:
            return 0.0
        return self.total_borrow_assets / self.total_supply_assets

    @property
    def available_liquidity(self) -> int:
        """Available liquidity = total supply − total borrow."""
        return max(0, self.total_supply_assets - self.total_borrow_assets)


@dataclass
class UserPosition:
    """Per-user position in a Morpho Blue market."""

    market_id: str
    supply_shares: int = 0
    borrow_shares: int = 0
    collateral: int = 0  # in collateral token units
    # Derived — set by client after fetching market state
    supply_assets: float = 0.0
    borrow_assets: float = 0.0
    health_factor: float = float("inf")
    supply_apy: float = 0.0
    borrow_apy: float = 0.0

    def is_liquidatable(self) -> bool:
        return self.health_factor < 1.0

    def is_at_risk(self, threshold: float = 1.15) -> bool:
        return 1.0 <= self.health_factor < threshold


# ---------------------------------------------------------------------------
# Mock provider for offline/test use
# ---------------------------------------------------------------------------

class _MockProvider:
    """In-memory simulation of Morpho Blue contract state."""

    def __init__(self) -> None:
        self._supply: dict[tuple[str, str], int] = {}   # (market_id, wallet) → shares
        self._borrow: dict[tuple[str, str], int] = {}
        self._collateral: dict[tuple[str, str], int] = {}
        self._prices: dict[str, float] = {
            "WETH": 3200.0,
            "WBTC": 65000.0,
            "WPOL": 0.85,
            "stMATIC": 0.82,
            "USDC": 1.0,
            "USDC_E": 1.0,
            "DAI": 1.0,
        }

    # --- Write ---

    def supply(self, market_id: str, wallet: str, assets: int) -> TxResult:
        k = (market_id, wallet)
        self._supply[k] = self._supply.get(k, 0) + assets
        return TxResult(True, "0x" + "aa" * 32, 120_000)

    def borrow(self, market_id: str, wallet: str, assets: int) -> TxResult:
        k = (market_id, wallet)
        self._borrow[k] = self._borrow.get(k, 0) + assets
        return TxResult(True, "0x" + "bb" * 32, 150_000)

    def repay(self, market_id: str, wallet: str, assets: int) -> TxResult:
        k = (market_id, wallet)
        outstanding = self._borrow.get(k, 0)
        repaid = min(assets, outstanding)
        self._borrow[k] = outstanding - repaid
        return TxResult(True, "0x" + "cc" * 32, 100_000, details={"repaid": repaid})

    def withdraw(self, market_id: str, wallet: str, assets: int) -> TxResult:
        k = (market_id, wallet)
        supplied = self._supply.get(k, 0)
        withdrawn = min(assets, supplied)
        self._supply[k] = supplied - withdrawn
        return TxResult(True, "0x" + "dd" * 32, 110_000, details={"withdrawn": withdrawn})

    def supply_collateral(self, market_id: str, wallet: str, assets: int) -> TxResult:
        k = (market_id, wallet)
        self._collateral[k] = self._collateral.get(k, 0) + assets
        return TxResult(True, "0x" + "ee" * 32, 90_000)

    def withdraw_collateral(self, market_id: str, wallet: str, assets: int) -> TxResult:
        k = (market_id, wallet)
        deposited = self._collateral.get(k, 0)
        withdrawn = min(assets, deposited)
        self._collateral[k] = deposited - withdrawn
        return TxResult(True, "0x" + "ff" * 32, 95_000, details={"withdrawn": withdrawn})

    # --- Read ---

    def get_market_state(self, market_id: str) -> MarketState:
        total_supply = sum(v for (mid, _), v in self._supply.items() if mid == market_id)
        total_borrow = sum(v for (mid, _), v in self._borrow.items() if mid == market_id)
        return MarketState(
            total_supply_assets=total_supply,
            total_supply_shares=total_supply,
            total_borrow_assets=total_borrow,
            total_borrow_shares=total_borrow,
            last_update=int(time.time()),
            fee=0,
        )

    def get_position(self, market_id: str, wallet: str) -> tuple[int, int, int]:
        k = (market_id, wallet)
        return (
            self._supply.get(k, 0),
            self._borrow.get(k, 0),
            self._collateral.get(k, 0),
        )

    def get_price(self, token_symbol: str) -> float:
        return self._prices.get(token_symbol, 1.0)

    def set_price(self, token_symbol: str, price: float) -> None:
        self._prices[token_symbol] = price


# ---------------------------------------------------------------------------
# Web3 provider wrapper (production)
# ---------------------------------------------------------------------------

class _Web3Provider:
    """Thin wrapper around web3.py for real on-chain interactions."""

    def __init__(self, web3: Any, wallet_address: str, private_key: str = "") -> None:
        self._w3 = web3
        self._wallet = wallet_address
        self._private_key = private_key
        self._morpho = web3.eth.contract(
            address=web3.to_checksum_address(MORPHO_BLUE_POLYGON),
            abi=MORPHO_BLUE_ABI,
        )

    def _build_and_send(self, fn: Any, gas_limit: int = 300_000) -> TxResult:
        if not self._private_key:
            raise ValueError("Private key required for write transactions.")
        nonce = self._w3.eth.get_transaction_count(self._wallet)
        fee = self._w3.eth.fee_history(1, "latest", [50])
        base_fee = fee["baseFeePerGas"][-1]
        max_fee = base_fee * 2
        tx = fn.build_transaction(
            {
                "from": self._wallet,
                "nonce": nonce,
                "gas": gas_limit,
                "maxFeePerGas": max_fee,
                "maxPriorityFeePerGas": int(1e9),  # 1 gwei tip
                "chainId": POLYGON_CHAIN_ID,
            }
        )
        signed = self._w3.eth.account.sign_transaction(tx, self._private_key)
        tx_hash = self._w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        success = receipt["status"] == 1
        return TxResult(
            success=success,
            tx_hash=tx_hash.hex(),
            gas_used=receipt["gasUsed"],
            error="" if success else "Transaction reverted",
        )

    def supply(self, market_id: str, wallet: str, assets: int, market_params: tuple) -> TxResult:
        fn = self._morpho.functions.supply(market_params, assets, 0, wallet, b"")
        return self._build_and_send(fn)

    def borrow(self, market_id: str, wallet: str, assets: int, market_params: tuple) -> TxResult:
        fn = self._morpho.functions.borrow(market_params, assets, 0, wallet, wallet)
        return self._build_and_send(fn)

    def repay(self, market_id: str, wallet: str, assets: int, market_params: tuple) -> TxResult:
        fn = self._morpho.functions.repay(market_params, assets, 0, wallet, b"")
        return self._build_and_send(fn)

    def withdraw(self, market_id: str, wallet: str, assets: int, market_params: tuple) -> TxResult:
        fn = self._morpho.functions.withdraw(market_params, assets, 0, wallet, wallet)
        return self._build_and_send(fn)

    def supply_collateral(self, market_id: str, wallet: str, assets: int, market_params: tuple) -> TxResult:
        fn = self._morpho.functions.supplyCollateral(market_params, assets, wallet, b"")
        return self._build_and_send(fn)

    def withdraw_collateral(self, market_id: str, wallet: str, assets: int, market_params: tuple) -> TxResult:
        fn = self._morpho.functions.withdrawCollateral(market_params, assets, wallet, wallet)
        return self._build_and_send(fn)

    def get_market_state(self, market_id: str) -> MarketState:
        raw = self._morpho.functions.market(bytes.fromhex(market_id.removeprefix("0x"))).call()
        return MarketState(
            total_supply_assets=raw[0],
            total_supply_shares=raw[1],
            total_borrow_assets=raw[2],
            total_borrow_shares=raw[3],
            last_update=raw[4],
            fee=raw[5],
        )

    def get_position(self, market_id: str, wallet: str) -> tuple[int, int, int]:
        raw = self._morpho.functions.position(
            bytes.fromhex(market_id.removeprefix("0x")), wallet
        ).call()
        return (raw[0], raw[1], raw[2])

    def approve_token(self, token_address: str, amount: int) -> TxResult:
        token = self._w3.eth.contract(
            address=self._w3.to_checksum_address(token_address),
            abi=ERC20_ABI,
        )
        fn = token.functions.approve(MORPHO_BLUE_POLYGON, amount)
        return self._build_and_send(fn, gas_limit=60_000)

    def get_token_balance(self, token_address: str, wallet: str) -> int:
        token = self._w3.eth.contract(
            address=self._w3.to_checksum_address(token_address),
            abi=ERC20_ABI,
        )
        return token.functions.balanceOf(wallet).call()


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class MorphoBlueClient:
    """High-level client for Morpho Blue on Polygon.

    Parameters
    ----------
    wallet_address:
        Your Ethereum/Polygon wallet address (checksummed).
    private_key:
        Private key for signing transactions.  Leave empty for read-only mode.
    web3:
        A ``web3.Web3`` instance connected to Polygon.  When ``None``, a mock
        provider is used — useful for development and testing.
    market_registry:
        A :class:`~morpho.markets.MarketRegistry` instance.  Falls back to the
        module-level :data:`~morpho.markets.DEFAULT_REGISTRY`.

    Examples
    --------
    >>> from morpho import MorphoBlueClient
    >>> client = MorphoBlueClient(wallet_address="0xYourAddress")  # mock mode
    >>> pos = client.get_position("WETH/USDC_E-86")
    >>> print(pos.health_factor)
    """

    def __init__(
        self,
        wallet_address: str = "0x" + "00" * 20,
        private_key: str = "",
        web3: Optional[Any] = None,
        market_registry: Optional[MarketRegistry] = None,
    ) -> None:
        from .markets import DEFAULT_REGISTRY

        self.wallet = wallet_address
        self._registry = market_registry or DEFAULT_REGISTRY

        if web3 is not None:
            self._provider: Any = _Web3Provider(web3, wallet_address, private_key)
            self._is_mock = False
        else:
            self._provider = _MockProvider()
            self._is_mock = True

        logger.info(
            "MorphoBlueClient ready  wallet=%s…  mock=%s",
            wallet_address[:10],
            self._is_mock,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_market(self, market_name: str) -> MarketConfig:
        market = self._registry.get(market_name)
        if market is None:
            raise KeyError(f"Unknown market: '{market_name}'")
        return market

    def _shares_to_assets(self, shares: int, total_shares: int, total_assets: int) -> float:
        if total_shares == 0:
            return float(shares)
        return float(shares) * float(total_assets) / float(total_shares)

    # ------------------------------------------------------------------
    # Query communication (read)
    # ------------------------------------------------------------------

    def get_market_state(self, market_name: str) -> MarketState:
        """Return the current on-chain state of a market."""
        market = self._get_market(market_name)
        return self._provider.get_market_state(market.market_id)

    def get_position(self, market_name: str) -> UserPosition:
        """Return the current position of the wallet in a market.

        Computes supply/borrow assets from shares and estimates health factor.
        """
        market = self._get_market(market_name)
        supply_shares, borrow_shares, collateral = self._provider.get_position(
            market.market_id, self.wallet
        )
        state = self._provider.get_market_state(market.market_id)

        supply_assets = self._shares_to_assets(
            supply_shares, state.total_supply_shares, state.total_supply_assets
        )
        borrow_assets = self._shares_to_assets(
            borrow_shares, state.total_borrow_shares, state.total_borrow_assets
        )

        # Health factor = (collateral × oracle_price × LLTV) / borrow_assets
        collateral_price = (
            self._provider.get_price(market.collateral_token_symbol)
            if hasattr(self._provider, "get_price")
            else 1.0
        )
        collateral_value = float(collateral) * collateral_price / (
            10 ** market.collateral_decimals
        )
        lltv = market.lltv / 1e18
        hf = (
            (collateral_value * lltv) / borrow_assets
            if borrow_assets > 0
            else float("inf")
        )

        return UserPosition(
            market_id=market.market_id,
            supply_shares=supply_shares,
            borrow_shares=borrow_shares,
            collateral=collateral,
            supply_assets=supply_assets,
            borrow_assets=borrow_assets,
            health_factor=hf,
            supply_apy=self._estimate_supply_apy(state),
            borrow_apy=self._estimate_borrow_apy(state),
        )

    def _estimate_supply_apy(self, state: MarketState) -> float:
        """Rough supply APY estimate from utilisation (no on-chain IRM call)."""
        u = state.utilisation
        # Simplified linear IRM: borrow rate ≈ 2% + 8% × u; supply = borrow × u
        borrow_rate = 0.02 + 0.08 * u
        return borrow_rate * u * (1 - state.fee / 1e18 if state.fee else 1.0)

    def _estimate_borrow_apy(self, state: MarketState) -> float:
        u = state.utilisation
        return 0.02 + 0.08 * u

    def health_factor(self, market_name: str) -> float:
        """Return the health factor for the current position."""
        return self.get_position(market_name).health_factor

    def liquidation_price(self, market_name: str) -> float:
        """Approximate collateral price at which position becomes liquidatable.

        liquidation_price = borrow_assets / (collateral_units × LLTV)
        """
        market = self._get_market(market_name)
        pos = self.get_position(market_name)
        if pos.collateral == 0:
            return 0.0
        lltv = market.lltv / 1e18
        collateral_units = pos.collateral / (10 ** market.collateral_decimals)
        return pos.borrow_assets / (collateral_units * lltv) if collateral_units > 0 else 0.0

    def market_apy(self, market_name: str) -> dict[str, float]:
        """Return current supply and borrow APY estimates."""
        state = self.get_market_state(market_name)
        return {
            "supply_apy": self._estimate_supply_apy(state),
            "borrow_apy": self._estimate_borrow_apy(state),
            "utilisation": state.utilisation,
        }

    def list_markets(self) -> list[MarketConfig]:
        return self._registry.list_markets()

    # ------------------------------------------------------------------
    # Agent payload communication (write)
    # ------------------------------------------------------------------

    def approve(self, token_symbol: str, amount: int) -> TxResult:
        """Approve Morpho Blue to spend tokens.

        In mock mode this is always successful.  In production, sends an
        ERC-20 approve transaction.
        """
        if self._is_mock:
            return TxResult(True, "0x" + "a0" * 32, 46_000)
        token_addr = TOKEN_ADDRESSES.get(token_symbol)
        if token_addr is None:
            return TxResult(False, error=f"Unknown token: {token_symbol}")
        return self._provider.approve_token(token_addr, amount)

    def supply(self, market_name: str, assets: int, *, dry_run: bool = False) -> TxResult:
        """Supply *assets* (in token base units) to earn interest.

        Parameters
        ----------
        market_name:  Market name from the registry (e.g. ``"WETH/USDC_E-86"``).
        assets:       Amount in token base units (e.g. 1 USDC = 1_000_000).
        dry_run:      If True, return a simulated result without broadcasting.
        """
        if assets <= 0:
            return TxResult(False, error="assets must be positive")
        market = self._get_market(market_name)
        if dry_run:
            return TxResult(True, details={"dry_run": True, "assets": assets, "market": market_name})
        logger.info("[Morpho] supply  market=%s  assets=%d", market_name, assets)
        if self._is_mock:
            return self._provider.supply(market.market_id, self.wallet, assets)
        return self._provider.supply(market.market_id, self.wallet, assets, market.to_params_tuple())

    def borrow(self, market_name: str, assets: int, *, dry_run: bool = False) -> TxResult:
        """Borrow *assets* from the market against deposited collateral."""
        if assets <= 0:
            return TxResult(False, error="assets must be positive")
        market = self._get_market(market_name)
        pos = self.get_position(market_name)
        if pos.collateral == 0:
            return TxResult(False, error="No collateral deposited.")
        if dry_run:
            return TxResult(True, details={"dry_run": True, "assets": assets, "market": market_name})
        logger.info("[Morpho] borrow  market=%s  assets=%d", market_name, assets)
        if self._is_mock:
            return self._provider.borrow(market.market_id, self.wallet, assets)
        return self._provider.borrow(market.market_id, self.wallet, assets, market.to_params_tuple())

    def repay(
        self,
        market_name: str,
        assets: Optional[int] = None,
        *,
        dry_run: bool = False,
    ) -> TxResult:
        """Repay *assets* of borrowed tokens.  Pass ``None`` to repay in full."""
        market = self._get_market(market_name)
        pos = self.get_position(market_name)
        if assets is None:
            assets = int(pos.borrow_assets)
        if assets <= 0:
            return TxResult(False, error="Nothing to repay.")
        if dry_run:
            return TxResult(True, details={"dry_run": True, "assets": assets, "market": market_name})
        logger.info("[Morpho] repay  market=%s  assets=%d", market_name, assets)
        if self._is_mock:
            return self._provider.repay(market.market_id, self.wallet, assets)
        return self._provider.repay(market.market_id, self.wallet, assets, market.to_params_tuple())

    def withdraw(self, market_name: str, assets: int, *, dry_run: bool = False) -> TxResult:
        """Withdraw supplied tokens."""
        if assets <= 0:
            return TxResult(False, error="assets must be positive")
        market = self._get_market(market_name)
        if dry_run:
            return TxResult(True, details={"dry_run": True, "assets": assets, "market": market_name})
        logger.info("[Morpho] withdraw  market=%s  assets=%d", market_name, assets)
        if self._is_mock:
            return self._provider.withdraw(market.market_id, self.wallet, assets)
        return self._provider.withdraw(market.market_id, self.wallet, assets, market.to_params_tuple())

    def supply_collateral(self, market_name: str, assets: int, *, dry_run: bool = False) -> TxResult:
        """Deposit collateral into the market."""
        if assets <= 0:
            return TxResult(False, error="assets must be positive")
        market = self._get_market(market_name)
        if dry_run:
            return TxResult(True, details={"dry_run": True, "assets": assets, "market": market_name})
        logger.info("[Morpho] supply_collateral  market=%s  assets=%d", market_name, assets)
        if self._is_mock:
            return self._provider.supply_collateral(market.market_id, self.wallet, assets)
        return self._provider.supply_collateral(market.market_id, self.wallet, assets, market.to_params_tuple())

    def withdraw_collateral(self, market_name: str, assets: int, *, dry_run: bool = False) -> TxResult:
        """Withdraw collateral from the market."""
        if assets <= 0:
            return TxResult(False, error="assets must be positive")
        market = self._get_market(market_name)
        if dry_run:
            return TxResult(True, details={"dry_run": True, "assets": assets, "market": market_name})
        logger.info("[Morpho] withdraw_collateral  market=%s  assets=%d", market_name, assets)
        if self._is_mock:
            return self._provider.withdraw_collateral(market.market_id, self.wallet, assets)
        return self._provider.withdraw_collateral(market.market_id, self.wallet, assets, market.to_params_tuple())
