"""Morpho Protocol client for Polygon.

Provides MorphoClient with supply, borrow, repay, withdraw, and
collateral swap operations. Uses Web3.py when available; falls back
to a mock provider for offline use / testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Morpho Blue on Polygon (mainnet)
MORPHO_BLUE_ADDRESS = "0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb"

# Market IDs (example placeholders — real IDs are keccak256 hashes)
MARKET_WMATIC_USDC = "0x" + "a1" * 32
MARKET_WETH_USDC = "0x" + "b2" * 32
MARKET_WBTC_USDC = "0x" + "c3" * 32

KNOWN_MARKETS: Dict[str, str] = {
    "WMATIC/USDC": MARKET_WMATIC_USDC,
    "WETH/USDC": MARKET_WETH_USDC,
    "WBTC/USDC": MARKET_WBTC_USDC,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MarketParams:
    loan_token: str
    collateral_token: str
    oracle: str
    irm: str
    lltv: float  # liquidation LTV, e.g. 0.86


@dataclass
class Position:
    market_id: str
    supply_shares: float = 0.0
    borrow_shares: float = 0.0
    collateral: float = 0.0
    # Derived fields (set by client)
    health_factor: float = float("inf")
    supply_apy: float = 0.0
    borrow_apy: float = 0.0

    def is_liquidatable(self) -> bool:
        return self.health_factor < 1.0


@dataclass
class TxResult:
    success: bool
    tx_hash: str = ""
    gas_used: int = 0
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mock Web3 Provider (for testing / offline use)
# ---------------------------------------------------------------------------

class MockWeb3Provider:
    """Simulates on-chain interactions without a real RPC."""

    def __init__(self) -> None:
        self._supply: Dict[Tuple[str, str], float] = {}
        self._borrow: Dict[Tuple[str, str], float] = {}
        self._collateral: Dict[Tuple[str, str], float] = {}
        self._prices: Dict[str, float] = {
            "WMATIC": 0.85,
            "WETH": 3200.0,
            "WBTC": 65000.0,
            "USDC": 1.0,
        }

    def supply(self, market_id: str, wallet: str, amount: float) -> TxResult:
        key = (market_id, wallet)
        self._supply[key] = self._supply.get(key, 0.0) + amount
        return TxResult(success=True, tx_hash=f"0x{'aa' * 32}", gas_used=120_000)

    def borrow(self, market_id: str, wallet: str, amount: float) -> TxResult:
        key = (market_id, wallet)
        self._borrow[key] = self._borrow.get(key, 0.0) + amount
        return TxResult(success=True, tx_hash=f"0x{'bb' * 32}", gas_used=150_000)

    def repay(self, market_id: str, wallet: str, amount: float) -> TxResult:
        key = (market_id, wallet)
        outstanding = self._borrow.get(key, 0.0)
        repaid = min(amount, outstanding)
        self._borrow[key] = outstanding - repaid
        return TxResult(
            success=True, tx_hash=f"0x{'cc' * 32}", gas_used=100_000,
            details={"repaid": repaid}
        )

    def withdraw(self, market_id: str, wallet: str, amount: float) -> TxResult:
        key = (market_id, wallet)
        available = self._supply.get(key, 0.0)
        withdrawn = min(amount, available)
        self._supply[key] = available - withdrawn
        return TxResult(
            success=True, tx_hash=f"0x{'dd' * 32}", gas_used=110_000,
            details={"withdrawn": withdrawn}
        )

    def deposit_collateral(self, market_id: str, wallet: str, amount: float) -> TxResult:
        key = (market_id, wallet)
        self._collateral[key] = self._collateral.get(key, 0.0) + amount
        return TxResult(success=True, tx_hash=f"0x{'ee' * 32}", gas_used=90_000)

    def get_position(self, market_id: str, wallet: str) -> Position:
        key = (market_id, wallet)
        supply = self._supply.get(key, 0.0)
        borrow = self._borrow.get(key, 0.0)
        collateral = self._collateral.get(key, 0.0)
        # Simple HF: collateral / borrow (avoid div-by-zero)
        hf = (collateral / borrow) if borrow > 0 else float("inf")
        return Position(
            market_id=market_id,
            supply_shares=supply,
            borrow_shares=borrow,
            collateral=collateral,
            health_factor=hf,
            supply_apy=0.035,
            borrow_apy=0.055,
        )

    def get_price(self, token: str) -> float:
        return self._prices.get(token, 1.0)

    def set_price(self, token: str, price: float) -> None:
        self._prices[token] = price


# ---------------------------------------------------------------------------
# Morpho Client
# ---------------------------------------------------------------------------

class MorphoClient:
    """High-level client for Morpho Blue on Polygon.

    When *provider* is None, a :class:`MockWeb3Provider` is used.
    Pass a real ``web3.Web3`` instance (with Morpho Blue ABI) for
    production use.
    """

    def __init__(
        self,
        wallet_address: str = "0x" + "00" * 20,
        provider: Optional[Any] = None,
    ) -> None:
        self.wallet = wallet_address
        self._provider: Any = provider or MockWeb3Provider()
        self._is_mock = provider is None
        logger.info(
            "MorphoClient initialised (wallet=%s, mock=%s)",
            wallet_address[:10] + "…",
            self._is_mock,
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def supply(self, market_id: str, amount: float) -> TxResult:
        """Supply *amount* of loan token to the given market."""
        if amount <= 0:
            return TxResult(success=False, error="Amount must be positive.")
        logger.info("[Morpho] Supply market=%s amount=%.4f", market_id[:10], amount)
        result = self._provider.supply(market_id, self.wallet, amount)
        if result.success:
            logger.info("[Morpho] Supply successful tx=%s", result.tx_hash[:10])
        return result

    def borrow(self, market_id: str, amount: float) -> TxResult:
        """Borrow *amount* from the given market (collateral must be deposited)."""
        if amount <= 0:
            return TxResult(success=False, error="Amount must be positive.")
        pos = self.get_position(market_id)
        if pos.collateral <= 0:
            return TxResult(success=False, error="No collateral deposited.")
        logger.info("[Morpho] Borrow market=%s amount=%.4f", market_id[:10], amount)
        result = self._provider.borrow(market_id, self.wallet, amount)
        return result

    def repay(self, market_id: str, amount: Optional[float] = None) -> TxResult:
        """Repay *amount* of borrowed tokens (or full balance if None)."""
        pos = self.get_position(market_id)
        if amount is None:
            amount = pos.borrow_shares
        if amount <= 0:
            return TxResult(success=False, error="Nothing to repay.")
        logger.info("[Morpho] Repay market=%s amount=%.4f", market_id[:10], amount)
        return self._provider.repay(market_id, self.wallet, amount)

    def withdraw(self, market_id: str, amount: float) -> TxResult:
        """Withdraw *amount* of supplied tokens from the market."""
        if amount <= 0:
            return TxResult(success=False, error="Amount must be positive.")
        logger.info("[Morpho] Withdraw market=%s amount=%.4f", market_id[:10], amount)
        return self._provider.withdraw(market_id, self.wallet, amount)

    def deposit_collateral(self, market_id: str, amount: float) -> TxResult:
        """Deposit collateral into the given market."""
        if amount <= 0:
            return TxResult(success=False, error="Amount must be positive.")
        logger.info("[Morpho] DepositCollateral market=%s amount=%.4f", market_id[:10], amount)
        return self._provider.deposit_collateral(market_id, self.wallet, amount)

    # ------------------------------------------------------------------
    # Collateral swap
    # ------------------------------------------------------------------

    def collateral_swap(
        self,
        market_id: str,
        swap_amount: float,
        min_received: float = 0.0,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Reduce borrowed amount via a collateral swap.

        Flow
        ----
        1. Fetch current position.
        2. Simulate swap: *swap_amount* of collateral → loan token.
        3. Use received loan tokens to repay part of the borrow.
        4. If *dry_run* is False, execute on-chain.

        Returns a summary dict with before/after position details.
        """
        pos_before = self.get_position(market_id)
        if pos_before.collateral < swap_amount:
            return {
                "success": False,
                "error": f"Insufficient collateral: have {pos_before.collateral}, need {swap_amount}",
            }

        # Simulate swap (using mock 1:1 ratio for simplicity)
        received = swap_amount * 0.995  # 0.5% slippage
        if received < min_received:
            return {
                "success": False,
                "error": f"Slippage too high: received {received:.4f} < min {min_received}",
            }

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "swap_amount": swap_amount,
                "received": received,
                "repay_amount": min(received, pos_before.borrow_shares),
                "projected_borrow_after": max(0.0, pos_before.borrow_shares - received),
            }

        # Step 1: withdraw collateral
        withdraw_result = self._provider.withdraw(market_id, self.wallet, swap_amount)
        if not withdraw_result.success:
            return {"success": False, "error": f"Withdraw failed: {withdraw_result.error}"}

        # Step 2: repay with received tokens
        repay_amount = min(received, pos_before.borrow_shares)
        repay_result = self._provider.repay(market_id, self.wallet, repay_amount)
        if not repay_result.success:
            return {"success": False, "error": f"Repay failed: {repay_result.error}"}

        pos_after = self.get_position(market_id)
        logger.info(
            "[Morpho] CollateralSwap complete: borrow %.4f → %.4f, HF %.2f → %.2f",
            pos_before.borrow_shares,
            pos_after.borrow_shares,
            pos_before.health_factor,
            pos_after.health_factor,
        )
        return {
            "success": True,
            "swap_amount": swap_amount,
            "received": received,
            "repaid": repay_amount,
            "pos_before": {
                "collateral": pos_before.collateral,
                "borrow": pos_before.borrow_shares,
                "health_factor": pos_before.health_factor,
            },
            "pos_after": {
                "collateral": pos_after.collateral,
                "borrow": pos_after.borrow_shares,
                "health_factor": pos_after.health_factor,
            },
        }

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_position(self, market_id: str) -> Position:
        return self._provider.get_position(market_id, self.wallet)

    def get_all_positions(self) -> List[Position]:
        return [self.get_position(mid) for mid in KNOWN_MARKETS.values()]

    def health_factor(self, market_id: str) -> float:
        return self.get_position(market_id).health_factor

    def liquidation_price(self, market_id: str) -> float:
        """Approximate price at which position becomes liquidatable."""
        pos = self.get_position(market_id)
        if pos.collateral <= 0:
            return 0.0
        # liquidation_price ≈ borrow / (collateral × LLTV)
        lltv = 0.86  # default
        return pos.borrow_shares / (pos.collateral * lltv)

    def market_apy(self, market_id: str) -> Dict[str, float]:
        pos = self.get_position(market_id)
        return {"supply_apy": pos.supply_apy, "borrow_apy": pos.borrow_apy}

    def list_markets(self) -> Dict[str, str]:
        return dict(KNOWN_MARKETS)
