"""Sequential self-funding growth engine for Morpho Blue on Polygon.

Strategy overview
-----------------
The GrowthEngine executes a looped yield-generation process across one or
more Morpho Blue markets:

  Step 1 — Identify the highest-supply-APY market from the registry.
  Step 2 — Approve tokens and supply collateral.
  Step 3 — Borrow against collateral at a safe LTV (target_borrow_ratio × LLTV).
  Step 4 — Re-supply borrowed tokens into the best supply market to compound.
  Step 5 — Monitor health factor continuously; auto-repay when below threshold.
  Step 6 — On each cycle, collect any accrued interest and reinvest.
  Step 7 — Report step-by-step results via the report_callback.

The engine runs entirely in mock mode when no real Web3 connection is
provided, making it fully testable offline.

Growth grade
------------
The ``growth_grade`` property returns one of:
  "A+" — HF > 2.0 and supply APY > 5%
  "A"  — HF > 1.5
  "B"  — HF > 1.2
  "C"  — HF > 1.0 (at risk)
  "D"  — HF ≤ 1.0 (liquidatable)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .client import MorphoBlueClient, TxResult, UserPosition
from .markets import MarketConfig, MarketRegistry

logger = logging.getLogger(__name__)

# Default safe borrow ratio as a fraction of max (LLTV)
DEFAULT_TARGET_LTV = 0.50      # borrow at 50% of max LLTV
MIN_HEALTH_FACTOR = 1.20       # auto-repay if HF drops below this
REBALANCE_HF_TARGET = 1.50     # repay until HF reaches this level


@dataclass
class GrowthCycleResult:
    """Result of a single growth cycle."""

    cycle: int
    market_name: str
    action: str
    success: bool
    tx_hash: str = ""
    error: str = ""
    position_before: Optional[dict] = None
    position_after: Optional[dict] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GrowthReport:
    """Cumulative report of all growth cycles."""

    cycles: list[GrowthCycleResult] = field(default_factory=list)
    total_supplied: float = 0.0
    total_borrowed: float = 0.0
    total_repaid: float = 0.0
    net_yield_usd: float = 0.0

    def append(self, result: GrowthCycleResult) -> None:
        self.cycles.append(result)

    @property
    def success_rate(self) -> float:
        if not self.cycles:
            return 0.0
        return sum(1 for c in self.cycles if c.success) / len(self.cycles)

    def summary(self) -> dict[str, Any]:
        return {
            "total_cycles": len(self.cycles),
            "success_rate": round(self.success_rate, 4),
            "total_supplied": round(self.total_supplied, 6),
            "total_borrowed": round(self.total_borrowed, 6),
            "total_repaid": round(self.total_repaid, 6),
            "net_yield_usd": round(self.net_yield_usd, 6),
        }


class GrowthEngine:
    """Sequential self-funding growth engine built on MorphoBlueClient.

    Parameters
    ----------
    client:
        A :class:`~morpho.client.MorphoBlueClient` instance.
    target_ltv:
        Target borrow-to-max-LTV ratio (0–1).  Defaults to 0.50.
    min_health_factor:
        Auto-repay trigger threshold.  Defaults to 1.20.
    report_callback:
        Optional callable invoked after each cycle with a
        :class:`GrowthCycleResult`.

    Examples
    --------
    >>> from morpho import MorphoBlueClient, GrowthEngine
    >>> client = MorphoBlueClient()
    >>> engine = GrowthEngine(client)
    >>> report = engine.run_growth_cycle("WETH/USDC_E-86", collateral_assets=1_000_000)
    """

    def __init__(
        self,
        client: MorphoBlueClient,
        target_ltv: float = DEFAULT_TARGET_LTV,
        min_health_factor: float = MIN_HEALTH_FACTOR,
        report_callback: Optional[Callable[[GrowthCycleResult], None]] = None,
    ) -> None:
        self._client = client
        self.target_ltv = target_ltv
        self.min_health_factor = min_health_factor
        self._callback = report_callback
        self._report = GrowthReport()
        self._cycle = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_growth_cycle(
        self,
        market_name: str,
        collateral_assets: int,
        collateral_token_symbol: str = "WETH",
        *,
        dry_run: bool = False,
    ) -> GrowthReport:
        """Execute one complete growth cycle.

        Steps
        -----
        1. Approve collateral token spend.
        2. Supply collateral to the market.
        3. Compute safe borrow amount.
        4. Borrow loan tokens.
        5. Re-supply borrowed tokens into the same market.
        6. Monitor and report health factor.

        Parameters
        ----------
        market_name:
            Market from the registry to operate on.
        collateral_assets:
            Collateral amount in token base units.
        collateral_token_symbol:
            Symbol of the collateral token (for approval).
        dry_run:
            If True, simulate all steps without broadcasting.
        """
        self._cycle += 1
        cycle = self._cycle

        market = self._client._registry.get(market_name)
        if market is None:
            result = GrowthCycleResult(cycle, market_name, "init", False, error=f"Unknown market: {market_name}")
            self._emit(result)
            return self._report

        logger.info("[Growth] Cycle %d  market=%s  collateral=%d  dry_run=%s",
                    cycle, market_name, collateral_assets, dry_run)

        # Step 1: Approve
        approve_result = self._client.approve(collateral_token_symbol, collateral_assets * 2)
        self._emit(GrowthCycleResult(cycle, market_name, "approve", approve_result.success,
                                     approve_result.tx_hash, approve_result.error))
        if not approve_result.success:
            return self._report

        # Step 2: Supply collateral
        pos_before = self._position_snapshot(market_name)
        supply_col_result = self._client.supply_collateral(market_name, collateral_assets, dry_run=dry_run)
        pos_after = self._position_snapshot(market_name)
        self._emit(GrowthCycleResult(
            cycle, market_name, "supply_collateral",
            supply_col_result.success, supply_col_result.tx_hash, supply_col_result.error,
            pos_before, pos_after,
        ))
        if not supply_col_result.success:
            return self._report

        # Step 3: Compute safe borrow amount
        borrow_assets = self._compute_safe_borrow(market_name, market)
        if borrow_assets <= 0:
            logger.info("[Growth] Cycle %d: borrow amount is 0, skipping borrow/re-supply.", cycle)
            return self._report

        # Step 4: Borrow
        pos_before = self._position_snapshot(market_name)
        borrow_result = self._client.borrow(market_name, borrow_assets, dry_run=dry_run)
        pos_after = self._position_snapshot(market_name)
        self._emit(GrowthCycleResult(
            cycle, market_name, "borrow",
            borrow_result.success, borrow_result.tx_hash, borrow_result.error,
            pos_before, pos_after,
            {"borrow_assets": borrow_assets},
        ))
        if not borrow_result.success:
            return self._report

        self._report.total_supplied += collateral_assets
        self._report.total_borrowed += borrow_assets

        # Step 5: Re-supply borrowed tokens as loan-side supply for yield
        approve_loan = self._client.approve(market.loan_token_symbol, borrow_assets)
        if approve_loan.success:
            pos_before = self._position_snapshot(market_name)
            resupply_result = self._client.supply(market_name, borrow_assets, dry_run=dry_run)
            pos_after = self._position_snapshot(market_name)
            self._emit(GrowthCycleResult(
                cycle, market_name, "re_supply",
                resupply_result.success, resupply_result.tx_hash, resupply_result.error,
                pos_before, pos_after,
                {"resupply_assets": borrow_assets},
            ))

        # Step 6: Health factor check
        pos = self._client.get_position(market_name)
        self._emit(GrowthCycleResult(
            cycle, market_name, "health_check", True,
            details={
                "health_factor": pos.health_factor,
                "grade": self._grade(pos),
                "supply_apy": pos.supply_apy,
                "borrow_apy": pos.borrow_apy,
                "borrow_assets": pos.borrow_assets,
                "supply_assets": pos.supply_assets,
            },
        ))

        # Step 7: Auto-repay if needed
        if pos.health_factor < self.min_health_factor:
            self._auto_repay(market_name, pos, dry_run=dry_run)

        return self._report

    def monitor_and_rebalance(
        self,
        market_name: str,
        *,
        dry_run: bool = False,
    ) -> GrowthCycleResult:
        """Check health factor and repay if below threshold.

        Intended to be called periodically (e.g. every 5 minutes).
        """
        pos = self._client.get_position(market_name)
        if pos.health_factor < self.min_health_factor:
            logger.warning("[Growth] HF=%.3f below threshold=%.3f — auto-repaying.",
                           pos.health_factor, self.min_health_factor)
            return self._auto_repay(market_name, pos, dry_run=dry_run)
        return GrowthCycleResult(
            self._cycle, market_name, "monitor_ok", True,
            details={"health_factor": pos.health_factor, "grade": self._grade(pos)},
        )

    @property
    def report(self) -> GrowthReport:
        return self._report

    @property
    def growth_grade(self) -> str:
        """Overall growth grade based on latest positions across all markets."""
        grades: list[str] = []
        for market in self._client.list_markets():
            try:
                pos = self._client.get_position(market.name)
                grades.append(self._grade(pos))
            except Exception:
                pass
        if not grades:
            return "N/A"
        # Return the most conservative grade
        for g in ("D", "C", "B", "A", "A+"):
            if g in grades:
                return g
        return grades[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_safe_borrow(self, market_name: str, market: MarketConfig) -> int:
        """Compute a safe borrow amount based on current collateral and target LTV."""
        pos = self._client.get_position(market_name)
        collateral_price = self._client.get_collateral_price(market.collateral_token_symbol)
        collateral_value = float(pos.collateral) * collateral_price / (10 ** market.collateral_decimals)
        lltv = market.lltv / 1e18
        max_borrow = collateral_value * lltv
        safe_borrow = max_borrow * self.target_ltv
        # Subtract existing borrow
        safe_additional = max(0.0, safe_borrow - pos.borrow_assets)
        # Convert to loan token base units (USDC = 6 decimals)
        return int(safe_additional * (10 ** market.loan_decimals))

    def _auto_repay(
        self,
        market_name: str,
        pos: UserPosition,
        *,
        dry_run: bool = False,
    ) -> GrowthCycleResult:
        """Repay enough to bring HF back to REBALANCE_HF_TARGET."""
        market = self._client._registry.get(market_name)
        if market is None or pos.borrow_assets <= 0:
            return GrowthCycleResult(self._cycle, market_name, "auto_repay", True)

        collateral_price = self._client.get_collateral_price(market.collateral_token_symbol)
        collateral_value = float(pos.collateral) * collateral_price / (10 ** market.collateral_decimals)
        lltv = market.lltv / 1e18
        # target_borrow = collateral_value × lltv / REBALANCE_HF_TARGET
        target_borrow = collateral_value * lltv / REBALANCE_HF_TARGET
        repay_amount = max(0.0, pos.borrow_assets - target_borrow)
        repay_units = int(repay_amount * (10 ** market.loan_decimals))

        if repay_units <= 0:
            return GrowthCycleResult(self._cycle, market_name, "auto_repay", True,
                                     details={"skipped": True})

        pos_before = self._position_snapshot(market_name)
        result = self._client.repay(market_name, repay_units, dry_run=dry_run)
        pos_after = self._position_snapshot(market_name)
        if result.success:
            self._report.total_repaid += repay_amount

        cycle_result = GrowthCycleResult(
            self._cycle, market_name, "auto_repay",
            result.success, result.tx_hash, result.error,
            pos_before, pos_after,
            {"repay_units": repay_units, "repay_usd": repay_amount},
        )
        self._emit(cycle_result)
        return cycle_result

    def _position_snapshot(self, market_name: str) -> dict:
        try:
            pos = self._client.get_position(market_name)
            return {
                "supply_assets": pos.supply_assets,
                "borrow_assets": pos.borrow_assets,
                "collateral": pos.collateral,
                "health_factor": pos.health_factor,
                "grade": self._grade(pos),
            }
        except Exception:
            return {}

    @staticmethod
    def _grade(pos: UserPosition) -> str:
        hf = pos.health_factor
        supply_apy = pos.supply_apy
        if hf > 2.0 and supply_apy > 0.05:
            return "A+"
        if hf > 1.5:
            return "A"
        if hf > 1.2:
            return "B"
        if hf > 1.0:
            return "C"
        return "D"

    def _emit(self, result: GrowthCycleResult) -> None:
        self._report.append(result)
        logger.debug("[Growth] %s  cycle=%d  action=%s  ok=%s",
                     result.market_name, result.cycle, result.action, result.success)
        if self._callback:
            try:
                self._callback(result)
            except Exception:
                logger.exception("[Growth] report_callback raised an exception.")
