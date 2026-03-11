"""DeFi account growth strategy — the 1/2=3 methodology.

Strategy overview
-----------------
The "1/2=3" concept:

  1 (supply asset)  = Supply ETH as collateral — earns supply APY while
                      collateral value appreciates in a bull run.
  ÷ 2 (borrow half) = Borrow USDC at ~50% of max LTV — interest paid is
                      covered by supply APY + trading returns.
  = 3 (three gains) = Net result: ETH appreciation + USDC trading profit +
                      DeFi supply yield − borrow cost = compounded growth.

Health-factor framework
-----------------------
Four operating zones based on Morpho's liquidation mechanics:

  Zone  Label     HF range      Action
  ----  -----     --------      ------
  A+    Safe      HF > 2.0      Normal operation; may increase position
  A     Stable    1.5–2.0       Normal operation; hold current position
  B     Watch     1.2–1.5       Reduce borrow or add collateral
  C     Danger    1.0–1.2       Auto-repay to reach HF 1.5
  D     Critical  HF ≤ 1.0      Emergency full repay; stop borrowing

Bear/Bull market entry/exit signals
------------------------------------
  - Bear market ending → enter: supply ETH collateral, borrow USDC
    (ETH cheap → high collateral upside; borrow cheap while waiting)
  - Bull market ATH approached → exit or reduce: repay USDC, withdraw ETH

Usage
-----
::

    from app.evaluation.defi_strategy import HalfHalfThreeStrategy
    from app.defi.morpho import MorphoClient

    client = MorphoClient()
    strategy = HalfHalfThreeStrategy(client, market_id="...", symbol="WETH")
    result = strategy.enter(collateral_amount=10.0)
    print(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Health-factor zone constants (aligned with morpho/growth.py)
# ---------------------------------------------------------------------------

HF_SAFE = 2.0        # HF above this → safe to increase position
HF_STABLE = 1.5      # HF above this → hold; target for auto-repay
HF_WATCH = 1.2       # HF below this → start reducing borrow
HF_DANGER = 1.0      # HF below this → emergency full repay

# Borrow utilisation as a fraction of max LTV when entering a position
DEFAULT_BORROW_RATIO = 0.50   # borrow at 50% of LLTV (the "÷2" in 1/2=3)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PositionStatus:
    """Snapshot of a single DeFi position."""

    market_id: str
    collateral: float           # collateral amount in token units
    collateral_usd: float       # collateral USD value
    borrow_usd: float           # outstanding loan in USD
    supply_apy: float           # annualised supply APY
    borrow_apy: float           # annualised borrow APY
    health_factor: float
    net_apy: float              # supply_apy × collateral − borrow_apy × borrow
    zone: str                   # "A+", "A", "B", "C", "D"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "collateral": round(self.collateral, 6),
            "collateral_usd": round(self.collateral_usd, 2),
            "borrow_usd": round(self.borrow_usd, 2),
            "supply_apy_pct": round(self.supply_apy * 100, 2),
            "borrow_apy_pct": round(self.borrow_apy * 100, 2),
            "health_factor": round(self.health_factor, 4),
            "net_apy_pct": round(self.net_apy * 100, 2),
            "zone": self.zone,
        }


@dataclass
class StrategyAction:
    """Records an action taken by the strategy."""

    action: str                 # "enter", "rebalance", "exit", "monitor_ok"
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


# ---------------------------------------------------------------------------
# Health-factor manager
# ---------------------------------------------------------------------------

class HealthFactorManager:
    """Evaluate the current health-factor zone and recommend actions.

    Parameters
    ----------
    safe_hf:
        HF threshold for fully safe operation.
    stable_hf:
        Target HF after rebalancing.
    watch_hf:
        HF below which position should be watched and possibly reduced.
    danger_hf:
        HF below which emergency repay is triggered.
    """

    def __init__(
        self,
        safe_hf: float = HF_SAFE,
        stable_hf: float = HF_STABLE,
        watch_hf: float = HF_WATCH,
        danger_hf: float = HF_DANGER,
    ) -> None:
        self.safe_hf = safe_hf
        self.stable_hf = stable_hf
        self.watch_hf = watch_hf
        self.danger_hf = danger_hf

    def zone(self, hf: float) -> str:
        """Return the zone string for the given health factor."""
        if hf > self.safe_hf:
            return "A+"
        if hf > self.stable_hf:
            return "A"
        if hf > self.watch_hf:
            return "B"
        if hf > self.danger_hf:
            return "C"
        return "D"

    def requires_action(self, hf: float) -> bool:
        """Return True if any rebalancing action is required."""
        return hf < self.watch_hf

    def repay_fraction(self, hf: float) -> float:
        """Fraction of outstanding borrow to repay to reach ``stable_hf``.

        Returns 0.0 when no repay is needed.

        Derivation
        ----------
        HF_new = (collateral_value × LLTV) / (borrow × (1 − f))
        Solving for f to achieve HF_new = stable_hf:
          f = 1 − (collateral_value × LLTV) / (borrow × stable_hf)
          f = 1 − HF_current / stable_hf
        """
        if hf >= self.stable_hf:
            return 0.0
        f = 1.0 - hf / self.stable_hf
        return float(max(0.0, min(1.0, f)))

    def evaluate(self, hf: float) -> Dict[str, Any]:
        """Return a full evaluation dict for a given health factor."""
        return {
            "health_factor": round(hf, 4),
            "zone": self.zone(hf),
            "requires_action": self.requires_action(hf),
            "repay_fraction": round(self.repay_fraction(hf), 4),
            "safe_to_increase": hf > self.safe_hf,
        }


# ---------------------------------------------------------------------------
# 1/2=3 strategy
# ---------------------------------------------------------------------------

class HalfHalfThreeStrategy:
    """Implement the 1/2=3 DeFi account growth methodology.

    The strategy manages a single Morpho Blue position:
      - **Supply** ETH (asset1) as collateral
      - **Borrow** USDC (asset2) at *borrow_ratio* × max LTV
      - **Trade** with borrowed USDC to earn additional returns
      - **Repay** automatically when health factor falls below threshold
      - **Compound** DeFi yield back into the position

    Parameters
    ----------
    morpho_client:
        A :class:`~app.defi.morpho.MorphoClient` instance (or any object
        implementing the same interface).
    market_id:
        The Morpho Blue market ID to operate on.
    symbol:
        Collateral token symbol (e.g. ``"WETH"``).
    collateral_price:
        Current USD price of the collateral token.  Used for position
        sizing calculations.
    borrow_ratio:
        Target borrow as a fraction of max LTV (default 0.50 = 50%).
    lltv:
        Liquidation loan-to-value ratio for the market (default 0.86).
    report_callback:
        Optional callable invoked after each strategy action with a
        :class:`StrategyAction`.
    """

    def __init__(
        self,
        morpho_client: Any,
        market_id: str,
        symbol: str = "WETH",
        collateral_price: float = 3200.0,
        borrow_ratio: float = DEFAULT_BORROW_RATIO,
        lltv: float = 0.86,
        report_callback: Optional[Callable[[StrategyAction], None]] = None,
    ) -> None:
        self._client = morpho_client
        self.market_id = market_id
        self.symbol = symbol
        self.collateral_price = collateral_price
        self.borrow_ratio = borrow_ratio
        self.lltv = lltv
        self._hf_manager = HealthFactorManager()
        self._callback = report_callback
        self._history: List[StrategyAction] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enter(self, collateral_amount: float) -> StrategyAction:
        """Open or increase a collateralised borrow position.

        Steps
        -----
        1. Deposit *collateral_amount* of ETH as collateral.
        2. Compute the safe borrow amount: collateral_usd × LLTV × borrow_ratio.
        3. Borrow the calculated USDC amount.

        Parameters
        ----------
        collateral_amount:
            Number of ETH tokens to deposit as collateral.
        """
        collateral_usd = collateral_amount * self.collateral_price
        borrow_amount = collateral_usd * self.lltv * self.borrow_ratio

        result_col = self._client.deposit_collateral(self.market_id, collateral_amount)
        if not result_col.success:
            action = StrategyAction("enter", False, error=result_col.error)
            self._emit(action)
            return action

        result_borrow = self._client.borrow(self.market_id, borrow_amount)
        if not result_borrow.success:
            action = StrategyAction("enter", False, error=result_borrow.error)
            self._emit(action)
            return action

        pos = self._get_position()
        action = StrategyAction(
            "enter",
            True,
            details={
                "collateral_deposited": collateral_amount,
                "collateral_usd": round(collateral_usd, 2),
                "borrowed_usdc": round(borrow_amount, 2),
                "health_factor": round(pos.health_factor if pos else 0.0, 4),
                "zone": self._hf_manager.zone(pos.health_factor if pos else 0.0),
                "tx_collateral": result_col.tx_hash,
                "tx_borrow": result_borrow.tx_hash,
            },
        )
        self._emit(action)
        return action

    def rebalance(self) -> StrategyAction:
        """Check health factor and repay if below watch threshold.

        This method implements the auto-repay loop that keeps the account
        safe while still earning DeFi yield.
        """
        pos = self._get_position()
        if pos is None:
            action = StrategyAction("rebalance", False, error="Could not fetch position.")
            self._emit(action)
            return action

        hf = pos.health_factor
        evaluation = self._hf_manager.evaluate(hf)

        if not evaluation["requires_action"]:
            action = StrategyAction(
                "monitor_ok", True,
                details={**evaluation, "position": pos.to_dict()},
            )
            self._emit(action)
            return action

        # Compute repay amount
        repay_fraction = evaluation["repay_fraction"]
        repay_amount = pos.borrow_usd * repay_fraction

        result = self._client.repay(self.market_id, repay_amount)
        pos_after = self._get_position()

        action = StrategyAction(
            "rebalance",
            result.success,
            details={
                "hf_before": round(hf, 4),
                "hf_after": round(pos_after.health_factor if pos_after else hf, 4),
                "zone_before": evaluation["zone"],
                "zone_after": self._hf_manager.zone(
                    pos_after.health_factor if pos_after else hf
                ),
                "repaid_usdc": round(repay_amount, 2),
                "repay_fraction": round(repay_fraction, 4),
                "tx_hash": result.tx_hash,
            },
            error=result.error if not result.success else "",
        )
        self._emit(action)
        return action

    def exit(self, full: bool = True) -> StrategyAction:
        """Close the position by repaying the borrow and withdrawing collateral.

        Parameters
        ----------
        full:
            If True, repay the entire outstanding loan and withdraw all
            collateral.  If False, repay only the amount needed to reach
            HF_SAFE.
        """
        pos = self._get_position()
        if pos is None:
            action = StrategyAction("exit", False, error="Could not fetch position.")
            self._emit(action)
            return action

        if full:
            repay_amount = pos.borrow_usd
        else:
            repay_fraction = self._hf_manager.repay_fraction(pos.health_factor)
            repay_amount = pos.borrow_usd * repay_fraction

        errors: List[str] = []
        if repay_amount > 0:
            result_repay = self._client.repay(self.market_id, repay_amount)
            if not result_repay.success:
                errors.append(result_repay.error)

        if full:
            result_withdraw = self._client.withdraw(self.market_id, pos.collateral)
            if not result_withdraw.success:
                errors.append(result_withdraw.error)

        action = StrategyAction(
            "exit",
            len(errors) == 0,
            details={
                "repaid_usdc": round(repay_amount, 2),
                "collateral_withdrawn": pos.collateral if full else 0.0,
                "full_exit": full,
            },
            error="; ".join(errors),
        )
        self._emit(action)
        return action

    def position_status(self) -> Optional[PositionStatus]:
        """Return the current position snapshot."""
        return self._get_position()

    def history(self) -> List[StrategyAction]:
        """Return all strategy actions taken so far."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_position(self) -> Optional[PositionStatus]:
        try:
            pos = self._client.get_position(self.market_id)
            collateral_usd = float(pos.collateral) * self.collateral_price
            borrow_usd = float(pos.borrow_shares)
            supply_apy = float(pos.supply_apy)
            borrow_apy = float(pos.borrow_apy)
            hf = float(pos.health_factor)
            net_apy = supply_apy * collateral_usd - borrow_apy * borrow_usd
            return PositionStatus(
                market_id=self.market_id,
                collateral=float(pos.collateral),
                collateral_usd=collateral_usd,
                borrow_usd=borrow_usd,
                supply_apy=supply_apy,
                borrow_apy=borrow_apy,
                health_factor=hf,
                net_apy=net_apy,
                zone=self._hf_manager.zone(hf),
            )
        except Exception as exc:
            logger.warning("[HalfHalfThree] Could not fetch position: %s", exc)
            return None

    def _emit(self, action: StrategyAction) -> None:
        self._history.append(action)
        level = logging.INFO if action.success else logging.WARNING
        logger.log(
            level,
            "[HalfHalfThree] action=%s success=%s  %s",
            action.action, action.success,
            action.details if action.success else action.error,
        )
        if self._callback:
            try:
                self._callback(action)
            except Exception:
                logger.exception("[HalfHalfThree] report_callback raised.")


# ---------------------------------------------------------------------------
# Bear/bull market entry signal
# ---------------------------------------------------------------------------

class MarketEntryAdvisor:
    """Advise on DeFi position entry/exit based on market regime.

    This implements the timing logic for the 1/2=3 strategy:

    - **Enter** (supply ETH + borrow USDC) when bearish market is ending or
      the market is at cycle lows (RSI < 35 on daily timeframe).
    - **Increase** when the bull run is confirmed but not yet at extreme
      overbought levels.
    - **Reduce/Exit** when ATH territory is reached (RSI > 80 on weekly or
      extreme greed signals).

    Parameters
    ----------
    bear_rsi_threshold:
        RSI below which to trigger entry.  Default 35 (end of bear market).
    bull_exit_rsi_threshold:
        RSI above which to trigger exit.  Default 80 (ATH territory).
    """

    def __init__(
        self,
        bear_rsi_threshold: float = 35.0,
        bull_exit_rsi_threshold: float = 80.0,
    ) -> None:
        self.bear_rsi_threshold = bear_rsi_threshold
        self.bull_exit_rsi_threshold = bull_exit_rsi_threshold

    def advise(self, closes: np.ndarray, rsi_period: int = 14) -> str:
        """Return an advice string for the given price series.

        Parameters
        ----------
        closes:
            Array of closing prices in ascending order.
        rsi_period:
            RSI look-back period.

        Returns
        -------
        str
            One of ``"enter"``, ``"hold"``, ``"reduce"``, ``"exit"``,
            ``"insufficient_data"``.
        """
        from app.trading.algorithms import _rsi

        if len(closes) < rsi_period + 5:
            return "insufficient_data"

        rsi_arr = _rsi(closes, rsi_period)
        last_rsi = rsi_arr[-1]

        if np.isnan(last_rsi):
            return "insufficient_data"

        if last_rsi < self.bear_rsi_threshold:
            return "enter"
        if last_rsi > self.bull_exit_rsi_threshold:
            return "exit"

        # Intermediate zone: look at price trend using simple moving averages
        sma_short = float(np.mean(closes[-20:]))
        sma_long = float(np.mean(closes[-50:])) if len(closes) >= 50 else sma_short

        if sma_short > sma_long:
            return "hold"
        return "reduce"

    def advise_multi_timeframe(
        self, tf_closes: Dict[str, np.ndarray]
    ) -> Dict[str, str]:
        """Return advice for each timeframe and an overall recommendation.

        Parameters
        ----------
        tf_closes:
            Mapping from timeframe label to closing price array.

        Returns
        -------
        dict
            Per-timeframe advice plus ``"overall"`` key.
        """
        result: Dict[str, str] = {}
        for tf, closes in tf_closes.items():
            result[tf] = self.advise(closes)

        # Overall: majority vote weighted by timeframe precedence
        tf_weights: Dict[str, float] = {"1d": 3.0, "4h": 2.0, "1h": 1.0}
        vote: Dict[str, float] = {}
        for tf, advice in result.items():
            w = tf_weights.get(tf, 1.0)
            vote[advice] = vote.get(advice, 0.0) + w

        result["overall"] = max(vote, key=vote.__getitem__) if vote else "hold"
        return result
