"""Morpho rewards and net APR calculations.

Provides helpers to estimate:

- Morpho Blue supply/borrow reward points accrual
- Net supply APR = base supply APY + reward token APY equivalent
- Net borrow APR = borrow APY − reward token APY equivalent
- Break-even horizon (days) considering rewards

Background
----------
Morpho runs reward programs where suppliers and borrowers of selected markets
accrue reward tokens (e.g. MORPHO, WELL, OP, etc.) proportional to their
USD-denominated position size and time held.  The reward rate is expressed as
USD value of rewards earned per USD supplied/borrowed per year.

When the reward APY offset is larger than the borrow APY, borrowing has a
*positive* effective cost (you get paid to borrow).  The opportunity scanner
uses these functions to surface such arbitrage windows.

Reference
---------
https://docs.morpho.org/rewards
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default Morpho reward rates (USD of reward earned per USD supplied/borrowed per year).
# These are rough market averages; override via RewardsCalculator.set_reward_rate().
# Real values fluctuate and should be fetched from the Morpho API.
DEFAULT_SUPPLY_REWARD_RATE: float = 0.015   # 1.5% per year on supply
DEFAULT_BORROW_REWARD_RATE: float = 0.010   # 1.0% per year on borrow


@dataclass
class RewardEstimate:
    """Point-in-time reward estimate for a position."""

    market_key: str
    side: str                          # "supply" or "borrow"
    position_usd: float
    days: float
    reward_rate_annual: float          # USD reward / USD position / year
    reward_usd: float
    reward_pct_annual: float           # annualised reward as % of position

    @property
    def effective_apr_offset(self) -> float:
        """Annualised reward offset as a decimal (e.g. 0.015 = 1.5%)."""
        return self.reward_rate_annual


@dataclass
class NetAPR:
    """Net APR breakdown for a supply or borrow position."""

    market_key: str
    side: str                          # "supply" or "borrow"
    base_apy: float                    # underlying Morpho APY
    reward_rate: float                 # reward APY equivalent
    net_apr: float                     # base_apy + reward_rate (supply) or base_apy - reward_rate (borrow)
    is_positive_carry: bool            # True when net borrow APR < 0 (rewards > cost)

    @property
    def net_apr_pct(self) -> float:
        return round(self.net_apr * 100, 4)

    @property
    def base_apy_pct(self) -> float:
        return round(self.base_apy * 100, 4)

    @property
    def reward_pct(self) -> float:
        return round(self.reward_rate * 100, 4)


@dataclass
class BreakEvenAnalysis:
    """Break-even analysis for a supply position versus borrow cost."""

    market_key: str
    supply_usd: float
    borrow_usd: float
    net_daily_earn_usd: float
    break_even_days: float             # inf when position is net-negative
    profitable: bool


class RewardsCalculator:
    """Calculates reward accrual and net APR for Morpho positions.

    Parameters
    ----------
    supply_reward_rates:
        Map of ``market_key → annual_reward_rate`` for supply side.
        Defaults to ``DEFAULT_SUPPLY_REWARD_RATE`` for unknown markets.
    borrow_reward_rates:
        Same for borrow side.

    Examples
    --------
    >>> calc = RewardsCalculator()
    >>> est = calc.estimate_supply_rewards("0xabc", 10_000, days=30)
    >>> print(est.reward_usd)   # ≈ 12.33 USD
    """

    def __init__(
        self,
        supply_reward_rates: Optional[Dict[str, float]] = None,
        borrow_reward_rates: Optional[Dict[str, float]] = None,
    ) -> None:
        self._supply_rates: Dict[str, float] = supply_reward_rates or {}
        self._borrow_rates: Dict[str, float] = borrow_reward_rates or {}

    # ------------------------------------------------------------------
    # Rate management
    # ------------------------------------------------------------------

    def set_supply_reward_rate(self, market_key: str, rate: float) -> None:
        """Set the annual reward rate for supply on *market_key*.

        Parameters
        ----------
        market_key:  Morpho market uniqueKey.
        rate:        Annual reward rate as a decimal (e.g. 0.02 = 2% per year).
        """
        self._supply_rates[market_key] = rate

    def set_borrow_reward_rate(self, market_key: str, rate: float) -> None:
        """Set the annual reward rate for borrow on *market_key*."""
        self._borrow_rates[market_key] = rate

    def get_supply_reward_rate(self, market_key: str) -> float:
        return self._supply_rates.get(market_key, DEFAULT_SUPPLY_REWARD_RATE)

    def get_borrow_reward_rate(self, market_key: str) -> float:
        return self._borrow_rates.get(market_key, DEFAULT_BORROW_REWARD_RATE)

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------

    def estimate_supply_rewards(
        self,
        market_key: str,
        position_usd: float,
        days: float = 365.25,
    ) -> RewardEstimate:
        """Estimate supply-side reward accrual over *days* days.

        Parameters
        ----------
        market_key:     Morpho market uniqueKey (used to look up reward rate).
        position_usd:   USD value of supplied position.
        days:           Projection horizon in days.
        """
        rate = self.get_supply_reward_rate(market_key)
        reward_usd = position_usd * rate * days / 365.25
        return RewardEstimate(
            market_key=market_key,
            side="supply",
            position_usd=position_usd,
            days=days,
            reward_rate_annual=rate,
            reward_usd=round(reward_usd, 6),
            reward_pct_annual=round(rate * 100, 4),
        )

    def estimate_borrow_rewards(
        self,
        market_key: str,
        position_usd: float,
        days: float = 365.25,
    ) -> RewardEstimate:
        """Estimate borrow-side reward accrual over *days* days."""
        rate = self.get_borrow_reward_rate(market_key)
        reward_usd = position_usd * rate * days / 365.25
        return RewardEstimate(
            market_key=market_key,
            side="borrow",
            position_usd=position_usd,
            days=days,
            reward_rate_annual=rate,
            reward_usd=round(reward_usd, 6),
            reward_pct_annual=round(rate * 100, 4),
        )

    def net_supply_apr(
        self,
        market_key: str,
        base_apy: float,
    ) -> NetAPR:
        """Compute net supply APR = base_apy + supply_reward_rate.

        Parameters
        ----------
        market_key:  Market uniqueKey.
        base_apy:    Base supply APY from on-chain / API (as a decimal).
        """
        reward_rate = self.get_supply_reward_rate(market_key)
        net = base_apy + reward_rate
        return NetAPR(
            market_key=market_key,
            side="supply",
            base_apy=base_apy,
            reward_rate=reward_rate,
            net_apr=round(net, 6),
            is_positive_carry=net > 0,
        )

    def net_borrow_apr(
        self,
        market_key: str,
        borrow_apy: float,
    ) -> NetAPR:
        """Compute net borrow APR = borrow_apy − borrow_reward_rate.

        A negative net borrow APR means you are being paid to borrow
        (rewards exceed interest cost) — a positive carry opportunity.
        """
        reward_rate = self.get_borrow_reward_rate(market_key)
        net = borrow_apy - reward_rate
        return NetAPR(
            market_key=market_key,
            side="borrow",
            base_apy=borrow_apy,
            reward_rate=reward_rate,
            net_apr=round(net, 6),
            is_positive_carry=net < 0,
        )

    def net_spread(
        self,
        market_key: str,
        supply_apy: float,
        borrow_apy: float,
    ) -> float:
        """Compute the net spread = net_supply_apr − net_borrow_apr.

        Positive spread means supplying earns more (after rewards) than
        borrowing costs (after rewards).
        """
        net_s = self.net_supply_apr(market_key, supply_apy).net_apr
        net_b = self.net_borrow_apr(market_key, borrow_apy).net_apr
        return round(net_s - net_b, 6)

    def break_even_analysis(
        self,
        market_key: str,
        supply_usd: float,
        borrow_usd: float,
        supply_apy: float,
        borrow_apy: float,
    ) -> BreakEvenAnalysis:
        """Compute how many days until supply rewards offset borrow cost.

        Considers both base APY and reward rates on both sides.

        Parameters
        ----------
        supply_usd:  USD value of supplied position.
        borrow_usd:  USD value of borrowed position.
        supply_apy:  Base supply APY (decimal).
        borrow_apy:  Base borrow APY (decimal).
        """
        net_s = self.net_supply_apr(market_key, supply_apy).net_apr
        net_b = self.net_borrow_apr(market_key, borrow_apy).net_apr

        daily_earn = supply_usd * net_s / 365.25
        daily_cost = borrow_usd * net_b / 365.25  # may be negative (paid to borrow)
        net_daily = daily_earn - daily_cost

        if net_daily > 0:
            break_even = 0.0
            profitable = True
        elif net_daily == 0:
            break_even = float("inf")
            profitable = False
        else:
            break_even = float("inf")
            profitable = False

        return BreakEvenAnalysis(
            market_key=market_key,
            supply_usd=supply_usd,
            borrow_usd=borrow_usd,
            net_daily_earn_usd=round(net_daily, 6),
            break_even_days=round(break_even, 2),
            profitable=profitable,
        )

    def compare_markets(
        self,
        markets: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        """Rank markets by net supply APR (including rewards).

        Parameters
        ----------
        markets:
            List of dicts with keys ``market_key``, ``supply_apy``, ``borrow_apy``.

        Returns a sorted list (highest net supply APR first) with added
        ``net_supply_apr_pct`` and ``net_borrow_apr_pct`` fields.
        """
        result = []
        for m in markets:
            key = str(m["market_key"])
            s_apy = float(m["supply_apy"])
            b_apy = float(m["borrow_apy"])
            net_s = self.net_supply_apr(key, s_apy)
            net_b = self.net_borrow_apr(key, b_apy)
            row = dict(m)
            row["net_supply_apr_pct"] = net_s.net_apr_pct
            row["net_borrow_apr_pct"] = net_b.net_apr_pct
            row["supply_reward_pct"] = net_s.reward_pct
            row["borrow_reward_pct"] = net_b.reward_pct
            row["positive_carry"] = net_b.is_positive_carry
            result.append(row)
        return sorted(result, key=lambda x: x["net_supply_apr_pct"], reverse=True)
