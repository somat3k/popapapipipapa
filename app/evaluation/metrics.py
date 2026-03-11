"""Agent and strategy evaluation metrics.

Extends the base :class:`~app.trading.algorithms.TradingMetrics` with
additional metrics relevant to the Multiplex Financials DeFi AI platform:

  - Account growth rate (CAGR)
  - Risk-adjusted growth (growth per unit of volatility)
  - DeFi yield contribution (how much of the return comes from DeFi vs. trading)
  - Health-factor weighted score (penalises strategies that run dangerously
    close to liquidation)
  - Hit-rate (% of evaluation windows where the account grew)
  - Recovery speed after drawdown
  - Information coefficient (rank correlation of predictions with returns)
  - Composite ``AgentScore`` combining all metrics into a single grade

Usage
-----
::

    from app.evaluation.metrics import AgentEvaluationMetrics, AccountGrowthTracker

    metrics = AgentEvaluationMetrics(
        returns=np.array([...]),
        defi_yields=np.array([...]),
        health_factors=np.array([...]),
    )
    report = metrics.full_report()
    print(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.trading.algorithms import TradingMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health-factor scoring helpers
# ---------------------------------------------------------------------------

# Health-factor grade thresholds (consistent with morpho/growth.py)
HF_GRADE_THRESHOLDS: Dict[str, float] = {
    "A+": 2.0,
    "A":  1.5,
    "B":  1.2,
    "C":  1.0,
}


def health_factor_grade(hf: float) -> str:
    """Return the health-factor grade string."""
    if hf > HF_GRADE_THRESHOLDS["A+"]:
        return "A+"
    if hf > HF_GRADE_THRESHOLDS["A"]:
        return "A"
    if hf > HF_GRADE_THRESHOLDS["B"]:
        return "B"
    if hf > HF_GRADE_THRESHOLDS["C"]:
        return "C"
    return "D"


def hf_penalty(hf: float) -> float:
    """Return a 0–1 penalty factor for low health factors.

    Returns 0 for HF ≥ 2 (no penalty) and 1 for HF ≤ 1 (maximum penalty).
    """
    if hf >= 2.0:
        return 0.0
    if hf <= 1.0:
        return 1.0
    return float((2.0 - hf) / 1.0)  # linear interpolation between 1 and 2


# ---------------------------------------------------------------------------
# Core evaluation metrics
# ---------------------------------------------------------------------------

class AgentEvaluationMetrics:
    """Comprehensive evaluation of an agent's combined trading + DeFi performance.

    Parameters
    ----------
    returns:
        Array of per-period portfolio returns (e.g. daily).  Must be of
        the same length as *defi_yields* and *health_factors* when supplied.
    defi_yields:
        Per-period DeFi yield contributions (positive = supply APY earned
        minus borrow APY paid).  Used to decompose total return.
    health_factors:
        Per-period Morpho health-factor readings.  Used to penalise
        strategies that operate with dangerously low HF values.
    predictions:
        (Optional) Model price predictions, used to compute the
        Information Coefficient.
    risk_free_rate:
        Annual risk-free rate.  Defaults to 0.04.
    """

    def __init__(
        self,
        returns: np.ndarray,
        defi_yields: Optional[np.ndarray] = None,
        health_factors: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.returns = np.asarray(returns, dtype=float)
        n = len(self.returns)
        self.defi_yields = (
            np.asarray(defi_yields, dtype=float)
            if defi_yields is not None
            else np.zeros(n)
        )
        self.health_factors = (
            np.asarray(health_factors, dtype=float)
            if health_factors is not None
            else np.full(n, 2.0)
        )
        self.predictions = (
            np.asarray(predictions, dtype=float)
            if predictions is not None
            else None
        )
        self.risk_free_rate = risk_free_rate
        self._base = TradingMetrics(self.returns, risk_free_rate)

    # ------------------------------------------------------------------
    # Delegated base metrics
    # ------------------------------------------------------------------

    def sharpe(self) -> float:
        return self._base.sharpe()

    def sortino(self) -> float:
        return self._base.sortino()

    def max_drawdown(self) -> float:
        return self._base.max_drawdown()

    def win_rate(self) -> float:
        return self._base.win_rate()

    def profit_factor(self) -> float:
        return self._base.profit_factor()

    def calmar(self) -> float:
        return self._base.calmar()

    # ------------------------------------------------------------------
    # Extended metrics
    # ------------------------------------------------------------------

    def cagr(self, periods_per_year: float = 252.0) -> float:
        """Compound Annual Growth Rate.

        Parameters
        ----------
        periods_per_year:
            Number of return periods in a year (252 for daily, 52 for weekly).
        """
        n = len(self.returns)
        if n == 0:
            return 0.0
        cum_return = float(np.prod(1.0 + self.returns))
        if cum_return <= 0:
            return -1.0
        years = n / periods_per_year
        return float(cum_return ** (1.0 / years) - 1.0)

    def risk_adjusted_growth(self, periods_per_year: float = 252.0) -> float:
        """CAGR divided by annualised volatility (like Sharpe but uses CAGR)."""
        vol = float(np.std(self.returns)) * np.sqrt(periods_per_year)
        c = self.cagr(periods_per_year)
        return c / vol if vol > 1e-12 else 0.0

    def defi_yield_contribution(self) -> float:
        """Fraction of total return attributable to DeFi yield.

        Returns a value between 0 and 1 (or negative if DeFi was a net drag).
        """
        total = float(np.sum(self.returns))
        defi_total = float(np.sum(self.defi_yields))
        if abs(total) < 1e-12:
            return 0.0
        return defi_total / total

    def hit_rate(self) -> float:
        """Fraction of periods where the account grew (return > 0)."""
        if len(self.returns) == 0:
            return 0.0
        return float(np.mean(self.returns > 0))

    def health_factor_score(self) -> float:
        """Average health-factor score, penalised for dangerous HF values.

        Returns a score in [0, 1]:
          1.0 = always maintained HF ≥ 2 (no risk)
          0.0 = frequently at or below liquidation threshold
        """
        penalties = np.array([hf_penalty(hf) for hf in self.health_factors])
        return float(1.0 - np.mean(penalties))

    def average_health_factor(self) -> float:
        """Arithmetic mean of observed health factors."""
        return float(np.mean(self.health_factors))

    def min_health_factor(self) -> float:
        """Minimum observed health factor (worst-case scenario)."""
        return float(np.min(self.health_factors))

    def recovery_speed(self) -> float:
        """Average number of periods to recover from a drawdown of any size.

        Returns 0.0 if there are no drawdowns, or inf if recovery never
        completes.
        """
        cum = np.cumprod(1.0 + self.returns)
        peak = np.maximum.accumulate(cum)
        underwater = cum < peak

        if not np.any(underwater):
            return 0.0

        recovery_periods: List[int] = []
        i = 0
        n = len(cum)
        while i < n:
            if underwater[i]:
                start = i
                while i < n and cum[i] < peak[start]:
                    i += 1
                if i < n:
                    recovery_periods.append(i - start)
                else:
                    recovery_periods.append(n - start)
            else:
                i += 1

        return float(np.mean(recovery_periods)) if recovery_periods else float("inf")

    def information_coefficient(self) -> float:
        """Rank correlation (Spearman) between predictions and actual returns.

        Returns 0.0 when predictions are not provided.
        """
        if self.predictions is None or len(self.predictions) < 2:
            return 0.0
        n = min(len(self.predictions), len(self.returns))
        pred = self.predictions[:n]
        actual = self.returns[:n]
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(pred, actual)
            return float(corr) if np.isfinite(corr) else 0.0
        except ImportError:
            # Manual rank correlation fallback
            rank_pred = np.argsort(np.argsort(pred)).astype(float)
            rank_actual = np.argsort(np.argsort(actual)).astype(float)
            d2 = np.sum((rank_pred - rank_actual) ** 2)
            return float(1 - 6 * d2 / (n * (n ** 2 - 1)))

    def hf_adjusted_sharpe(self) -> float:
        """Sharpe ratio multiplied by the health-factor score.

        Penalises high-return strategies that achieved returns by running
        dangerously close to liquidation.
        """
        return self.sharpe() * self.health_factor_score()

    def composite_score(self) -> float:
        """Single composite score combining all key metrics.

        Weighted combination:
          0.25 × hf_adjusted_sharpe (normalised)
          0.25 × CAGR (clipped to [0, 1])
          0.20 × win_rate
          0.15 × health_factor_score
          0.15 × (1 + IC) / 2  (IC normalised to [0, 1])
        """
        sharpe_norm = max(0.0, min(1.0, self.hf_adjusted_sharpe() / 3.0))
        cagr_norm = max(0.0, min(1.0, self.cagr()))
        wr = self.win_rate()
        hf_score = self.health_factor_score()
        ic_norm = (self.information_coefficient() + 1.0) / 2.0

        return (
            0.25 * sharpe_norm
            + 0.25 * cagr_norm
            + 0.20 * wr
            + 0.15 * hf_score
            + 0.15 * ic_norm
        )

    def letter_grade(self) -> str:
        """Letter grade based on composite score."""
        score = self.composite_score()
        if score >= 0.8:
            return "A+"
        if score >= 0.65:
            return "A"
        if score >= 0.50:
            return "B"
        if score >= 0.35:
            return "C"
        return "D"

    def full_report(self) -> Dict[str, Any]:
        """Return a complete evaluation report as a flat dict."""
        return {
            # Base metrics
            "sharpe": round(self.sharpe(), 4),
            "sortino": round(self.sortino(), 4),
            "max_drawdown": round(self.max_drawdown(), 4),
            "win_rate": round(self.win_rate(), 4),
            "profit_factor": round(self.profit_factor(), 4),
            "calmar": round(self.calmar(), 4),
            # Extended metrics
            "cagr": round(self.cagr(), 4),
            "risk_adjusted_growth": round(self.risk_adjusted_growth(), 4),
            "defi_yield_contribution": round(self.defi_yield_contribution(), 4),
            "hit_rate": round(self.hit_rate(), 4),
            "health_factor_score": round(self.health_factor_score(), 4),
            "avg_health_factor": round(self.average_health_factor(), 4),
            "min_health_factor": round(self.min_health_factor(), 4),
            "recovery_speed": round(self.recovery_speed(), 2),
            "information_coefficient": round(self.information_coefficient(), 4),
            "hf_adjusted_sharpe": round(self.hf_adjusted_sharpe(), 4),
            # Composite
            "composite_score": round(self.composite_score(), 4),
            "letter_grade": self.letter_grade(),
        }


# ---------------------------------------------------------------------------
# Account Growth Tracker
# ---------------------------------------------------------------------------

@dataclass
class GrowthMilestone:
    """Records when the account passed a growth milestone."""

    milestone: float   # e.g. 1.5 for +50%
    period: int        # index into the equity curve
    value: float       # actual portfolio value at the milestone


class AccountGrowthTracker:
    """Track cumulative account value and growth milestones over time.

    Implements the "constant get-better" target: every new episode should
    advance toward the next milestone.  The tracker records each milestone
    crossing and reports whether the account is on track.

    Parameters
    ----------
    initial_value:
        Starting account value (e.g. 10_000 USD).
    milestone_step:
        Growth percentage step between milestones (e.g. 0.10 for 10% steps).

    Examples
    --------
    ::

        tracker = AccountGrowthTracker(initial_value=10_000.0)
        for period, return_pct in enumerate(daily_returns):
            tracker.update(period, return_pct)
        print(tracker.summary())
    """

    def __init__(
        self,
        initial_value: float = 10_000.0,
        milestone_step: float = 0.10,
    ) -> None:
        self.initial_value = initial_value
        self.milestone_step = milestone_step
        self._value = initial_value
        self._history: List[Tuple[int, float]] = []        # (period, value)
        self._milestones: List[GrowthMilestone] = []
        self._next_milestone = initial_value * (1.0 + milestone_step)

    def update(self, period: int, period_return: float) -> Optional[GrowthMilestone]:
        """Apply a single-period return and check for milestone crossings.

        Parameters
        ----------
        period:
            Period index (e.g. bar number or day number).
        period_return:
            Fractional return for this period (e.g. 0.02 for +2%).

        Returns
        -------
        GrowthMilestone or None
            A milestone object if one was crossed this period.
        """
        self._value *= 1.0 + period_return
        self._history.append((period, self._value))

        milestone_crossed: Optional[GrowthMilestone] = None
        while self._value >= self._next_milestone:
            m = GrowthMilestone(
                milestone=self._next_milestone / self.initial_value,
                period=period,
                value=self._value,
            )
            self._milestones.append(m)
            milestone_crossed = m
            logger.info(
                "[GrowthTracker] Milestone reached: +%.0f%%  value=%.2f  period=%d",
                (m.milestone - 1) * 100,
                m.value,
                period,
            )
            self._next_milestone *= 1.0 + self.milestone_step

        return milestone_crossed

    @property
    def current_value(self) -> float:
        return self._value

    @property
    def total_return(self) -> float:
        return (self._value / self.initial_value) - 1.0

    @property
    def milestones_reached(self) -> List[GrowthMilestone]:
        return list(self._milestones)

    def equity_curve(self) -> List[float]:
        """Return the list of account values in chronological order."""
        return [v for _, v in self._history]

    def cagr(self, periods_per_year: float = 252.0) -> float:
        """CAGR based on the tracked equity curve."""
        n = len(self._history)
        if n == 0:
            return 0.0
        years = n / periods_per_year
        cum = self._value / self.initial_value
        if cum <= 0 or years <= 0:
            return -1.0
        return float(cum ** (1.0 / years) - 1.0)

    def is_on_track(self, target_annual_growth: float = 0.20) -> bool:
        """Return True if current CAGR is on track for *target_annual_growth*."""
        return self.cagr() >= target_annual_growth * 0.75  # 75% of target

    def summary(self) -> Dict[str, Any]:
        n = len(self._history)
        return {
            "initial_value": round(self.initial_value, 2),
            "current_value": round(self._value, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "cagr_pct": round(self.cagr() * 100, 2),
            "periods": n,
            "milestones_reached": len(self._milestones),
            "next_milestone_value": round(self._next_milestone, 2),
            "on_track_20pct": self.is_on_track(0.20),
        }
