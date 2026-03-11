"""Real-time opportunity scanner for Morpho Blue on Polygon.

Answers the question: *"Where can we gain better opportunity than we actually
are, based on actual calculations and availability for actual conditions?"*

Features
--------
- **Market scanning** — compare all live markets by net APR (with rewards)
- **Best supply market** — highest net supply APR for a given liquidity amount
- **Best borrow token swap classification** — identify when switching the
  borrowed token delivers a better net position
- **Borrow capacity scale-up** — calculate how much additional borrow is
  available after an incremental supply action
- **Rebalance trigger** — determine whether the current position should be
  moved to a better market based on a configurable improvement threshold
- **Opportunity ranking** — ranked list of all opportunities with scores

Opportunity score
-----------------
Each opportunity is scored 0–100:

  score = 50 × (net_supply_apr / max_supply_apr) +
          30 × (1 − utilization) +
          20 × (liquidity_usd / max_liquidity_usd)

A score above 70 is considered "prime".  The scanner surfaces only
opportunities with score > *min_score* (default 40) and liquidity
above *min_liquidity_usd* (default 1000).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .api import APIMarketState, MorphoAPIClient
from .rewards import RewardsCalculator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_OPPORTUNITY_SCORE: float = 40.0
MIN_LIQUIDITY_USD: float = 1_000.0
PRIME_SCORE_THRESHOLD: float = 70.0
DEFAULT_REBALANCE_DELTA_PCT: float = 0.5   # rebalance if > 0.5% net APR improvement
SAFE_BORROW_LTV_RATIO: float = 0.50        # borrow at 50% of max allowed LTV


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class OpportunityScore:
    """Scored opportunity for a single Morpho market."""

    market_key: str
    loan_symbol: str
    collateral_symbol: str
    lltv: float
    supply_apy_pct: float
    borrow_apy_pct: float
    net_supply_apr_pct: float
    net_borrow_apr_pct: float
    utilization_pct: float
    liquidity_usd: float
    score: float

    @property
    def is_prime(self) -> bool:
        return self.score >= PRIME_SCORE_THRESHOLD

    @property
    def label(self) -> str:
        if self.score >= PRIME_SCORE_THRESHOLD:
            return "PRIME"
        if self.score >= 55:
            return "GOOD"
        if self.score >= MIN_OPPORTUNITY_SCORE:
            return "FAIR"
        return "POOR"


@dataclass
class BorrowSwapOpportunity:
    """Opportunity to switch the borrowed token for better net economics.

    When borrowing USDC_E in one market, but USDC in another market has a
    lower borrow APY (or higher borrow rewards), swapping the borrow token
    improves net carry.
    """

    from_market_key: str
    from_loan_symbol: str
    from_net_borrow_apr_pct: float
    to_market_key: str
    to_loan_symbol: str
    to_net_borrow_apr_pct: float
    saving_pct: float              # improvement in net borrow cost (positive = saving)
    swap_route_available: bool
    classification: str            # "cost_saving" | "neutral" | "worse"


@dataclass
class BorrowCapacity:
    """Available borrow capacity in a market after a supply action."""

    market_key: str
    loan_symbol: str
    collateral_symbol: str
    current_collateral_usd: float
    additional_supply_usd: float
    lltv: float
    max_borrow_usd: float
    current_borrow_usd: float
    available_borrow_usd: float    # max_borrow − current_borrow (at safe LTV)
    safe_additional_borrow_usd: float  # at SAFE_BORROW_LTV_RATIO of max


@dataclass
class RebalanceRecommendation:
    """Recommendation for whether to rebalance from current → best market."""

    current_market_key: str
    current_net_apr_pct: float
    best_market_key: str
    best_net_apr_pct: float
    improvement_pct: float
    should_rebalance: bool
    reason: str


# ---------------------------------------------------------------------------
# Opportunity Scanner
# ---------------------------------------------------------------------------

class OpportunityScanner:
    """Real-time opportunity scanner for Morpho Blue markets.

    Parameters
    ----------
    api_client:
        :class:`~morpho.api.MorphoAPIClient` instance.  When None, a default
        client is created (mock mode when offline).
    rewards_calculator:
        :class:`~morpho.rewards.RewardsCalculator` instance.  When None, a
        default calculator with baseline reward rates is used.
    min_score:
        Minimum opportunity score to include in results (0–100).
    min_liquidity_usd:
        Minimum available liquidity in USD for a market to be considered.
    rebalance_threshold_pct:
        Minimum net APR improvement (in percentage points) required to
        recommend rebalancing.

    Examples
    --------
    >>> scanner = OpportunityScanner()
    >>> ranked = scanner.rank_opportunities()
    >>> for opp in ranked[:3]:
    ...     print(opp.loan_symbol, opp.net_supply_apr_pct, opp.label)
    """

    def __init__(
        self,
        api_client: Optional[MorphoAPIClient] = None,
        rewards_calculator: Optional[RewardsCalculator] = None,
        min_score: float = MIN_OPPORTUNITY_SCORE,
        min_liquidity_usd: float = MIN_LIQUIDITY_USD,
        rebalance_threshold_pct: float = DEFAULT_REBALANCE_DELTA_PCT,
    ) -> None:
        self._api = api_client or MorphoAPIClient()
        self._calc = rewards_calculator or RewardsCalculator()
        self.min_score = min_score
        self.min_liquidity_usd = min_liquidity_usd
        self.rebalance_threshold = rebalance_threshold_pct

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def scan_markets(self) -> List[APIMarketState]:
        """Fetch and return all live markets (sorted by supply APY desc)."""
        return self._api.fetch_markets()

    def rank_opportunities(
        self,
        min_score: Optional[float] = None,
        min_liquidity_usd: Optional[float] = None,
    ) -> List[OpportunityScore]:
        """Rank all markets as scored opportunities.

        Returns a list of :class:`OpportunityScore` objects sorted by
        score (highest first), filtered by *min_score* and
        *min_liquidity_usd*.

        Parameters
        ----------
        min_score:          Override instance-level minimum score filter.
        min_liquidity_usd:  Override instance-level minimum liquidity filter.
        """
        threshold = min_score if min_score is not None else self.min_score
        liq_min = min_liquidity_usd if min_liquidity_usd is not None else self.min_liquidity_usd
        markets = self._api.fetch_markets()
        if not markets:
            return []

        max_apr = max(
            (self._calc.net_supply_apr(m.unique_key, m.supply_apy).net_apr for m in markets),
            default=0.01,
        )
        max_liq = max((m.liquidity_usd for m in markets), default=1.0) or 1.0

        scored: List[OpportunityScore] = []
        for m in markets:
            if m.liquidity_usd < liq_min:
                continue
            net_s = self._calc.net_supply_apr(m.unique_key, m.supply_apy)
            net_b = self._calc.net_borrow_apr(m.unique_key, m.borrow_apy)
            score = self._score(m, net_s.net_apr, max_apr, max_liq)
            if score < threshold:
                continue
            scored.append(
                OpportunityScore(
                    market_key=m.unique_key,
                    loan_symbol=m.loan_symbol,
                    collateral_symbol=m.collateral_symbol,
                    lltv=m.lltv,
                    supply_apy_pct=m.supply_apy_pct,
                    borrow_apy_pct=m.borrow_apy_pct,
                    net_supply_apr_pct=net_s.net_apr_pct,
                    net_borrow_apr_pct=net_b.net_apr_pct,
                    utilization_pct=m.utilization_pct,
                    liquidity_usd=m.liquidity_usd,
                    score=round(score, 2),
                )
            )
        return sorted(scored, key=lambda x: x.score, reverse=True)

    def find_best_supply_market(
        self,
        amount_usd: float = 10_000.0,
        loan_symbol: Optional[str] = None,
    ) -> Optional[OpportunityScore]:
        """Return the single best market to supply *amount_usd* into.

        Parameters
        ----------
        amount_usd:    Amount to supply (used for liquidity feasibility check).
        loan_symbol:   If provided, only consider markets with this loan token.
        """
        opportunities = self.rank_opportunities()
        for opp in opportunities:
            if loan_symbol and opp.loan_symbol != loan_symbol:
                continue
            if opp.liquidity_usd >= amount_usd:
                return opp
        # Fallback: return best even if liquidity is lower
        for opp in opportunities:
            if loan_symbol and opp.loan_symbol != loan_symbol:
                continue
            return opp
        return None

    # ------------------------------------------------------------------
    # Borrow token swap classification
    # ------------------------------------------------------------------

    def classify_borrow_token_swap(
        self,
        current_market_key: str,
        current_loan_symbol: str,
        collateral_symbol: str,
    ) -> List[BorrowSwapOpportunity]:
        """Find and classify opportunities to swap to a different borrow token.

        Examines all markets with the same collateral token and compares the
        net borrow APR.  Returns a list of :class:`BorrowSwapOpportunity`
        objects for each alternative loan token market.

        Parameters
        ----------
        current_market_key:   UniqueKey of the current market.
        current_loan_symbol:  Symbol of the currently borrowed token.
        collateral_symbol:    Symbol of the collateral token.
        """
        from .markets import BORROW_TOKEN_SWAP_ROUTES

        markets = self._api.fetch_markets()
        current_net_b = None
        same_collateral = []
        for m in markets:
            if m.collateral_symbol != collateral_symbol:
                continue
            net_b = self._calc.net_borrow_apr(m.unique_key, m.borrow_apy)
            if m.unique_key == current_market_key:
                current_net_b = net_b.net_apr
            else:
                same_collateral.append((m, net_b.net_apr))

        if current_net_b is None:
            # Current market not found in API results — use zero as baseline
            current_net_b = 0.0

        results: List[BorrowSwapOpportunity] = []
        for m, net_b in same_collateral:
            saving = current_net_b - net_b
            # Check if a borrow token swap route exists
            route_exists = any(
                r["from_token"] == current_loan_symbol and r["to_token"] == m.loan_symbol
                for r in BORROW_TOKEN_SWAP_ROUTES
            )
            if saving > 0.001:
                classification = "cost_saving"
            elif saving < -0.001:
                classification = "worse"
            else:
                classification = "neutral"
            results.append(
                BorrowSwapOpportunity(
                    from_market_key=current_market_key,
                    from_loan_symbol=current_loan_symbol,
                    from_net_borrow_apr_pct=round(current_net_b * 100, 4),
                    to_market_key=m.unique_key,
                    to_loan_symbol=m.loan_symbol,
                    to_net_borrow_apr_pct=round(net_b * 100, 4),
                    saving_pct=round(saving * 100, 4),
                    swap_route_available=route_exists,
                    classification=classification,
                )
            )
        return sorted(results, key=lambda x: x.saving_pct, reverse=True)

    # ------------------------------------------------------------------
    # Borrow capacity scale-up
    # ------------------------------------------------------------------

    def get_borrow_capacity(
        self,
        market_key: str,
        loan_symbol: str,
        collateral_symbol: str,
        current_collateral_usd: float,
        current_borrow_usd: float,
        lltv: float,
        additional_supply_usd: float = 0.0,
        safe_ltv_ratio: float = SAFE_BORROW_LTV_RATIO,
    ) -> BorrowCapacity:
        """Calculate available borrow capacity after a (possibly incremental) supply.

        Shows how much additional borrowing capacity opens up when the
        supplied collateral is increased by *additional_supply_usd*.

        Parameters
        ----------
        market_key:               Market uniqueKey.
        loan_symbol:              Symbol of loan token.
        collateral_symbol:        Symbol of collateral token.
        current_collateral_usd:   Current collateral value in USD.
        current_borrow_usd:       Current outstanding borrow in USD.
        lltv:                     Liquidation LTV as a decimal (e.g. 0.86).
        additional_supply_usd:    Extra collateral being supplied now.
        safe_ltv_ratio:           Fraction of max LTV to use for safe borrowing.
        """
        total_collateral = current_collateral_usd + additional_supply_usd
        max_borrow = total_collateral * lltv
        safe_max = max_borrow * safe_ltv_ratio
        available = max(0.0, safe_max - current_borrow_usd)

        return BorrowCapacity(
            market_key=market_key,
            loan_symbol=loan_symbol,
            collateral_symbol=collateral_symbol,
            current_collateral_usd=current_collateral_usd,
            additional_supply_usd=additional_supply_usd,
            lltv=lltv,
            max_borrow_usd=round(max_borrow, 2),
            current_borrow_usd=current_borrow_usd,
            available_borrow_usd=round(max(0.0, max_borrow - current_borrow_usd), 2),
            safe_additional_borrow_usd=round(available, 2),
        )

    # ------------------------------------------------------------------
    # Rebalance recommendation
    # ------------------------------------------------------------------

    def should_rebalance(
        self,
        current_market_key: str,
        current_supply_apy: float,
        amount_usd: float = 10_000.0,
        threshold_pct: Optional[float] = None,
    ) -> RebalanceRecommendation:
        """Recommend whether to rebalance to a better market.

        Compares the net supply APR of the *current_market_key* against the
        best available market.  If the improvement exceeds *threshold_pct*
        percentage points, recommends rebalancing.

        Parameters
        ----------
        current_market_key:  UniqueKey of the currently used market.
        current_supply_apy:  Current base supply APY in the active market.
        amount_usd:          Supply amount for liquidity feasibility.
        threshold_pct:       Override the instance-level rebalance threshold.
        """
        delta = threshold_pct if threshold_pct is not None else self.rebalance_threshold
        current_net = self._calc.net_supply_apr(current_market_key, current_supply_apy)
        current_pct = current_net.net_apr_pct

        best = self.find_best_supply_market(amount_usd=amount_usd)
        if best is None:
            return RebalanceRecommendation(
                current_market_key=current_market_key,
                current_net_apr_pct=current_pct,
                best_market_key=current_market_key,
                best_net_apr_pct=current_pct,
                improvement_pct=0.0,
                should_rebalance=False,
                reason="No alternative markets found.",
            )

        improvement = best.net_supply_apr_pct - current_pct
        should = improvement >= delta and best.market_key != current_market_key
        reason = (
            f"Net APR improvement of {improvement:.2f}% ≥ threshold {delta:.2f}% — "
            f"move to {best.loan_symbol}/{best.collateral_symbol}."
            if should
            else (
                f"Current market net APR ({current_pct:.2f}%) is within {delta:.2f}% "
                f"of best ({best.net_supply_apr_pct:.2f}%) — no rebalance needed."
            )
        )
        return RebalanceRecommendation(
            current_market_key=current_market_key,
            current_net_apr_pct=current_pct,
            best_market_key=best.market_key,
            best_net_apr_pct=best.net_supply_apr_pct,
            improvement_pct=round(improvement, 4),
            should_rebalance=should,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(
        m: APIMarketState,
        net_apr: float,
        max_apr: float,
        max_liq: float,
    ) -> float:
        """Compute a 0–100 opportunity score for a market.

        Score = 50 × (net_apr / max_apr) + 30 × (1 − utilization) + 20 × (liq / max_liq)
        """
        apr_component = 50.0 * (net_apr / max_apr) if max_apr > 0 else 0.0
        util_component = 30.0 * (1.0 - m.utilization)
        liq_component = 20.0 * (m.liquidity_usd / max_liq)
        return round(apr_component + util_component + liq_component, 2)
