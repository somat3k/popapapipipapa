"""Position simulation and projection for Morpho Blue markets.

Provides what-if analysis without touching any real funds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .client import MorphoBlueClient, UserPosition

logger = logging.getLogger(__name__)

# Seconds in common time horizons
SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass
class SimulationResult:
    """Projected position after a given time horizon."""

    market_name: str
    horizon_days: float
    initial: dict
    projected: dict
    net_yield_usd: float = 0.0
    break_even_days: float = 0.0


class PositionSimulator:
    """Simulate Morpho Blue position evolution over time.

    Uses the current supply/borrow APY from the client and projects
    balance changes without executing any transactions.

    Parameters
    ----------
    client:
        A :class:`~morpho.client.MorphoBlueClient` instance.

    Examples
    --------
    >>> from morpho import MorphoBlueClient, PositionSimulator
    >>> client = MorphoBlueClient()
    >>> sim = PositionSimulator(client)
    >>> result = sim.project("WETH/USDC_E-86", horizon_days=30)
    """

    def __init__(self, client: MorphoBlueClient) -> None:
        self._client = client

    def project(
        self,
        market_name: str,
        horizon_days: float = 30.0,
        supply_delta: float = 0.0,
        borrow_delta: float = 0.0,
    ) -> SimulationResult:
        """Project position value after *horizon_days* days.

        Parameters
        ----------
        market_name:    Market to simulate.
        horizon_days:   Number of days to project forward.
        supply_delta:   Additional supply assets (in USD) to add at t=0.
        borrow_delta:   Additional borrow (in USD) to add at t=0.
        """
        pos = self._client.get_position(market_name)
        apy_info = self._client.market_apy(market_name)
        supply_apy = apy_info["supply_apy"]
        borrow_apy = apy_info["borrow_apy"]

        supply_start = pos.supply_assets + supply_delta
        borrow_start = pos.borrow_assets + borrow_delta
        horizon_years = horizon_days / 365.25

        supply_end = supply_start * (1 + supply_apy) ** horizon_years
        borrow_end = borrow_start * (1 + borrow_apy) ** horizon_years

        supply_earned = supply_end - supply_start
        borrow_cost = borrow_end - borrow_start
        net_yield = supply_earned - borrow_cost

        # Break-even: days until supply_earned equals borrow_cost
        # (i.e. net yield goes from 0 → positive)
        if supply_apy > borrow_apy and borrow_start > 0:
            # solve: supply_start*(1+supply_apy)^t = supply_start + borrow_start*(1+borrow_apy)^t - borrow_start
            # Approximate linearly: break_even = borrow_cost / (daily_supply_rate * supply_start)
            daily_net_earn = supply_start * supply_apy / 365.25 - borrow_start * borrow_apy / 365.25
            break_even = borrow_cost / daily_net_earn if daily_net_earn > 0 else float("inf")
        elif supply_earned > borrow_cost:
            break_even = 0.0
        else:
            break_even = float("inf")

        initial = {
            "supply_usd": round(supply_start, 4),
            "borrow_usd": round(borrow_start, 4),
            "health_factor": round(pos.health_factor, 4),
            "supply_apy_pct": round(supply_apy * 100, 2),
            "borrow_apy_pct": round(borrow_apy * 100, 2),
        }
        projected = {
            "supply_usd": round(supply_end, 4),
            "borrow_usd": round(borrow_end, 4),
            "supply_earned_usd": round(supply_earned, 4),
            "borrow_cost_usd": round(borrow_cost, 4),
        }
        return SimulationResult(
            market_name=market_name,
            horizon_days=horizon_days,
            initial=initial,
            projected=projected,
            net_yield_usd=round(net_yield, 4),
            break_even_days=round(break_even, 2),
        )

    def compare_markets(
        self,
        horizon_days: float = 30.0,
        supply_amount_usd: float = 10_000.0,
    ) -> list[dict]:
        """Compare projected yield across all registered markets.

        Returns a list of dicts sorted by projected net yield (descending).
        """
        results = []
        for market in self._client.list_markets():
            try:
                sim = self.project(
                    market.name,
                    horizon_days=horizon_days,
                    supply_delta=supply_amount_usd,
                )
                results.append(
                    {
                        "market": market.name,
                        "supply_apy_pct": sim.initial["supply_apy_pct"],
                        "borrow_apy_pct": sim.initial["borrow_apy_pct"],
                        "net_yield_usd": sim.net_yield_usd,
                        "horizon_days": horizon_days,
                    }
                )
            except Exception:
                logger.exception("[Simulator] Failed to project market %s", market.name)
        return sorted(results, key=lambda x: x["net_yield_usd"], reverse=True)

    def what_if_price_drop(
        self,
        market_name: str,
        price_drop_pct: float,
    ) -> dict:
        """Compute the new health factor if collateral price drops by *price_drop_pct* %.

        Parameters
        ----------
        market_name:      Market to stress-test.
        price_drop_pct:   Percentage drop in collateral price (e.g. 30 for -30%).
        """
        pos = self._client.get_position(market_name)
        if pos.borrow_assets == 0:
            return {"new_health_factor": float("inf"), "liquidatable": False}

        market = self._client._registry.get(market_name)
        if market is None:
            return {"error": f"Unknown market: {market_name}"}

        original_price = self._client.get_collateral_price(market.collateral_token_symbol)
        new_price = original_price * (1 - price_drop_pct / 100)
        lltv = market.lltv / 1e18
        collateral_units = pos.collateral / (10 ** market.collateral_decimals)
        collateral_value = collateral_units * new_price
        new_hf = (collateral_value * lltv) / pos.borrow_assets if pos.borrow_assets > 0 else float("inf")

        return {
            "original_price": original_price,
            "new_price": new_price,
            "price_drop_pct": price_drop_pct,
            "original_health_factor": round(pos.health_factor, 4),
            "new_health_factor": round(new_hf, 4),
            "liquidatable": new_hf < 1.0,
            "at_risk": 1.0 <= new_hf < 1.15,
        }
