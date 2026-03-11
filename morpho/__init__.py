"""Morpho Blue on Polygon — standalone production-grade package.

Structure
---------
morpho/
  contracts.py    — Contract addresses, token addresses, minimal ABIs
  markets.py      — Market configurations and ID computation
  client.py       — Full Morpho Blue client (query + agent payload)
  growth.py       — Sequential self-funding growth engine
  simulation.py   — Position simulation and projections
  api.py          — Morpho GraphQL API client + transaction payload builders
  rewards.py      — Rewards calculations and net APR analysis
  opportunity.py  — Real-time opportunity scanner and rebalance logic
  visuals.py      — Text-based data visualisations (ASCII charts/tables)
"""

from .client import MorphoBlueClient
from .contracts import (
    MORPHO_BLUE_POLYGON,
    TOKEN_ADDRESSES,
    ORACLE_ADDRESSES,
    IRM_ADDRESS,
)
from .growth import GrowthEngine
from .markets import MarketRegistry, build_market_id
from .simulation import PositionSimulator
from .api import (
    MorphoAPIClient,
    APIMarketState,
    APIUserPosition,
    MarketRewards,
    RewardEntry,
    build_supply_payload,
    build_borrow_payload,
    build_repay_payload,
    build_withdraw_payload,
    build_supply_collateral_payload,
    build_withdraw_collateral_payload,
    MORPHO_API_URL,
)
from .rewards import RewardsCalculator, NetAPR, RewardEstimate, BreakEvenAnalysis
from .opportunity import (
    OpportunityScanner,
    OpportunityScore,
    BorrowSwapOpportunity,
    BorrowCapacity,
    RebalanceRecommendation,
)
from . import visuals

__all__ = [
    # client
    "MorphoBlueClient",
    # growth
    "GrowthEngine",
    # markets / registry
    "MarketRegistry",
    "build_market_id",
    # simulation
    "PositionSimulator",
    # contracts
    "MORPHO_BLUE_POLYGON",
    "TOKEN_ADDRESSES",
    "ORACLE_ADDRESSES",
    "IRM_ADDRESS",
    # api
    "MorphoAPIClient",
    "APIMarketState",
    "APIUserPosition",
    "MarketRewards",
    "RewardEntry",
    "MORPHO_API_URL",
    "build_supply_payload",
    "build_borrow_payload",
    "build_repay_payload",
    "build_withdraw_payload",
    "build_supply_collateral_payload",
    "build_withdraw_collateral_payload",
    # rewards
    "RewardsCalculator",
    "NetAPR",
    "RewardEstimate",
    "BreakEvenAnalysis",
    # opportunity
    "OpportunityScanner",
    "OpportunityScore",
    "BorrowSwapOpportunity",
    "BorrowCapacity",
    "RebalanceRecommendation",
    # visuals
    "visuals",
]
