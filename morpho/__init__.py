"""Morpho Blue on Polygon — standalone production-grade package.

Structure
---------
morpho/
  contracts.py   — Contract addresses, token addresses, minimal ABIs
  markets.py     — Market configurations and ID computation
  client.py      — Full Morpho Blue client (query + agent payload)
  growth.py      — Sequential self-funding growth engine
  simulation.py  — Position simulation and projections
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

__all__ = [
    "MorphoBlueClient",
    "GrowthEngine",
    "MarketRegistry",
    "PositionSimulator",
    "build_market_id",
    "MORPHO_BLUE_POLYGON",
    "TOKEN_ADDRESSES",
    "ORACLE_ADDRESSES",
    "IRM_ADDRESS",
]
