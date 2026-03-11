"""Morpho GraphQL API client for real-time market data.

Provides both query (read) and inference functions backed by the public
Morpho API at https://api.morpho.org/graphql.  Falls back to a mock
response set when the network is unavailable, keeping the module fully
functional offline and in unit tests.

Query system
------------
- ``query(gql, variables)``       — raw GraphQL executor
- ``fetch_markets()``              — all markets with APY / utilization
- ``fetch_market(unique_key)``     — single market by its uniqueKey
- ``fetch_user_positions(addr)``   — user supply/borrow/collateral per market
- ``fetch_rewards(addr)``          — accrued Morpho reward tokens per market

Payload helpers
---------------
- ``build_supply_payload(market_params, assets, on_behalf, data)``
- ``build_borrow_payload(market_params, assets, on_behalf, receiver)``
- ``build_repay_payload(market_params, assets, on_behalf, data)``
- ``build_withdraw_payload(market_params, assets, on_behalf, receiver)``
- ``build_supply_collateral_payload(market_params, assets, on_behalf, data)``
- ``build_withdraw_collateral_payload(market_params, assets, on_behalf, receiver)``

Each payload builder returns a dict ready to be ABI-encoded by web3.py:
  ``morpho_contract.functions.supply(**payload).build_transaction(...)``
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MORPHO_API_URL = "https://api.morpho.org/graphql"
MORPHO_API_TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# GraphQL query templates
# ---------------------------------------------------------------------------

# All markets with current state (APY, utilization, liquidity)
_GQL_MARKETS = """
query MorphoMarkets($first: Int, $chainId: Int) {
  markets(first: $first, where: { chainId_in: [$chainId] }) {
    items {
      uniqueKey
      lltv
      loanAsset {
        symbol
        address
        decimals
      }
      collateralAsset {
        symbol
        address
        decimals
      }
      state {
        supplyApy
        borrowApy
        utilization
        totalSupplyAssets
        totalBorrowAssets
        totalSupplyAssetsUsd
        totalBorrowAssetsUsd
        liquidityAssetsUsd
      }
    }
  }
}
"""

# Single market by uniqueKey
_GQL_MARKET = """
query MorphoMarket($uniqueKey: String!) {
  marketByUniqueKey(uniqueKey: $uniqueKey) {
    uniqueKey
    lltv
    loanAsset {
      symbol
      address
      decimals
    }
    collateralAsset {
      symbol
      address
      decimals
    }
    state {
      supplyApy
      borrowApy
      utilization
      totalSupplyAssets
      totalBorrowAssets
      totalSupplyAssetsUsd
      totalBorrowAssetsUsd
      liquidityAssetsUsd
    }
  }
}
"""

# User positions across all markets
_GQL_USER_POSITIONS = """
query MorphoUserPositions($address: String!, $chainId: Int) {
  userByAddress(address: $address, chainId: $chainId) {
    positions {
      market {
        uniqueKey
        loanAsset { symbol decimals }
        collateralAsset { symbol decimals }
        state { supplyApy borrowApy utilization }
      }
      supplyShares
      borrowShares
      collateral
      supplyAssets
      borrowAssets
      healthFactor
    }
  }
}
"""

# User accrued rewards / points
_GQL_REWARDS = """
query MorphoRewards($address: String!, $chainId: Int) {
  userByAddress(address: $address, chainId: $chainId) {
    rewardPrograms {
      market {
        uniqueKey
        loanAsset { symbol }
      }
      supplyRewards {
        asset { symbol address }
        claimableNow
        claimableLater
        claimed
      }
      borrowRewards {
        asset { symbol address }
        claimableNow
        claimableLater
        claimed
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Mock data for offline / test use
# ---------------------------------------------------------------------------

def _mock_markets() -> dict:
    return {
        "data": {
            "markets": {
                "items": [
                    {
                        "uniqueKey": "0x" + "a1" * 32,
                        "lltv": "860000000000000000",
                        "loanAsset": {"symbol": "USDC_E", "address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "decimals": 6},
                        "collateralAsset": {"symbol": "WETH", "address": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619", "decimals": 18},
                        "state": {
                            "supplyApy": 0.042,
                            "borrowApy": 0.058,
                            "utilization": 0.72,
                            "totalSupplyAssets": 50_000_000_000,
                            "totalBorrowAssets": 36_000_000_000,
                            "totalSupplyAssetsUsd": 50000.0,
                            "totalBorrowAssetsUsd": 36000.0,
                            "liquidityAssetsUsd": 14000.0,
                        },
                    },
                    {
                        "uniqueKey": "0x" + "b2" * 32,
                        "lltv": "860000000000000000",
                        "loanAsset": {"symbol": "USDC_E", "address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "decimals": 6},
                        "collateralAsset": {"symbol": "WBTC", "address": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6", "decimals": 8},
                        "state": {
                            "supplyApy": 0.035,
                            "borrowApy": 0.051,
                            "utilization": 0.60,
                            "totalSupplyAssets": 20_000_000_000,
                            "totalBorrowAssets": 12_000_000_000,
                            "totalSupplyAssetsUsd": 20000.0,
                            "totalBorrowAssetsUsd": 12000.0,
                            "liquidityAssetsUsd": 8000.0,
                        },
                    },
                    {
                        "uniqueKey": "0x" + "c3" * 32,
                        "lltv": "770000000000000000",
                        "loanAsset": {"symbol": "USDC_E", "address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "decimals": 6},
                        "collateralAsset": {"symbol": "WPOL", "address": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270", "decimals": 18},
                        "state": {
                            "supplyApy": 0.028,
                            "borrowApy": 0.044,
                            "utilization": 0.50,
                            "totalSupplyAssets": 8_000_000_000,
                            "totalBorrowAssets": 4_000_000_000,
                            "totalSupplyAssetsUsd": 8000.0,
                            "totalBorrowAssetsUsd": 4000.0,
                            "liquidityAssetsUsd": 4000.0,
                        },
                    },
                    {
                        "uniqueKey": "0x" + "d4" * 32,
                        "lltv": "860000000000000000",
                        "loanAsset": {"symbol": "USDC", "address": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359", "decimals": 6},
                        "collateralAsset": {"symbol": "WETH", "address": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619", "decimals": 18},
                        "state": {
                            "supplyApy": 0.039,
                            "borrowApy": 0.055,
                            "utilization": 0.67,
                            "totalSupplyAssets": 30_000_000_000,
                            "totalBorrowAssets": 20_100_000_000,
                            "totalSupplyAssetsUsd": 30000.0,
                            "totalBorrowAssetsUsd": 20100.0,
                            "liquidityAssetsUsd": 9900.0,
                        },
                    },
                ]
            }
        }
    }


def _mock_user_positions(address: str) -> dict:
    return {
        "data": {
            "userByAddress": {
                "positions": [
                    {
                        "market": {
                            "uniqueKey": "0x" + "a1" * 32,
                            "loanAsset": {"symbol": "USDC_E", "decimals": 6},
                            "collateralAsset": {"symbol": "WETH", "decimals": 18},
                            "state": {"supplyApy": 0.042, "borrowApy": 0.058, "utilization": 0.72},
                        },
                        "supplyShares": "0",
                        "borrowShares": "0",
                        "collateral": "0",
                        "supplyAssets": 0.0,
                        "borrowAssets": 0.0,
                        "healthFactor": None,
                    }
                ]
            }
        }
    }


def _mock_rewards(address: str) -> dict:
    return {
        "data": {
            "userByAddress": {
                "rewardPrograms": []
            }
        }
    }


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class APIMarketState:
    """Market state as returned by the Morpho API."""

    unique_key: str
    loan_symbol: str
    loan_address: str
    loan_decimals: int
    collateral_symbol: str
    collateral_address: str
    collateral_decimals: int
    lltv: float                      # e.g. 0.86
    supply_apy: float                # e.g. 0.042  (= 4.2%)
    borrow_apy: float
    utilization: float               # 0–1
    total_supply_usd: float
    total_borrow_usd: float
    liquidity_usd: float

    @property
    def supply_apy_pct(self) -> float:
        return round(self.supply_apy * 100, 4)

    @property
    def borrow_apy_pct(self) -> float:
        return round(self.borrow_apy * 100, 4)

    @property
    def utilization_pct(self) -> float:
        return round(self.utilization * 100, 2)


@dataclass
class APIUserPosition:
    """User position as returned by the Morpho API."""

    market_key: str
    loan_symbol: str
    collateral_symbol: str
    supply_assets: float
    borrow_assets: float
    collateral: float
    health_factor: Optional[float]
    supply_apy: float
    borrow_apy: float


@dataclass
class RewardEntry:
    """Single reward token entry for a market."""

    asset_symbol: str
    asset_address: str
    claimable_now: float
    claimable_later: float
    claimed: float

    @property
    def total_claimable(self) -> float:
        return self.claimable_now + self.claimable_later


@dataclass
class MarketRewards:
    """Supply + borrow rewards for a market."""

    market_key: str
    loan_symbol: str
    supply_rewards: List[RewardEntry] = field(default_factory=list)
    borrow_rewards: List[RewardEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Morpho API Client
# ---------------------------------------------------------------------------

class MorphoAPIClient:
    """Client for the Morpho GraphQL API.

    Uses ``urllib`` (stdlib only) to send POST requests to the API endpoint.
    Falls back to mock data when the network is unavailable, so all methods
    work offline in tests and development.

    Parameters
    ----------
    chain_id:
        Chain ID to filter markets.  137 = Polygon, 1 = Ethereum mainnet.
    api_url:
        Override the default API URL.
    timeout:
        HTTP timeout in seconds.

    Examples
    --------
    >>> client = MorphoAPIClient()
    >>> markets = client.fetch_markets()
    >>> best = max(markets, key=lambda m: m.supply_apy)
    >>> print(best.loan_symbol, best.supply_apy_pct)
    """

    def __init__(
        self,
        chain_id: int = 137,
        api_url: str = MORPHO_API_URL,
        timeout: int = MORPHO_API_TIMEOUT,
    ) -> None:
        self.chain_id = chain_id
        self._url = api_url
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Core GraphQL executor
    # ------------------------------------------------------------------

    def query(
        self,
        gql: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a GraphQL query against the Morpho API.

        Returns the parsed JSON response dict.  On network failure, raises
        ``urllib.error.URLError`` — callers should handle this or use the
        higher-level methods that fall back to mocks automatically.

        Parameters
        ----------
        gql:        GraphQL query string.
        variables:  Optional dict of query variables.
        """
        payload = json.dumps(
            {"query": gql, "variables": variables or {}}
        ).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _safe_query(
        self,
        gql: str,
        variables: Dict[str, Any],
        fallback_fn: Any,
    ) -> Dict[str, Any]:
        """Execute query; on error, log and return fallback data."""
        try:
            return self.query(gql, variables)
        except Exception as exc:
            logger.debug("[MorphoAPI] Network unavailable (%s), using mock data.", exc)
            return fallback_fn()

    # ------------------------------------------------------------------
    # High-level query methods
    # ------------------------------------------------------------------

    def fetch_markets(self, first: int = 100) -> List[APIMarketState]:
        """Fetch all markets for the configured chain.

        Returns a list of :class:`APIMarketState` objects sorted by
        supply APY (highest first).
        """
        raw = self._safe_query(
            _GQL_MARKETS,
            {"first": first, "chainId": self.chain_id},
            _mock_markets,
        )
        items = (
            raw.get("data", {})
               .get("markets", {})
               .get("items", [])
        )
        markets: List[APIMarketState] = []
        for item in items:
            try:
                markets.append(self._parse_market(item))
            except Exception:
                logger.exception("[MorphoAPI] Failed to parse market item: %s", item)
        markets.sort(key=lambda m: m.supply_apy, reverse=True)
        return markets

    def fetch_market(self, unique_key: str) -> Optional[APIMarketState]:
        """Fetch a single market by its uniqueKey."""
        raw = self._safe_query(
            _GQL_MARKET,
            {"uniqueKey": unique_key},
            lambda: {"data": {"marketByUniqueKey": None}},
        )
        item = raw.get("data", {}).get("marketByUniqueKey")
        if item is None:
            return None
        try:
            return self._parse_market(item)
        except Exception:
            logger.exception("[MorphoAPI] Failed to parse single market: %s", item)
            return None

    def fetch_user_positions(self, address: str) -> List[APIUserPosition]:
        """Fetch user positions across all markets for *address*."""
        raw = self._safe_query(
            _GQL_USER_POSITIONS,
            {"address": address.lower(), "chainId": self.chain_id},
            lambda: _mock_user_positions(address),
        )
        positions_raw = (
            raw.get("data", {})
               .get("userByAddress") or {}
        ).get("positions", [])

        positions: List[APIUserPosition] = []
        for p in positions_raw:
            try:
                mkt = p["market"]
                hf_raw = p.get("healthFactor")
                positions.append(
                    APIUserPosition(
                        market_key=mkt["uniqueKey"],
                        loan_symbol=mkt["loanAsset"]["symbol"],
                        collateral_symbol=mkt["collateralAsset"]["symbol"],
                        supply_assets=float(p.get("supplyAssets") or 0),
                        borrow_assets=float(p.get("borrowAssets") or 0),
                        collateral=float(p.get("collateral") or 0),
                        health_factor=float(hf_raw) if hf_raw is not None else None,
                        supply_apy=mkt["state"]["supplyApy"],
                        borrow_apy=mkt["state"]["borrowApy"],
                    )
                )
            except Exception:
                logger.exception("[MorphoAPI] Failed to parse position: %s", p)
        return positions

    def fetch_rewards(self, address: str) -> List[MarketRewards]:
        """Fetch accrued rewards for *address* across all markets."""
        raw = self._safe_query(
            _GQL_REWARDS,
            {"address": address.lower(), "chainId": self.chain_id},
            lambda: _mock_rewards(address),
        )
        programs = (
            raw.get("data", {})
               .get("userByAddress") or {}
        ).get("rewardPrograms", [])

        result: List[MarketRewards] = []
        for prog in programs:
            try:
                mkt = prog["market"]
                mr = MarketRewards(
                    market_key=mkt["uniqueKey"],
                    loan_symbol=mkt["loanAsset"]["symbol"],
                    supply_rewards=[
                        RewardEntry(
                            asset_symbol=r["asset"]["symbol"],
                            asset_address=r["asset"]["address"],
                            claimable_now=float(r.get("claimableNow") or 0),
                            claimable_later=float(r.get("claimableLater") or 0),
                            claimed=float(r.get("claimed") or 0),
                        )
                        for r in prog.get("supplyRewards", [])
                    ],
                    borrow_rewards=[
                        RewardEntry(
                            asset_symbol=r["asset"]["symbol"],
                            asset_address=r["asset"]["address"],
                            claimable_now=float(r.get("claimableNow") or 0),
                            claimable_later=float(r.get("claimableLater") or 0),
                            claimed=float(r.get("claimed") or 0),
                        )
                        for r in prog.get("borrowRewards", [])
                    ],
                )
                result.append(mr)
            except Exception:
                logger.exception("[MorphoAPI] Failed to parse reward program: %s", prog)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_market(item: dict) -> APIMarketState:
        state = item["state"]
        lltv_raw = item.get("lltv", "0")
        lltv = int(lltv_raw) / 1e18 if str(lltv_raw).isdigit() else float(lltv_raw)
        loan = item["loanAsset"]
        coll = item["collateralAsset"]
        return APIMarketState(
            unique_key=item["uniqueKey"],
            loan_symbol=loan["symbol"],
            loan_address=loan["address"],
            loan_decimals=loan["decimals"],
            collateral_symbol=coll["symbol"],
            collateral_address=coll["address"],
            collateral_decimals=coll["decimals"],
            lltv=lltv,
            supply_apy=float(state.get("supplyApy") or 0),
            borrow_apy=float(state.get("borrowApy") or 0),
            utilization=float(state.get("utilization") or 0),
            total_supply_usd=float(state.get("totalSupplyAssetsUsd") or 0),
            total_borrow_usd=float(state.get("totalBorrowAssetsUsd") or 0),
            liquidity_usd=float(state.get("liquidityAssetsUsd") or 0),
        )


# ---------------------------------------------------------------------------
# Transaction payload builders
# ---------------------------------------------------------------------------

def build_supply_payload(
    market_params: tuple,
    assets: int,
    on_behalf: str,
    data: bytes = b"",
) -> Dict[str, Any]:
    """Build a ``supply`` call payload for ``web3.py``.

    Parameters
    ----------
    market_params:  ``(loanToken, collateralToken, oracle, irm, lltv)`` tuple.
    assets:         Amount in loan token base units.
    on_behalf:      Wallet address that receives the supply shares.
    data:           Optional callback data (usually empty).

    Returns a dict suitable for:
    ``morpho.functions.supply(**payload).build_transaction(tx_params)``
    """
    return {
        "marketParams": market_params,
        "assets": assets,
        "shares": 0,
        "onBehalf": on_behalf,
        "data": data,
    }


def build_borrow_payload(
    market_params: tuple,
    assets: int,
    on_behalf: str,
    receiver: str,
) -> Dict[str, Any]:
    """Build a ``borrow`` call payload for ``web3.py``."""
    return {
        "marketParams": market_params,
        "assets": assets,
        "shares": 0,
        "onBehalf": on_behalf,
        "receiver": receiver,
    }


def build_repay_payload(
    market_params: tuple,
    assets: int,
    on_behalf: str,
    data: bytes = b"",
) -> Dict[str, Any]:
    """Build a ``repay`` call payload for ``web3.py``."""
    return {
        "marketParams": market_params,
        "assets": assets,
        "shares": 0,
        "onBehalf": on_behalf,
        "data": data,
    }


def build_withdraw_payload(
    market_params: tuple,
    assets: int,
    on_behalf: str,
    receiver: str,
) -> Dict[str, Any]:
    """Build a ``withdraw`` call payload for ``web3.py``."""
    return {
        "marketParams": market_params,
        "assets": assets,
        "shares": 0,
        "onBehalf": on_behalf,
        "receiver": receiver,
    }


def build_supply_collateral_payload(
    market_params: tuple,
    assets: int,
    on_behalf: str,
    data: bytes = b"",
) -> Dict[str, Any]:
    """Build a ``supplyCollateral`` call payload for ``web3.py``."""
    return {
        "marketParams": market_params,
        "assets": assets,
        "onBehalf": on_behalf,
        "data": data,
    }


def build_withdraw_collateral_payload(
    market_params: tuple,
    assets: int,
    on_behalf: str,
    receiver: str,
) -> Dict[str, Any]:
    """Build a ``withdrawCollateral`` call payload for ``web3.py``."""
    return {
        "marketParams": market_params,
        "assets": assets,
        "onBehalf": on_behalf,
        "receiver": receiver,
    }
