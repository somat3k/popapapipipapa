"""Morpho Blue market registry and market ID computation.

Market ID = keccak256(abi.encode(MarketParams)), where MarketParams is
(loanToken, collateralToken, oracle, irm, lltv) packed as 5 × 32-byte words.

Default markets are loaded at import time from ``config/markets.json``
in the project root.

Reference: https://docs.morpho.org/morpho-blue/contracts/market-id
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Optional

from .contracts import (
    IRM_ADDRESS,
    ORACLE_ADDRESSES,
    TOKEN_ADDRESSES,
    TOKEN_DECIMALS,
)

logger = logging.getLogger(__name__)

_CONFIG_DIR = pathlib.Path(__file__).parent.parent / "config"

# Morpho Blue LLTV values (in WAD = 1e18 denominator)
# Common LLTVs: 62.5%, 77%, 86%, 91.5%, 94.5%, 96.5%, 98%
WAD = 10**18
LLTV_86 = int(0.86 * WAD)
LLTV_77 = int(0.77 * WAD)
LLTV_625 = int(0.625 * WAD)
LLTV_915 = int(0.915 * WAD)
LLTV_945 = int(0.945 * WAD)


def _address_to_bytes32(address: str) -> bytes:
    """Left-pad an Ethereum address to 32 bytes."""
    addr = address.lower().removeprefix("0x")
    return bytes.fromhex(addr.zfill(64))


def _uint256_to_bytes32(value: int) -> bytes:
    return value.to_bytes(32, "big")


def build_market_id(
    loan_token: str,
    collateral_token: str,
    oracle: str,
    irm: str,
    lltv: int,
) -> str:
    """Compute the Morpho Blue market ID (bytes32 as hex string).

    The ID is keccak256 of the ABI-encoded MarketParams struct, where
    each field is padded to 32 bytes.

    Parameters
    ----------
    loan_token:        address of the loan token
    collateral_token:  address of the collateral token
    oracle:            address of the price oracle
    irm:               address of the interest rate model
    lltv:              liquidation LTV in WAD (e.g. 860000000000000000 for 86%)

    Returns
    -------
    ``"0x" + 64 hex chars`` representing the bytes32 market ID.

    Notes
    -----
    For production use, install ``pysha3`` (``pip install pysha3``) or
    ``eth_hash`` to obtain the authentic Ethereum keccak256 hash.  When
    neither is available, a best-effort SHA-3 digest is returned — the
    resulting ID **will differ** from the on-chain ID and must not be used
    for live transactions.
    """
    encoded = (
        _address_to_bytes32(loan_token)
        + _address_to_bytes32(collateral_token)
        + _address_to_bytes32(oracle)
        + _address_to_bytes32(irm)
        + _uint256_to_bytes32(lltv)
    )
    # Prefer eth_hash (pysha3-backed) → pysha3 direct → fallback SHA-3
    try:
        from eth_hash.auto import keccak  # type: ignore
        digest = keccak(encoded)
    except ImportError:
        try:
            import sha3 as _sha3  # pysha3  # type: ignore
            digest = _sha3.keccak_256(encoded).digest()
        except ImportError:
            # Warning: Python's hashlib sha3_256 is NIST SHA-3, NOT Ethereum keccak256.
            # Install pysha3 for correct production market IDs.
            import warnings
            warnings.warn(
                "pysha3/eth_hash not installed — market IDs will NOT match on-chain values. "
                "Install pysha3: pip install pysha3",
                RuntimeWarning,
                stacklevel=2,
            )
            digest = hashlib.new("sha3_256", encoded).digest()
    return "0x" + digest.hex()


@dataclass
class MarketConfig:
    """Full description of a Morpho Blue market."""

    name: str
    loan_token_symbol: str
    collateral_token_symbol: str
    loan_token: str
    collateral_token: str
    oracle: str
    irm: str
    lltv: int  # WAD
    market_id: str = field(init=False)

    # Optional metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.market_id = build_market_id(
            self.loan_token,
            self.collateral_token,
            self.oracle,
            self.irm,
            self.lltv,
        )

    @property
    def lltv_pct(self) -> float:
        """LLTV as a percentage, e.g. 86.0."""
        return self.lltv / WAD * 100

    @property
    def loan_decimals(self) -> int:
        return TOKEN_DECIMALS.get(self.loan_token_symbol, 18)

    @property
    def collateral_decimals(self) -> int:
        return TOKEN_DECIMALS.get(self.collateral_token_symbol, 18)

    def to_params_tuple(self) -> tuple:
        """Return the (loanToken, collateralToken, oracle, irm, lltv) tuple
        expected by the Morpho Blue contract ABI.
        """
        return (
            self.loan_token,
            self.collateral_token,
            self.oracle,
            self.irm,
            self.lltv,
        )

    def __repr__(self) -> str:
        return (
            f"<MarketConfig {self.name} "
            f"loan={self.loan_token_symbol} "
            f"collateral={self.collateral_token_symbol} "
            f"lltv={self.lltv_pct:.1f}%>"
        )


class MarketRegistry:
    """Registry of known Morpho Blue markets on Polygon.

    Pre-populated with the primary USDC and USDT markets backed by WETH,
    WBTC, and WPOL collateral using Morpho's adaptive IRM.
    """

    def __init__(self) -> None:
        self._markets: dict[str, MarketConfig] = {}
        self._populate_defaults()

    def _populate_defaults(self) -> None:
        markets_path = _CONFIG_DIR / "markets.json"
        with open(markets_path, encoding="utf-8") as fh:
            market_defs = json.load(fh)

        for defn in market_defs:
            # Skip JSON comment-only entries (keys that all start with "_")
            if not any(k for k in defn if not k.startswith("_")):
                continue
            if "name" not in defn:
                continue
            lltv_pct = defn["lltv_pct"]
            lltv = int(lltv_pct / 100 * WAD)
            m = MarketConfig(
                name=defn["name"],
                loan_token_symbol=defn["loan_token_symbol"],
                collateral_token_symbol=defn["collateral_token_symbol"],
                loan_token=TOKEN_ADDRESSES[defn["loan_token_symbol"]],
                collateral_token=TOKEN_ADDRESSES[defn["collateral_token_symbol"]],
                oracle=ORACLE_ADDRESSES[defn["oracle_key"]],
                irm=IRM_ADDRESS,
                lltv=lltv,
                description=defn.get("description", ""),
                tags=defn.get("tags", []),
            )
            self._markets[m.name] = m
            logger.debug("Registered market: %s (id=%s)", m.name, m.market_id[:10])

    def register(self, market: MarketConfig) -> None:
        self._markets[market.name] = market
        logger.info("Registered custom market: %s", market.name)

    def get(self, name: str) -> Optional[MarketConfig]:
        return self._markets.get(name)

    def get_by_id(self, market_id: str) -> Optional[MarketConfig]:
        for m in self._markets.values():
            if m.market_id == market_id:
                return m
        return None

    def list_markets(self) -> list[MarketConfig]:
        return list(self._markets.values())

    def filter_by_tag(self, tag: str) -> list[MarketConfig]:
        return [m for m in self._markets.values() if tag in m.tags]

    def __len__(self) -> int:
        return len(self._markets)


# ---------------------------------------------------------------------------
# Swap route registry — loaded from config/swap_routes.json
# ---------------------------------------------------------------------------

def _load_swap_routes() -> dict:
    """Load swap route definitions from config/swap_routes.json."""
    path = _CONFIG_DIR / "swap_routes.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_swap_routes_cfg = _load_swap_routes()

# List of dicts: {from_token, to_token, slippage_pct, description}
COLLATERAL_SWAP_ROUTES: list[dict] = _swap_routes_cfg["collateral_swap_routes"]
BORROW_TOKEN_SWAP_ROUTES: list[dict] = _swap_routes_cfg["borrow_token_swap_routes"]


def get_collateral_swap_route(
    from_token: str, to_token: str
) -> Optional[dict]:
    """Return the swap route config for a collateral → loan token swap, or None."""
    for route in COLLATERAL_SWAP_ROUTES:
        if route["from_token"] == from_token and route["to_token"] == to_token:
            return route
    return None


def get_borrow_token_swap_route(
    from_token: str, to_token: str
) -> Optional[dict]:
    """Return the swap route config for a borrow token → token swap, or None."""
    for route in BORROW_TOKEN_SWAP_ROUTES:
        if route["from_token"] == from_token and route["to_token"] == to_token:
            return route
    return None


# Module-level default registry
DEFAULT_REGISTRY = MarketRegistry()
