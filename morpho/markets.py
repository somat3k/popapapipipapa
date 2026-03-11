"""Morpho Blue market registry and market ID computation.

Market ID = keccak256(abi.encode(MarketParams)), where MarketParams is
(loanToken, collateralToken, oracle, irm, lltv) packed as 5 × 32-byte words.

Reference: https://docs.morpho.org/morpho-blue/contracts/market-id
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from .contracts import (
    IRM_ADDRESS,
    ORACLE_ADDRESSES,
    TOKEN_ADDRESSES,
    TOKEN_DECIMALS,
)

logger = logging.getLogger(__name__)

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
        default_markets = [
            # USDC.e loan markets
            MarketConfig(
                name="WETH/USDC_E-86",
                loan_token_symbol="USDC_E",
                collateral_token_symbol="WETH",
                loan_token=TOKEN_ADDRESSES["USDC_E"],
                collateral_token=TOKEN_ADDRESSES["WETH"],
                oracle=ORACLE_ADDRESSES["WETH_USD"],
                irm=IRM_ADDRESS,
                lltv=LLTV_86,
                description="WETH collateral, USDC.e loan, 86% LLTV",
                tags=["stable-borrow", "eth-collateral"],
            ),
            MarketConfig(
                name="WBTC/USDC_E-86",
                loan_token_symbol="USDC_E",
                collateral_token_symbol="WBTC",
                loan_token=TOKEN_ADDRESSES["USDC_E"],
                collateral_token=TOKEN_ADDRESSES["WBTC"],
                oracle=ORACLE_ADDRESSES["WBTC_USD"],
                irm=IRM_ADDRESS,
                lltv=LLTV_86,
                description="WBTC collateral, USDC.e loan, 86% LLTV",
                tags=["stable-borrow", "btc-collateral"],
            ),
            MarketConfig(
                name="WPOL/USDC_E-77",
                loan_token_symbol="USDC_E",
                collateral_token_symbol="WPOL",
                loan_token=TOKEN_ADDRESSES["USDC_E"],
                collateral_token=TOKEN_ADDRESSES["WPOL"],
                oracle=ORACLE_ADDRESSES["WPOL_USD"],
                irm=IRM_ADDRESS,
                lltv=LLTV_77,
                description="WPOL collateral, USDC.e loan, 77% LLTV",
                tags=["stable-borrow", "pol-collateral"],
            ),
            # Native USDC loan markets
            MarketConfig(
                name="WETH/USDC-86",
                loan_token_symbol="USDC",
                collateral_token_symbol="WETH",
                loan_token=TOKEN_ADDRESSES["USDC"],
                collateral_token=TOKEN_ADDRESSES["WETH"],
                oracle=ORACLE_ADDRESSES["WETH_USD"],
                irm=IRM_ADDRESS,
                lltv=LLTV_86,
                description="WETH collateral, native USDC loan, 86% LLTV",
                tags=["stable-borrow", "eth-collateral", "native-usdc"],
            ),
            # Yield-bearing collateral markets
            MarketConfig(
                name="stMATIC/USDC_E-625",
                loan_token_symbol="USDC_E",
                collateral_token_symbol="stMATIC",
                loan_token=TOKEN_ADDRESSES["USDC_E"],
                collateral_token=TOKEN_ADDRESSES["stMATIC"],
                oracle=ORACLE_ADDRESSES["WPOL_USD"],
                irm=IRM_ADDRESS,
                lltv=LLTV_625,
                description="stMATIC collateral, USDC.e loan, 62.5% LLTV",
                tags=["yield-collateral", "lsd"],
            ),
        ]
        for m in default_markets:
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


# Module-level default registry
DEFAULT_REGISTRY = MarketRegistry()
