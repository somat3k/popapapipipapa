"""Contract addresses and minimal ABIs for Morpho Blue on Polygon.

All addresses are checksummed.  Token decimals are listed for reference.
Values are loaded at import time from ``config/contracts.json`` and
``config/tokens.json`` in the project root.

Reference
---------
- Morpho Blue: https://docs.morpho.org/morpho-blue/contracts/addresses
- Polygon token list: https://polygonscan.com
"""

from __future__ import annotations

import json
import pathlib

# ---------------------------------------------------------------------------
# JSON config loader helpers
# ---------------------------------------------------------------------------

_CONFIG_DIR = pathlib.Path(__file__).parent.parent / "config"


def _load_json(filename: str) -> dict:
    """Load a JSON file from the project-level config directory."""
    path = _CONFIG_DIR / filename
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_contracts_cfg = _load_json("contracts.json")
_tokens_cfg = _load_json("tokens.json")

# ---------------------------------------------------------------------------
# Morpho Blue — deployed at the same address across EVM chains
# ---------------------------------------------------------------------------

MORPHO_BLUE_POLYGON: str = _contracts_cfg["morpho_blue_polygon"]

# ---------------------------------------------------------------------------
# Token addresses and decimals on Polygon POS mainnet (chain ID 137)
# Loaded from config/tokens.json
# ---------------------------------------------------------------------------

TOKEN_ADDRESSES: dict[str, str] = {
    sym: info["address"]
    for sym, info in _tokens_cfg.items()
    if not sym.startswith("_")
}

TOKEN_DECIMALS: dict[str, int] = {
    sym: info["decimals"]
    for sym, info in _tokens_cfg.items()
    if not sym.startswith("_")
}

# Tokens that are accepted as Morpho collateral (collateral: true in tokens.json)
COLLATERAL_TOKENS: dict[str, str] = {
    sym: info["address"]
    for sym, info in _tokens_cfg.items()
    if not sym.startswith("_") and info.get("collateral", False)
}

# ---------------------------------------------------------------------------
# Oracle addresses (Chainlink / Morpho Oracle v1.1 on Polygon)
# Loaded from config/contracts.json
# ---------------------------------------------------------------------------

ORACLE_ADDRESSES: dict[str, str] = _contracts_cfg["oracle_addresses"]

# ---------------------------------------------------------------------------
# Adaptive IRM (Interest Rate Model) — Morpho's default adaptive IRM
# ---------------------------------------------------------------------------

IRM_ADDRESS: str = _contracts_cfg["irm_address"]

# ---------------------------------------------------------------------------
# Morpho Blue minimal ABI (only the functions we need)
# ---------------------------------------------------------------------------

MORPHO_BLUE_ABI: list[dict] = [
    # --- Write functions ---
    {
        "name": "supply",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "marketParams",
                "type": "tuple",
                "components": [
                    {"name": "loanToken", "type": "address"},
                    {"name": "collateralToken", "type": "address"},
                    {"name": "oracle", "type": "address"},
                    {"name": "irm", "type": "address"},
                    {"name": "lltv", "type": "uint256"},
                ],
            },
            {"name": "assets", "type": "uint256"},
            {"name": "shares", "type": "uint256"},
            {"name": "onBehalf", "type": "address"},
            {"name": "data", "type": "bytes"},
        ],
        "outputs": [
            {"name": "assetsSupplied", "type": "uint256"},
            {"name": "sharesSupplied", "type": "uint256"},
        ],
    },
    {
        "name": "borrow",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "marketParams",
                "type": "tuple",
                "components": [
                    {"name": "loanToken", "type": "address"},
                    {"name": "collateralToken", "type": "address"},
                    {"name": "oracle", "type": "address"},
                    {"name": "irm", "type": "address"},
                    {"name": "lltv", "type": "uint256"},
                ],
            },
            {"name": "assets", "type": "uint256"},
            {"name": "shares", "type": "uint256"},
            {"name": "onBehalf", "type": "address"},
            {"name": "receiver", "type": "address"},
        ],
        "outputs": [
            {"name": "assetsBorrowed", "type": "uint256"},
            {"name": "sharesBorrowed", "type": "uint256"},
        ],
    },
    {
        "name": "repay",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "marketParams",
                "type": "tuple",
                "components": [
                    {"name": "loanToken", "type": "address"},
                    {"name": "collateralToken", "type": "address"},
                    {"name": "oracle", "type": "address"},
                    {"name": "irm", "type": "address"},
                    {"name": "lltv", "type": "uint256"},
                ],
            },
            {"name": "assets", "type": "uint256"},
            {"name": "shares", "type": "uint256"},
            {"name": "onBehalf", "type": "address"},
            {"name": "data", "type": "bytes"},
        ],
        "outputs": [
            {"name": "assetsRepaid", "type": "uint256"},
            {"name": "sharesRepaid", "type": "uint256"},
        ],
    },
    {
        "name": "withdraw",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "marketParams",
                "type": "tuple",
                "components": [
                    {"name": "loanToken", "type": "address"},
                    {"name": "collateralToken", "type": "address"},
                    {"name": "oracle", "type": "address"},
                    {"name": "irm", "type": "address"},
                    {"name": "lltv", "type": "uint256"},
                ],
            },
            {"name": "assets", "type": "uint256"},
            {"name": "shares", "type": "uint256"},
            {"name": "onBehalf", "type": "address"},
            {"name": "receiver", "type": "address"},
        ],
        "outputs": [
            {"name": "assetsWithdrawn", "type": "uint256"},
            {"name": "sharesWithdrawn", "type": "uint256"},
        ],
    },
    {
        "name": "supplyCollateral",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "marketParams",
                "type": "tuple",
                "components": [
                    {"name": "loanToken", "type": "address"},
                    {"name": "collateralToken", "type": "address"},
                    {"name": "oracle", "type": "address"},
                    {"name": "irm", "type": "address"},
                    {"name": "lltv", "type": "uint256"},
                ],
            },
            {"name": "assets", "type": "uint256"},
            {"name": "onBehalf", "type": "address"},
            {"name": "data", "type": "bytes"},
        ],
        "outputs": [],
    },
    {
        "name": "withdrawCollateral",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {
                "name": "marketParams",
                "type": "tuple",
                "components": [
                    {"name": "loanToken", "type": "address"},
                    {"name": "collateralToken", "type": "address"},
                    {"name": "oracle", "type": "address"},
                    {"name": "irm", "type": "address"},
                    {"name": "lltv", "type": "uint256"},
                ],
            },
            {"name": "assets", "type": "uint256"},
            {"name": "onBehalf", "type": "address"},
            {"name": "receiver", "type": "address"},
        ],
        "outputs": [],
    },
    # --- Read functions ---
    {
        "name": "market",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "id", "type": "bytes32"}],
        "outputs": [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "totalSupplyAssets", "type": "uint128"},
                    {"name": "totalSupplyShares", "type": "uint128"},
                    {"name": "totalBorrowAssets", "type": "uint128"},
                    {"name": "totalBorrowShares", "type": "uint128"},
                    {"name": "lastUpdate", "type": "uint128"},
                    {"name": "fee", "type": "uint128"},
                ],
            }
        ],
    },
    {
        "name": "position",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "id", "type": "bytes32"},
            {"name": "user", "type": "address"},
        ],
        "outputs": [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "supplyShares", "type": "uint256"},
                    {"name": "borrowShares", "type": "uint128"},
                    {"name": "collateral", "type": "uint128"},
                ],
            }
        ],
    },
    {
        "name": "idToMarketParams",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "id", "type": "bytes32"}],
        "outputs": [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "loanToken", "type": "address"},
                    {"name": "collateralToken", "type": "address"},
                    {"name": "oracle", "type": "address"},
                    {"name": "irm", "type": "address"},
                    {"name": "lltv", "type": "uint256"},
                ],
            }
        ],
    },
    # ERC-20 approve (for token approvals)
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
]

ERC20_ABI: list[dict] = [
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "decimals",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
    },
]

# Polygon mainnet chain ID — loaded from config/contracts.json
POLYGON_CHAIN_ID: int = _contracts_cfg["polygon_chain_id"]

# Public RPC endpoints for Polygon — loaded from config/contracts.json
POLYGON_RPC_URLS: list[str] = _contracts_cfg["polygon_rpc_urls"]
