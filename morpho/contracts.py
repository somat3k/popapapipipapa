"""Contract addresses and minimal ABIs for Morpho Blue on Polygon.

All addresses are checksummed.  Token decimals are listed for reference.

Reference
---------
- Morpho Blue: https://docs.morpho.org/morpho-blue/contracts/addresses
- Polygon token list: https://polygonscan.com
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Morpho Blue — deployed at the same address across EVM chains
# ---------------------------------------------------------------------------

MORPHO_BLUE_POLYGON: str = "0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb"

# ---------------------------------------------------------------------------
# Token addresses on Polygon POS mainnet (chain ID 137)
# ---------------------------------------------------------------------------

TOKEN_ADDRESSES: dict[str, str] = {
    # Bridged USDC (USDC.e)
    "USDC_E": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    # Native USDC
    "USDC": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    # Wrapped ETH
    "WETH": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",
    # Wrapped BTC
    "WBTC": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
    # Wrapped POL (formerly WMATIC)
    "WPOL": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
    # DAI
    "DAI": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
    # USDT
    "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
    # stMATIC (Lido)
    "stMATIC": "0x3A58a54C066FdC0f2D55FC9C89F0415C92eBf3C4",
    # MaticX (Stader)
    "MaticX": "0xfa68FB4628DFF1028CFEc22b4162FCcd0d45efb6",
}

TOKEN_DECIMALS: dict[str, int] = {
    "USDC_E": 6,
    "USDC": 6,
    "WETH": 18,
    "WBTC": 8,
    "WPOL": 18,
    "DAI": 18,
    "USDT": 6,
    "stMATIC": 18,
    "MaticX": 18,
}

# ---------------------------------------------------------------------------
# Oracle addresses (Chainlink / Morpho Oracle v1.1 on Polygon)
# ---------------------------------------------------------------------------

ORACLE_ADDRESSES: dict[str, str] = {
    # Chainlink MATIC/USD
    "WPOL_USD": "0xAB594600376Ec9fD91F8e885dADF0CE036862dE0",
    # Chainlink ETH/USD
    "WETH_USD": "0xF9680D99D6C9589e2a93a78A04A279e509205945",
    # Chainlink BTC/USD
    "WBTC_USD": "0xDE31F8bFBD8c84b5360CFACCa3539B938dd78ae6",
    # Chainlink USDC/USD
    "USDC_USD": "0xfE4A8cc5b5B2366C1B58Bea3858e81843581b2F7",
}

# ---------------------------------------------------------------------------
# Adaptive IRM (Interest Rate Model) — Morpho's default adaptive IRM
# ---------------------------------------------------------------------------

IRM_ADDRESS: str = "0x870aC11D48B15DB9a138Cf899d20F13F79Ba00BC"

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

# Polygon mainnet chain ID
POLYGON_CHAIN_ID: int = 137

# Public RPC endpoints for Polygon (ordered by reliability)
POLYGON_RPC_URLS: list[str] = [
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.network",
    "https://rpc-mainnet.maticvigil.com",
    "https://polygon.llamarpc.com",
]
