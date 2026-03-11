"""Polygon blockchain tools for agent use.

Each function is designed to be registered with ToolRegistry so agents
can call them via ``agent.use_tool("polygon.<name>", ...)``.

All functions degrade gracefully when web3 is not installed or when no
RPC is configured — they return structured error dicts rather than raising.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider helper (cached per RPC URL)
# ---------------------------------------------------------------------------

_w3_cache: dict[str, Any] = {}
_w3_lock = threading.Lock()


def _get_web3(rpc_url: Optional[str] = None) -> Any:
    """Return a cached Web3 instance for *rpc_url*, or None if unavailable."""
    url = rpc_url or "https://polygon-rpc.com"
    with _w3_lock:
        if url in _w3_cache:
            return _w3_cache[url]
    try:
        from web3 import Web3  # type: ignore

        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 10}))
        with _w3_lock:
            _w3_cache[url] = w3
        return w3
    except ImportError:
        logger.debug("web3 not installed — Polygon tools in offline mode.")
        return None
    except Exception as exc:
        logger.warning("Web3 connection failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tool: get_block_number
# ---------------------------------------------------------------------------

def get_block_number(rpc_url: Optional[str] = None) -> dict[str, Any]:
    """Return the current Polygon block number.

    Returns
    -------
    {"block_number": int} or {"error": str}
    """
    w3 = _get_web3(rpc_url)
    if w3 is None:
        return {"error": "web3 unavailable", "block_number": -1}
    try:
        return {"block_number": w3.eth.block_number}
    except Exception as exc:
        return {"error": str(exc), "block_number": -1}


# ---------------------------------------------------------------------------
# Tool: get_gas_price
# ---------------------------------------------------------------------------

def get_gas_price(rpc_url: Optional[str] = None) -> dict[str, Any]:
    """Return current gas price in Gwei and Wei.

    Returns
    -------
    {"gas_price_gwei": float, "gas_price_wei": int} or {"error": str}
    """
    w3 = _get_web3(rpc_url)
    if w3 is None:
        return {"error": "web3 unavailable", "gas_price_gwei": -1.0, "gas_price_wei": -1}
    try:
        fee_history = w3.eth.fee_history(1, "latest", [50])
        base_fee_wei = fee_history["baseFeePerGas"][-1]
        return {
            "gas_price_gwei": round(base_fee_wei / 1e9, 4),
            "gas_price_wei": base_fee_wei,
        }
    except Exception as exc:
        return {"error": str(exc), "gas_price_gwei": -1.0, "gas_price_wei": -1}


# ---------------------------------------------------------------------------
# Tool: get_token_balance
# ---------------------------------------------------------------------------

_ERC20_BALANCE_ABI = [
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
    {
        "name": "symbol",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
    },
]


def get_token_balance(
    token_address: str,
    wallet_address: str,
    rpc_url: Optional[str] = None,
) -> dict[str, Any]:
    """Return the ERC-20 token balance of *wallet_address*.

    Returns
    -------
    {"symbol": str, "balance_raw": int, "balance": float, "decimals": int}
    """
    w3 = _get_web3(rpc_url)
    if w3 is None:
        return {"error": "web3 unavailable", "balance": -1.0}
    try:
        contract = w3.eth.contract(
            address=w3.to_checksum_address(token_address), abi=_ERC20_BALANCE_ABI
        )
        raw = contract.functions.balanceOf(w3.to_checksum_address(wallet_address)).call()
        decimals = contract.functions.decimals().call()
        symbol = contract.functions.symbol().call()
        return {
            "symbol": symbol,
            "balance_raw": raw,
            "balance": raw / (10**decimals),
            "decimals": decimals,
        }
    except Exception as exc:
        return {"error": str(exc), "balance": -1.0}


# ---------------------------------------------------------------------------
# Tool: get_matic_balance
# ---------------------------------------------------------------------------

def get_matic_balance(
    wallet_address: str,
    rpc_url: Optional[str] = None,
) -> dict[str, Any]:
    """Return the native POL/MATIC balance of *wallet_address*.

    Returns
    -------
    {"balance_wei": int, "balance_pol": float}
    """
    w3 = _get_web3(rpc_url)
    if w3 is None:
        return {"error": "web3 unavailable", "balance_pol": -1.0}
    try:
        wei = w3.eth.get_balance(w3.to_checksum_address(wallet_address))
        return {"balance_wei": wei, "balance_pol": wei / 1e18}
    except Exception as exc:
        return {"error": str(exc), "balance_pol": -1.0}


# ---------------------------------------------------------------------------
# Tool: get_transaction_receipt
# ---------------------------------------------------------------------------

def get_transaction_receipt(
    tx_hash: str,
    rpc_url: Optional[str] = None,
) -> dict[str, Any]:
    """Fetch and decode a Polygon transaction receipt.

    Returns
    -------
    {"success": bool, "gas_used": int, "block_number": int, ...}
    """
    w3 = _get_web3(rpc_url)
    if w3 is None:
        return {"error": "web3 unavailable"}
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        if receipt is None:
            return {"error": "pending or not found"}
        return {
            "success": receipt["status"] == 1,
            "gas_used": receipt["gasUsed"],
            "block_number": receipt["blockNumber"],
            "from": receipt["from"],
            "to": receipt["to"],
            "logs_count": len(receipt["logs"]),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool: estimate_gas
# ---------------------------------------------------------------------------

def estimate_gas(
    from_address: str,
    to_address: str,
    data: str = "0x",
    value_wei: int = 0,
    rpc_url: Optional[str] = None,
) -> dict[str, Any]:
    """Estimate gas for a transaction on Polygon.

    Returns
    -------
    {"gas_estimate": int, "gas_price_gwei": float, "cost_pol": float}
    """
    w3 = _get_web3(rpc_url)
    if w3 is None:
        return {"error": "web3 unavailable", "gas_estimate": -1}
    try:
        gas = w3.eth.estimate_gas(
            {
                "from": w3.to_checksum_address(from_address),
                "to": w3.to_checksum_address(to_address),
                "data": data,
                "value": value_wei,
            }
        )
        gas_price_result = get_gas_price(rpc_url)
        gas_price_wei = gas_price_result.get("gas_price_wei", int(50e9))
        return {
            "gas_estimate": gas,
            "gas_price_gwei": gas_price_result.get("gas_price_gwei", -1.0),
            "cost_wei": gas * gas_price_wei,
            "cost_pol": gas * gas_price_wei / 1e18,
        }
    except Exception as exc:
        return {"error": str(exc), "gas_estimate": -1}
