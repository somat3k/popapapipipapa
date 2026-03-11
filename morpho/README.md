# Morpho Blue — Polygon Integration

Standalone, production-grade Python package for interacting with
[Morpho Blue](https://docs.morpho.org/morpho-blue/overview) on Polygon.

---

## Package Structure

```
morpho/
├── __init__.py       — Public API exports
├── contracts.py      — Contract addresses, token addresses, ABIs
├── markets.py        — Market configurations and ID computation
├── client.py         — Full MorphoBlueClient (query + agent payload)
├── growth.py         — Sequential self-funding growth engine
└── simulation.py     — Position simulation and projections
```

---

## Quick Start

```python
from morpho import MorphoBlueClient, GrowthEngine, PositionSimulator

# Mock mode (no real RPC needed)
client = MorphoBlueClient(wallet_address="0xYourAddress")

# Query communication — read on-chain state
pos = client.get_position("WETH/USDC_E-86")
print(f"Health factor: {pos.health_factor:.2f}")
print(f"Supply APY: {pos.supply_apy * 100:.2f}%")

# Agent payload communication — write transactions
result = client.supply_collateral("WETH/USDC_E-86", assets=1_000_000_000_000_000_000)  # 1 WETH
if result:
    print(f"TX: {result.tx_hash}")

# Growth engine — sequential self-funding
engine = GrowthEngine(client, target_ltv=0.50)
report = engine.run_growth_cycle(
    "WETH/USDC_E-86",
    collateral_assets=1_000_000_000_000_000_000,  # 1 WETH in wei
    dry_run=True,
)
print(report.summary())
print(f"Growth grade: {engine.growth_grade}")
```

---

## Production Deployment

```python
from web3 import Web3
from morpho import MorphoBlueClient, GrowthEngine

# Connect to Polygon via a real RPC
w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
assert w3.is_connected()

client = MorphoBlueClient(
    wallet_address="0xYourChecksummedAddress",
    private_key="0xYourPrivateKey",   # store in env variable, not source
    web3=w3,
)

engine = GrowthEngine(client, target_ltv=0.50, min_health_factor=1.20)

# Execute one growth cycle with 1 WETH collateral
report = engine.run_growth_cycle(
    "WETH/USDC_E-86",
    collateral_assets=1_000_000_000_000_000_000,
    collateral_token_symbol="WETH",
)
print(report.summary())
```

---

## Sequential Self-Funding Growth Process

The `GrowthEngine` implements the following ordered steps, each with a
`(+)` growth potential grade:

| Step | Action                          | Growth Grade |
|------|---------------------------------|--------------|
| 1    | Approve collateral token        | prerequisite |
| 2    | Supply collateral               | **+**        |
| 3    | Compute safe borrow amount      | calculation  |
| 4    | Borrow loan tokens              | **++**       |
| 5    | Re-supply borrowed tokens       | **+++**      |
| 6    | Monitor health factor           | safety       |
| 7    | Auto-repay if HF < threshold    | protection   |

The cycle can be repeated with `engine.run_growth_cycle(...)` to
compound yield over time.  Use `engine.monitor_and_rebalance()` on a
periodic schedule (cron / asyncio) to maintain health.

### Growth Grades

| Grade | Condition                              |
|-------|----------------------------------------|
| A+    | HF > 2.0 **and** supply APY > 5%       |
| A     | HF > 1.5                               |
| B     | HF > 1.2                               |
| C     | HF > 1.0 (at risk)                     |
| D     | HF ≤ 1.0 (liquidatable — DANGER)       |

---

## Markets

Pre-registered markets on Polygon (chain ID 137):

| Name                | Collateral | Loan    | LLTV   |
|---------------------|------------|---------|--------|
| WETH/USDC_E-86      | WETH       | USDC.e  | 86%    |
| WBTC/USDC_E-86      | WBTC       | USDC.e  | 86%    |
| WPOL/USDC_E-77      | WPOL       | USDC.e  | 77%    |
| WETH/USDC-86        | WETH       | USDC    | 86%    |
| stMATIC/USDC_E-625  | stMATIC    | USDC.e  | 62.5%  |

---

## Security Notes

- **Never hard-code private keys.** Use environment variables or a secrets
  manager (`python-dotenv`, AWS Secrets Manager, HashiCorp Vault).
- Always run `dry_run=True` first to verify the transaction before
  broadcasting on mainnet.
- Monitor `min_health_factor`; the default is 1.20.  Dropping below 1.0
  results in liquidation.
- Morpho Blue contract: `0xBBBBBbbBBb9cC5e90e3b3Af64bdAF62C37EEFFCb`

---

## References

- [Morpho Blue Docs](https://docs.morpho.org/morpho-blue/overview)
- [Morpho Blue GitHub](https://github.com/morpho-org/morpho-blue)
- [Polygon PoS Network](https://polygon.technology)
