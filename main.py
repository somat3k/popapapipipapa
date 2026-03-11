#!/usr/bin/env python3
"""Multiplex Financials — DEFI AI Platform.

Entry point. Launches the main GUI window.

Usage
-----
    python main.py

CLI flags
---------
    --headless   Run without GUI (useful for CI / server environments)
    --version    Print version and exit
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="multiplex-financials",
        description="Multiplex Financials — DEFI AI Platform",
    )
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode (no GUI)")
    parser.add_argument("--version", action="store_true",
                        help="Print version and exit")
    return parser.parse_args(argv)


def run_headless() -> None:
    """Headless demo: run trading chain and print output."""
    import numpy as np
    from app.trading.algorithms import (
        Bar, DataIngestionStep, FeatureEngineeringStep,
        MeanReversionAlgo, MomentumAlgo, RiskFilterStep,
        SignalAggregator, SignalGenerationStep, TradingChain, TradingMetrics,
    )
    from app.defi.morpho import MorphoClient, MARKET_WMATIC_USDC
    from app.ml.models import LinearRegressionModel
    from app.ml.trainer import Trainer

    logger.info("=== Multiplex Financials — Headless Demo ===")

    # --- Trading chain demo ---
    rng = np.random.default_rng(0)
    prices = np.cumsum(rng.normal(0, 0.01, 100)) + 1.0
    prices = np.exp(prices) * 0.85
    bars = [
        Bar(float(i), prices[i], prices[i] * 1.005, prices[i] * 0.995,
            prices[i], float(rng.integers(100_000, 1_000_000)))
        for i in range(len(prices))
    ]
    chain = (
        TradingChain()
        .add_step(DataIngestionStep(bars))
        .add_step(FeatureEngineeringStep())
        .add_step(SignalGenerationStep(
            SignalAggregator([MeanReversionAlgo(), MomentumAlgo()])
        ))
        .add_step(RiskFilterStep())
    )
    result = chain.run({"portfolio_value": 100_000.0})
    sigs = result.get("filtered_signals", [])
    logger.info("Trading chain: %d bars → %d signals.", len(bars), len(sigs))

    returns = np.diff(prices) / prices[:-1]
    metrics = TradingMetrics(returns).summary()
    logger.info("Strategy metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    # --- DeFi demo ---
    client = MorphoClient()
    client.deposit_collateral(MARKET_WMATIC_USDC, 500.0)
    client.borrow(MARKET_WMATIC_USDC, 200.0)
    pos = client.get_position(MARKET_WMATIC_USDC)
    logger.info("DeFi position: collateral=%.2f borrow=%.2f HF=%.2f",
                pos.collateral, pos.borrow_shares, pos.health_factor)
    swap_result = client.collateral_swap(MARKET_WMATIC_USDC, 50.0, dry_run=True)
    logger.info("Collateral swap dry-run: %s", swap_result)

    # --- ML demo ---
    X = rng.standard_normal((200, 10)).astype(np.float64)
    y = X @ rng.standard_normal(10) + rng.standard_normal(200) * 0.1
    model = LinearRegressionModel(alpha=0.1)
    trainer = Trainer(model, hyperparams={"epochs": 10, "learning_rate": 0.01})
    summary = trainer.train(X, y)
    logger.info("ML training summary: %s", {k: round(float(v), 4)
                                            for k, v in summary.items()
                                            if isinstance(v, (int, float))})

    logger.info("=== Demo complete. ===")


def run_gui() -> None:
    try:
        import tkinter as tk
        tk.Tk().destroy()  # Verify tkinter is available
    except Exception as exc:
        logger.error("tkinter not available: %s", exc)
        logger.info("Falling back to headless mode.")
        run_headless()
        return

    from app.gui.main_window import launch
    launch()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.version:
        from app import VERSION, PLATFORM_NAME
        print(f"{PLATFORM_NAME} v{VERSION}")
        return 0

    if args.headless:
        run_headless()
    else:
        run_gui()

    return 0


if __name__ == "__main__":
    sys.exit(main())
