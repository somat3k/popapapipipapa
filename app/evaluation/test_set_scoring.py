"""Run test-set scoring with data retrieval and connection checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from app.evaluation.data_loader import OHLCVLoader
from app.evaluation.rl_pipeline import MIN_TRAINING_BARS, IterationResult, RLPipeline
from app.evaluation.test_set_storage import TestSetScoreStore
from app.ml.models import BaseModel
from app.trading.algorithms import Bar

logger = logging.getLogger(__name__)


@dataclass
class ConnectionReport:
    status: str
    market_checked: str = ""
    error: str = ""


class MorphoClientProtocol(Protocol):
    def list_markets(self) -> Any:
        ...


def _check_connection(morpho_client: Optional[MorphoClientProtocol]) -> ConnectionReport:
    if morpho_client is None:
        return ConnectionReport(status="skipped")
    try:
        markets = morpho_client.list_markets()
        market_checked = ""
        if isinstance(markets, dict):
            if markets:
                market_checked = next(iter(markets))
                if hasattr(morpho_client, "get_position"):
                    morpho_client.get_position(market_checked)
            status = "connected" if markets else "no_markets"
        else:
            if markets:
                market = markets[0]
                market_checked = getattr(market, "name", "")
                if hasattr(morpho_client, "get_market_state") and market_checked:
                    morpho_client.get_market_state(market_checked)
                status = "connected"
            else:
                status = "no_markets"
        return ConnectionReport(status=status, market_checked=market_checked)
    except Exception as exc:
        logger.exception("Connection check failed.")
        return ConnectionReport(status="error", error=str(exc))


def _load_bars(
    *,
    data_loader: OHLCVLoader,
    symbol: str,
    data_source: str,
    bars_count: int,
    days: int,
    timeframe: str,
    csv_path: Optional[Path],
    seed: Optional[int],
) -> List[Bar]:
    if data_source == "api":
        return data_loader.fetch_from_api(symbol, days=days, timeframe=timeframe)
    if data_source == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required when data_source='csv'.")
        return data_loader.load_from_csv(csv_path)
    if data_source == "synthetic":
        resolved_seed = seed if seed is not None else 42
        return data_loader.generate_synthetic(n=bars_count, seed=resolved_seed, symbol=symbol)
    raise ValueError(f"Unsupported data_source '{data_source}'.")


def run_test_set_scoring(
    *,
    model: BaseModel,
    symbol: str,
    data_source: str = "synthetic",
    bars_count: int = 200,
    days: int = 180,
    timeframe: str = "1d",
    seed: Optional[int] = None,
    csv_path: Optional[Path] = None,
    bars: Optional[List[Bar]] = None,
    data_loader: Optional[OHLCVLoader] = None,
    morpho_client: Optional[MorphoClientProtocol] = None,
    store: Optional[TestSetScoreStore] = None,
    iterations: int = 3,
    patience: int = 2,
    progress_callback: Optional[Callable[[int, int, IterationResult], None]] = None,
) -> Dict[str, Any]:
    """Run the RL pipeline with test-set scoring and persistence."""
    loader = data_loader or OHLCVLoader()
    if bars is not None:
        resolved_bars = bars
    else:
        resolved_bars = _load_bars(
            data_loader=loader,
            symbol=symbol,
            data_source=data_source,
            bars_count=bars_count,
            days=days,
            timeframe=timeframe,
            csv_path=csv_path,
            seed=seed,
        )

    connection_report = _check_connection(morpho_client)
    run_id: Optional[int] = None

    if store is not None and len(resolved_bars) >= MIN_TRAINING_BARS:
        run_id = store.create_training_run(
            model_name=model.name,
            symbol=symbol,
            total_iterations=iterations,
            total_bars=len(resolved_bars),
            data_source=data_source,
            connection_status=connection_report.status,
        )

    def _progress(it: int, total: int, result: IterationResult) -> None:
        if store is not None and run_id is not None:
            store.record_iteration(run_id, result)
        if progress_callback:
            progress_callback(it, total, result)

    pipeline = RLPipeline(
        model=model,
        bars=resolved_bars,
        iterations=iterations,
        patience=patience,
        progress_callback=_progress,
    )
    results = pipeline.run()

    if store is not None and run_id is not None and "error" not in results:
        store.update_training_run(
            run_id,
            best_composite_score=results["best_composite_score"],
            best_letter_grade=results["best_letter_grade"],
        )

    return {
        "run_id": run_id,
        "connection": {
            "status": connection_report.status,
            "market_checked": connection_report.market_checked,
            "error": connection_report.error,
        },
        "data_source": data_source,
        "total_bars": len(resolved_bars),
        "pipeline": results,
    }
