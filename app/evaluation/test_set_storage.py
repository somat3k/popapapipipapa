"""SQLite storage for test-set scoring results."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.evaluation.rl_pipeline import IterationResult


@dataclass
class TrainingRunRecord:
    run_id: int
    model_name: str
    symbol: str
    total_iterations: int
    total_bars: int
    data_source: str
    connection_status: str
    best_composite_score: Optional[float]
    best_letter_grade: Optional[str]


class TestSetScoreStore:
    """Persist test-set scores to SQLite with a fixed schema."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    total_iterations INTEGER,
                    total_bars INTEGER,
                    data_source TEXT, -- api/csv/synthetic
                    connection_status TEXT, -- connected/no_markets/error/skipped
                    best_composite_score REAL, -- composite score (0-1 range)
                    best_letter_grade TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_set_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_run_id INTEGER NOT NULL,
                    iteration INTEGER NOT NULL,
                    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    composite_score REAL NOT NULL,
                    letter_grade TEXT NOT NULL,
                    sharpe REAL,
                    sortino REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    calmar REAL,
                    cagr REAL,
                    risk_adjusted_growth REAL,
                    defi_yield_contribution REAL,
                    hit_rate REAL,
                    health_factor_score REAL,
                    avg_health_factor REAL,
                    min_health_factor REAL,
                    recovery_speed REAL,
                    information_coefficient REAL,
                    hf_adjusted_sharpe REAL,
                    episode_return_pct REAL,
                    total_reward REAL,
                    elapsed_s REAL,
                    test_set_bars INTEGER,
                    FOREIGN KEY(training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_test_set_scores_run_iter
                ON test_set_scores(training_run_id, iteration)
                """
            )

    def create_training_run(
        self,
        *,
        model_name: str,
        symbol: str,
        total_iterations: int,
        total_bars: int,
        data_source: str,
        connection_status: str,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO training_runs (
                    model_name,
                    symbol,
                    total_iterations,
                    total_bars,
                    data_source,
                    connection_status
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    symbol,
                    total_iterations,
                    total_bars,
                    data_source,
                    connection_status,
                ),
            )
            return int(cursor.lastrowid)

    def update_training_run(
        self,
        run_id: int,
        *,
        best_composite_score: float,
        best_letter_grade: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET best_composite_score = ?, best_letter_grade = ?
                WHERE id = ?
                """,
                (best_composite_score, best_letter_grade, run_id),
            )

    def record_iteration(self, run_id: int, result: IterationResult) -> None:
        metrics = result.test_set_metrics or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO test_set_scores (
                    training_run_id,
                    iteration,
                    composite_score,
                    letter_grade,
                    sharpe,
                    sortino,
                    max_drawdown,
                    win_rate,
                    profit_factor,
                    calmar,
                    cagr,
                    risk_adjusted_growth,
                    defi_yield_contribution,
                    hit_rate,
                    health_factor_score,
                    avg_health_factor,
                    min_health_factor,
                    recovery_speed,
                    information_coefficient,
                    hf_adjusted_sharpe,
                    episode_return_pct,
                    total_reward,
                    elapsed_s,
                    test_set_bars
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result.iteration,
                    result.composite_score,
                    result.letter_grade,
                    metrics.get("sharpe"),
                    metrics.get("sortino"),
                    metrics.get("max_drawdown"),
                    metrics.get("win_rate"),
                    metrics.get("profit_factor"),
                    metrics.get("calmar"),
                    metrics.get("cagr"),
                    metrics.get("risk_adjusted_growth"),
                    metrics.get("defi_yield_contribution"),
                    metrics.get("hit_rate"),
                    metrics.get("health_factor_score"),
                    metrics.get("avg_health_factor"),
                    metrics.get("min_health_factor"),
                    metrics.get("recovery_speed"),
                    metrics.get("information_coefficient"),
                    metrics.get("hf_adjusted_sharpe"),
                    result.episode_return_pct,
                    result.episode_total_reward,
                    result.elapsed_s,
                    result.test_set_bars,
                ),
            )

    def fetch_training_run(self, run_id: int) -> Optional[TrainingRunRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    model_name,
                    symbol,
                    total_iterations,
                    total_bars,
                    data_source,
                    connection_status,
                    best_composite_score,
                    best_letter_grade
                FROM training_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return TrainingRunRecord(
                run_id=row["id"],
                model_name=row["model_name"],
                symbol=row["symbol"],
                total_iterations=row["total_iterations"],
                total_bars=row["total_bars"],
                data_source=row["data_source"],
                connection_status=row["connection_status"],
                best_composite_score=row["best_composite_score"],
                best_letter_grade=row["best_letter_grade"],
            )

    def fetch_iteration_scores(self, run_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM test_set_scores
                WHERE training_run_id = ?
                ORDER BY iteration
                """,
                (run_id,),
            ).fetchall()
            return [dict(row) for row in rows]
