from __future__ import annotations

import numpy as np
import pytest

from app.evaluation.metrics import AgentEvaluationMetrics
from app.evaluation.rl_pipeline import IterationResult
from app.evaluation.test_set_scoring import run_test_set_scoring
from app.evaluation.test_set_storage import TestSetScoreStore
from app.ml.models import GradientBoostingModel, LinearRegressionModel, LSTMModel

# Expected deltas based on deterministic synthetic runs with tuned hyperparams.
BOOSTING_MIN_IMPROVEMENT = 0.2
LSTM_MAX_REGRESSION = 0.05


def _set_torch_seed():
    torch = pytest.importorskip("torch", reason="Torch is required for LSTM scoring.")
    torch.manual_seed(21)
    np.random.seed(21)


class DummyMorphoClient:
    def list_markets(self):
        return {"TEST": "0x" + "00" * 32}

    def get_position(self, market_id: str):
        return {"market_id": market_id}


def test_score_store_round_trip(tmp_path):
    returns = np.array([0.01, -0.005, 0.02, 0.0])
    hfs = np.array([1.8, 1.9, 2.1, 2.0])
    metrics = AgentEvaluationMetrics(returns=returns, health_factors=hfs).full_report()
    store = TestSetScoreStore(tmp_path / "scores.db")

    run_id = store.create_training_run(
        model_name="LinearRegression",
        symbol="ETH",
        total_iterations=1,
        total_bars=120,
        data_source="synthetic",
        connection_status="mock",
    )

    result = IterationResult(
        iteration=1,
        pretrain_metrics={},
        episode_total_reward=1.25,
        episode_return_pct=2.5,
        composite_score=metrics["composite_score"],
        letter_grade=metrics["letter_grade"],
        policy_update_metrics=None,
        elapsed_s=0.5,
        test_set_metrics=metrics,
        test_set_bars=40,
    )

    store.record_iteration(run_id, result)
    store.update_training_run(
        run_id,
        best_composite_score=result.composite_score,
        best_letter_grade=result.letter_grade,
    )

    run = store.fetch_training_run(run_id)
    scores = store.fetch_iteration_scores(run_id)

    assert run is not None
    assert run.model_name == "LinearRegression"
    assert run.best_letter_grade == result.letter_grade
    assert scores[0]["iteration"] == 1
    assert scores[0]["composite_score"] == pytest.approx(result.composite_score)
    assert scores[0]["sharpe"] == pytest.approx(metrics["sharpe"])


def test_run_test_set_scoring_stores_results(tmp_path):
    store = TestSetScoreStore(tmp_path / "scores.db")
    result = run_test_set_scoring(
        model=LinearRegressionModel(alpha=0.1),
        symbol="ETH",
        data_source="synthetic",
        bars_count=120,
        seed=7,
        morpho_client=DummyMorphoClient(),
        store=store,
        iterations=1,
        patience=1,
    )

    assert result["run_id"] is not None
    assert result["connection"]["status"] in {"connected", "no_markets"}
    run = store.fetch_training_run(result["run_id"])
    scores = store.fetch_iteration_scores(result["run_id"])

    assert run is not None
    assert run.total_bars == 120
    assert len(scores) == 1
    assert scores[0]["test_set_bars"] > 0


def _run_test_set_scoring(model, store=None):
    return run_test_set_scoring(
        model=model,
        symbol="ETH",
        data_source="synthetic",
        bars_count=160,
        seed=21,
        iterations=2,
        patience=1,
        store=store,
    )


def _assert_run_persisted(store, result, expected_bars, expected_iterations):
    run_id = result["run_id"]
    assert run_id is not None
    run = store.fetch_training_run(run_id)
    scores = store.fetch_iteration_scores(run_id)

    assert run is not None
    assert run.total_bars == expected_bars
    assert len(scores) == expected_iterations


def test_gradient_boosting_meets_performance_thresholds(tmp_path):
    store = TestSetScoreStore(tmp_path / "scores.db")
    baseline = _run_test_set_scoring(LinearRegressionModel(alpha=0.1), store=store)

    boosted = _run_test_set_scoring(
        GradientBoostingModel(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            random_state=21,
        ),
        store=store,
    )

    baseline_score = baseline["pipeline"]["best_composite_score"]
    assert (
        boosted["pipeline"]["best_composite_score"]
        >= baseline_score + BOOSTING_MIN_IMPROVEMENT
    )
    _assert_run_persisted(store, boosted, expected_bars=160, expected_iterations=2)


def test_lstm_meets_performance_thresholds(tmp_path):
    _set_torch_seed()
    store = TestSetScoreStore(tmp_path / "scores.db")
    baseline = _run_test_set_scoring(LinearRegressionModel(alpha=0.1), store=store)
    lstm = _run_test_set_scoring(
        LSTMModel(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            epochs=5,
            seq_len=10,
        ),
        store=store,
    )
    baseline_score = baseline["pipeline"]["best_composite_score"]
    assert (
        lstm["pipeline"]["best_composite_score"]
        >= baseline_score - LSTM_MAX_REGRESSION
    )
    _assert_run_persisted(store, lstm, expected_bars=160, expected_iterations=2)
