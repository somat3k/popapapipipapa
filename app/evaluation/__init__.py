"""Evaluation module for Multiplex Financials agents.

Sub-modules
-----------
data_loader
    Historical OHLCV data downloader (CoinGecko API + offline synthetic fallback).
timeframe_fusion
    Multi-timeframe signal aggregation and market-regime detection.
metrics
    Agent evaluation metrics: CAGR, risk-adjusted growth, DeFi yield
    contribution, health-factor scoring, composite grade.
rl_pipeline
    Reinforcement-learning feedback pipeline built on supervised learning:
    pre-train → episode → reward → replay → retrain.
realtime_inference
    Realtime bar-by-bar inference runner: streams bars through a trained
    model and exchanges inference payloads with agents via the MessageBus.
defi_strategy
    1/2=3 DeFi account growth strategy and health-factor management.
"""

from app.evaluation.data_loader import OHLCVLoader, resample_bars
from app.evaluation.timeframe_fusion import (
    FusedDecision,
    MarketRegimeDetector,
    MultiTimeframeAnalyzer,
    TimeframeFuser,
    TimeframeLayer,
)
from app.evaluation.metrics import (
    AccountGrowthTracker,
    AgentEvaluationMetrics,
    GrowthMilestone,
    health_factor_grade,
    hf_penalty,
)
from app.evaluation.rl_pipeline import (
    EnvStep,
    IterationResult,
    RLEnvironment,
    RLPipeline,
    SupervisedRLAgent,
)
from app.evaluation.realtime_inference import (
    INFERENCE_TOPIC,
    InferencePayload,
    InferenceSummary,
    RealtimeInferenceRunner,
)
from app.evaluation.defi_strategy import (
    HalfHalfThreeStrategy,
    HealthFactorManager,
    MarketEntryAdvisor,
    PositionStatus,
    StrategyAction,
)
from app.evaluation.test_set_scoring import run_test_set_scoring
from app.evaluation.test_set_storage import TestSetScoreStore, TrainingRunRecord

__all__ = [
    # data_loader
    "OHLCVLoader",
    "resample_bars",
    # timeframe_fusion
    "FusedDecision",
    "MarketRegimeDetector",
    "MultiTimeframeAnalyzer",
    "TimeframeFuser",
    "TimeframeLayer",
    # metrics
    "AccountGrowthTracker",
    "AgentEvaluationMetrics",
    "GrowthMilestone",
    "health_factor_grade",
    "hf_penalty",
    # rl_pipeline
    "EnvStep",
    "IterationResult",
    "RLEnvironment",
    "RLPipeline",
    "SupervisedRLAgent",
    # realtime_inference
    "INFERENCE_TOPIC",
    "InferencePayload",
    "InferenceSummary",
    "RealtimeInferenceRunner",
    # defi_strategy
    "HalfHalfThreeStrategy",
    "HealthFactorManager",
    "MarketEntryAdvisor",
    "PositionStatus",
    "StrategyAction",
    # test-set scoring
    "run_test_set_scoring",
    "TestSetScoreStore",
    "TrainingRunRecord",
]
