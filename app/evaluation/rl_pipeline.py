"""Reinforcement-learning feedback pipeline built on supervised learning.

Architecture
------------
The pipeline treats the supervised ML model as a policy network and wraps
the trading + DeFi environment as an RL-style feedback loop:

  1. **Pre-train**: fit the ML model on historical OHLCV features (supervised).
  2. **Episode**: run the trained policy on a held-out episode, collecting
     (state, action, reward) tuples.
  3. **Reward**: reward is a composite signal combining:
       - Per-step portfolio return
       - DeFi health-factor bonus (reward staying in the safe zone)
       - Penalty for drawdown
       - Milestone bonus for hitting account growth targets
  4. **Replay**: enrich the training set with episodic experience and
     retrain — closing the feedback loop.
  5. **Iterate**: repeat until the composite score stops improving
     (patience-based early stopping) or the max number of iterations
     is reached.

Components
----------
:class:`RLEnvironment`
    Wraps bar data and a :class:`~app.defi.morpho.MorphoClient` as a
    step-wise environment.

:class:`SupervisedRLAgent`
    Policy backed by any :class:`~app.ml.models.BaseModel`.  Converts
    market features to actions (+1 buy, 0 hold, -1 sell) and updates
    itself using episodic replay.

:class:`RLPipeline`
    Orchestrates the full training loop.  Single entry-point for end-to-end
    training.

Usage
-----
::

    from app.evaluation.rl_pipeline import RLPipeline
    from app.evaluation.data_loader import OHLCVLoader
    from app.ml.models import RandomForestModel

    loader = OHLCVLoader()
    bars = loader.generate_synthetic(n=500, symbol="ETH")

    pipeline = RLPipeline(
        model=RandomForestModel(n_estimators=50),
        bars=bars,
        iterations=5,
    )
    result = pipeline.run()
    print(result["best_composite_score"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.ml.models import BaseModel
from app.trading.algorithms import (
    Bar,
    FeatureEngineeringStep,
    Signal,
    TradingMetrics,
    _ema,
    _rsi,
)
from app.evaluation.metrics import AgentEvaluationMetrics, AccountGrowthTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

REWARD_WIN = 1.0          # positive return
REWARD_SAFE_HF = 0.2      # health factor in safe zone
REWARD_MILESTONE = 2.0    # account milestone crossed
PENALTY_DRAWDOWN = -0.5   # per-period drawdown penalty
PENALTY_DANGER_HF = -1.0  # health factor in danger zone


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class EnvStep:
    """Single step result from the RL environment."""

    period: int
    state: np.ndarray
    action: int          # +1 buy, 0 hold, -1 sell
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class RLEnvironment:
    """Step-wise trading + DeFi environment for the RL pipeline.

    The environment manages a simulated portfolio that can:
      - Execute trading signals (long/short/hold)
      - Track a DeFi collateral position (simplified HF simulation)
      - Compute composite rewards combining trading PnL and DeFi metrics

    Parameters
    ----------
    bars:
        Historical OHLCV bars.
    initial_capital:
        Starting portfolio value (USD).
    defi_collateral_ratio:
        Fraction of capital allocated to DeFi collateral.
    target_hf:
        Target health factor maintained by the simulated DeFi position.
    morpho_client:
        Optional real :class:`~app.defi.morpho.MorphoClient`.  When None,
        a simplified HF simulation is used.
    """

    def __init__(
        self,
        bars: List[Bar],
        initial_capital: float = 10_000.0,
        defi_collateral_ratio: float = 0.50,
        target_hf: float = 1.80,
        morpho_client: Optional[Any] = None,
    ) -> None:
        self.bars = bars
        self.initial_capital = initial_capital
        self.defi_collateral_ratio = defi_collateral_ratio
        self.target_hf = target_hf
        self._morpho = morpho_client

        # Build feature matrix once
        self._features = self._build_features(bars)
        self._n = len(bars)

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to the beginning of the episode."""
        self._step = 0
        self._capital = self.initial_capital
        self._position = 0       # -1, 0, +1
        self._entry_price = 0.0
        self._equity: List[float] = [self.initial_capital]
        self._health_factors: List[float] = [self.target_hf]
        self._tracker = AccountGrowthTracker(self.initial_capital)
        return self._get_state(0)

    def step(self, action: int) -> EnvStep:
        """Execute *action* and advance one bar.

        Parameters
        ----------
        action:
            +1 = buy/long, 0 = hold, -1 = sell/short

        Returns
        -------
        EnvStep
            Step result with reward and next state.
        """
        i = self._step
        if i >= self._n - 1:
            return EnvStep(i, self._get_state(i), action, 0.0,
                           self._get_state(i), done=True)

        bar = self.bars[i]
        next_bar = self.bars[i + 1]

        # Trading PnL from action
        price_return = (next_bar.close - bar.close) / (bar.close + 1e-12)
        trade_return = price_return * action   # long: +return, short: -return

        # Update capital
        self._capital *= 1.0 + trade_return
        self._equity.append(self._capital)

        # DeFi health factor simulation (simplified)
        hf = self._simulate_hf(price_return)
        self._health_factors.append(hf)

        # Compute reward
        reward = self._compute_reward(trade_return, hf, i)

        # Update growth tracker
        self._tracker.update(i, trade_return)

        self._step += 1
        done = self._step >= self._n - 1
        next_state = self._get_state(self._step) if not done else self._get_state(i + 1)

        return EnvStep(
            period=i,
            state=self._get_state(i),
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info={
                "price_return": round(price_return, 6),
                "trade_return": round(trade_return, 6),
                "capital": round(self._capital, 2),
                "health_factor": round(hf, 4),
            },
        )

    @property
    def equity_curve(self) -> List[float]:
        return list(self._equity)

    @property
    def portfolio_returns(self) -> np.ndarray:
        eq = np.array(self._equity)
        if len(eq) < 2:
            return np.array([])
        return np.diff(eq) / (eq[:-1] + 1e-12)

    @property
    def health_factors(self) -> np.ndarray:
        return np.array(self._health_factors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, bars: List[Bar]) -> np.ndarray:
        """Build a feature matrix from OHLCV bars."""
        n = len(bars)
        if n == 0:
            return np.zeros((0, 8))

        step = FeatureEngineeringStep()
        ctx = step.process({"bars": bars})
        base_features = ctx.get("features", np.zeros((n, 5)))

        closes = np.array([b.close for b in bars])

        # Additional features: RSI, MACD signal, normalised volume
        # Guard against too-short series by falling back to zeros
        rsi_period = min(14, max(2, n - 1))
        ema_fast_period = min(12, max(2, n - 1))
        ema_slow_period = min(26, max(2, n - 1))

        rsi = _rsi(closes, rsi_period) if n > rsi_period else np.zeros(n)
        ema12 = _ema(closes, ema_fast_period)
        ema26 = _ema(closes, ema_slow_period)
        macd = ema12 - ema26
        vols = np.array([b.volume for b in bars])
        vol_norm = vols / (np.mean(vols) + 1e-12)

        extra = np.column_stack([
            np.nan_to_num(rsi / 100.0),
            np.nan_to_num(macd / (closes + 1e-12)),
            vol_norm,
        ])
        combined = np.column_stack([base_features, extra])
        # Replace NaNs with 0
        combined = np.nan_to_num(combined)
        return combined

    def _get_state(self, i: int) -> np.ndarray:
        if i >= len(self._features):
            return self._features[-1].copy()
        return self._features[i].copy()

    def _simulate_hf(self, price_return: float) -> float:
        """Simulate health-factor change based on collateral price movement."""
        # When price goes up (collateral appreciates), HF improves
        # When price goes down, HF degrades
        last_hf = self._health_factors[-1] if self._health_factors else self.target_hf
        # HF changes proportionally to collateral price change
        hf = last_hf * (1.0 + price_return * self.defi_collateral_ratio)
        return max(0.5, float(hf))  # floor at 0.5 to avoid numerical issues

    def _compute_reward(
        self, trade_return: float, hf: float, period: int
    ) -> float:
        """Composite reward function."""
        reward = 0.0

        # 1. Trading return signal
        if trade_return > 0:
            reward += REWARD_WIN * trade_return * 100
        else:
            reward += PENALTY_DRAWDOWN * abs(trade_return) * 100

        # 2. Health-factor bonus/penalty
        if hf >= 2.0:
            reward += REWARD_SAFE_HF
        elif hf < 1.0:
            reward += PENALTY_DANGER_HF

        # 3. Milestone bonus
        m = self._tracker.update(period, trade_return)
        if m is not None:
            reward += REWARD_MILESTONE

        return float(reward)


# ---------------------------------------------------------------------------
# Supervised RL Agent
# ---------------------------------------------------------------------------

class SupervisedRLAgent:
    """Policy agent using a supervised ML model as its action predictor.

    The agent:
      1. Trains on historical features → labels (direction of next bar).
      2. Runs episodes using the trained policy.
      3. Enriches the training set with experience replay.
      4. Retrains, closing the feedback loop.

    Parameters
    ----------
    model:
        Any :class:`~app.ml.models.BaseModel` instance.
    n_features:
        Expected number of input features (used to validate data).
    """

    def __init__(self, model: BaseModel, n_features: int = 8) -> None:
        self.model = model
        self.n_features = n_features
        self._experience: List[Tuple[np.ndarray, float]] = []

    def pretrain(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Supervised pre-training on historical data.

        Parameters
        ----------
        X:
            Feature matrix (n_samples × n_features).
        y:
            Target labels — typically next-bar direction (+1/−1) or return.

        Returns
        -------
        dict
            Training evaluation metrics.
        """
        self.model.fit(X, y)
        metrics = self.model.evaluate(X, y)
        logger.info("[RL-Agent] Pre-trained model '%s': %s", self.model.name, metrics)
        return metrics

    def run_episode(self, env: RLEnvironment) -> List[EnvStep]:
        """Run one full episode using the current policy.

        Parameters
        ----------
        env:
            A reset :class:`RLEnvironment`.

        Returns
        -------
        List[EnvStep]
            All steps from the episode.
        """
        state = env.reset()
        steps: List[EnvStep] = []
        done = False

        while not done:
            action = self._select_action(state)
            step_result = env.step(action)
            steps.append(step_result)
            # Store experience: (state, reward-weighted direction)
            self._experience.append((state, float(action) * max(0.0, step_result.reward)))
            state = step_result.next_state
            done = step_result.done

        return steps

    def update_policy(self, min_new_samples: int = 10) -> Optional[Dict[str, float]]:
        """Retrain the model using accumulated experience.

        Parameters
        ----------
        min_new_samples:
            Minimum number of experience samples required before retraining.

        Returns
        -------
        dict or None
            Evaluation metrics, or None if not enough data.
        """
        if len(self._experience) < min_new_samples:
            return None

        X_exp = np.array([s for s, _ in self._experience])
        y_exp = np.array([l for _, l in self._experience])

        if X_exp.ndim != 2 or X_exp.shape[1] == 0:
            return None

        self.model.fit(X_exp, y_exp)
        metrics = self.model.evaluate(X_exp, y_exp)
        logger.info("[RL-Agent] Policy updated on %d samples: %s", len(X_exp), metrics)
        return metrics

    def _select_action(self, state: np.ndarray) -> int:
        """Use the model to predict the action for a given state."""
        if not self.model.is_trained:
            # Random action before training
            return int(np.random.choice([-1, 0, 1]))
        pred = self.model.predict(state.reshape(1, -1))
        val = float(pred[0]) if len(pred) > 0 else 0.0
        if val > 0.01:
            return 1
        if val < -0.01:
            return -1
        return 0


# ---------------------------------------------------------------------------
# RL Pipeline
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """Summary of one RL training iteration."""

    iteration: int
    pretrain_metrics: Dict[str, float]
    episode_total_reward: float
    episode_return_pct: float
    composite_score: float
    letter_grade: str
    policy_update_metrics: Optional[Dict[str, float]]
    elapsed_s: float


class RLPipeline:
    """End-to-end RL feedback pipeline.

    Combines supervised pre-training, episodic experience collection, and
    iterative policy improvement using experience replay.

    Parameters
    ----------
    model:
        Supervised ML model used as the policy network.
    bars:
        Historical OHLCV bars (≥ 100 recommended).
    iterations:
        Maximum number of training iterations.
    patience:
        Stop early when composite score hasn't improved for this many
        iterations.
    initial_capital:
        Starting portfolio value.
    episode_bars:
        Number of bars per episode.  Defaults to 80% of total bars.
    progress_callback:
        Optional callable(iteration, total, result) for live reporting.
    """

    def __init__(
        self,
        model: BaseModel,
        bars: List[Bar],
        iterations: int = 10,
        patience: int = 3,
        initial_capital: float = 10_000.0,
        episode_bars: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, IterationResult], None]] = None,
    ) -> None:
        self.model = model
        self.bars = bars
        self.iterations = iterations
        self.patience = patience
        self.initial_capital = initial_capital
        self.episode_bars = episode_bars or max(50, int(len(bars) * 0.80))
        self.progress_callback = progress_callback

        self._agent: Optional[SupervisedRLAgent] = None
        self._history: List[IterationResult] = []
        self._best_score: float = -float("inf")
        self._best_iteration: int = 0

    def run(self) -> Dict[str, Any]:
        """Execute the full RL training pipeline.

        Returns
        -------
        dict
            Summary containing the best composite score, letter grade,
            full iteration history, and final model metrics.
        """
        logger.info(
            "[RLPipeline] Starting: bars=%d  iterations=%d  patience=%d",
            len(self.bars), self.iterations, self.patience,
        )

        # Early exit if there is not enough data to train
        if len(self.bars) < 20:
            return {"error": "Insufficient data for training (need ≥ 20 bars)."}

        # Build feature matrix and labels from the full bar dataset
        X, y = self._build_dataset(self.bars)

        # Initialise agent
        self._agent = SupervisedRLAgent(self.model, n_features=X.shape[1])

        no_improvement = 0
        for it in range(1, self.iterations + 1):
            t0 = time.perf_counter()

            # Step 1: Pre-train / re-train on current dataset
            pretrain_metrics = self._agent.pretrain(X, y)

            # Step 2: Run episode on held-out bars
            episode_bars = self.bars[-self.episode_bars:]
            env = RLEnvironment(episode_bars, self.initial_capital)
            steps = self._agent.run_episode(env)

            total_reward = sum(s.reward for s in steps)
            returns = env.portfolio_returns
            hf_arr = env.health_factors

            # Step 3: Compute composite score
            eval_metrics = AgentEvaluationMetrics(
                returns=returns if len(returns) > 0 else np.zeros(1),
                health_factors=hf_arr if len(hf_arr) > 0 else np.ones(1) * 2.0,
            )
            score = eval_metrics.composite_score()
            grade = eval_metrics.letter_grade()
            ep_return = (
                float(env.equity_curve[-1] / self.initial_capital - 1.0)
                if env.equity_curve
                else 0.0
            )

            # Step 4: Update policy using experience
            policy_metrics = self._agent.update_policy(min_new_samples=5)

            # Augment training data with experience
            if len(self._agent._experience) > 0:
                X, y = self._augment_dataset(X, y)

            elapsed = time.perf_counter() - t0

            result = IterationResult(
                iteration=it,
                pretrain_metrics=pretrain_metrics,
                episode_total_reward=round(total_reward, 4),
                episode_return_pct=round(ep_return * 100, 2),
                composite_score=round(score, 4),
                letter_grade=grade,
                policy_update_metrics=policy_metrics,
                elapsed_s=round(elapsed, 3),
            )
            self._history.append(result)

            logger.info(
                "[RLPipeline] Iter %d/%d  score=%.4f  grade=%s  return=%.2f%%  t=%.2fs",
                it, self.iterations, score, grade, ep_return * 100, elapsed,
            )

            if self.progress_callback:
                try:
                    self.progress_callback(it, self.iterations, result)
                except Exception:
                    logger.exception("[RLPipeline] progress_callback raised.")

            # Early stopping
            if score > self._best_score + 1e-4:
                self._best_score = score
                self._best_iteration = it
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    logger.info("[RLPipeline] Early stop at iter %d.", it)
                    break

        best_result = next(
            (r for r in self._history if r.iteration == self._best_iteration),
            self._history[-1] if self._history else None,
        )

        return {
            "best_composite_score": round(self._best_score, 4),
            "best_iteration": self._best_iteration,
            "best_letter_grade": best_result.letter_grade if best_result else "N/A",
            "best_episode_return_pct": best_result.episode_return_pct if best_result else 0.0,
            "total_iterations": len(self._history),
            "history": [
                {
                    "iteration": r.iteration,
                    "composite_score": r.composite_score,
                    "letter_grade": r.letter_grade,
                    "episode_return_pct": r.episode_return_pct,
                    "total_reward": r.episode_total_reward,
                    "elapsed_s": r.elapsed_s,
                }
                for r in self._history
            ],
        }

    def history(self) -> List[IterationResult]:
        """Return the full iteration history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_dataset(
        self, bars: List[Bar]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) dataset from bars for supervised pre-training.

        Labels are the sign of the next-bar return (1 = up, -1 = down).
        """
        env = RLEnvironment(bars, self.initial_capital)
        features = env._features
        closes = np.array([b.close for b in bars])

        # Next-bar return direction as label
        n = len(closes) - 1
        if n < 1:
            return np.zeros((1, features.shape[1])), np.zeros(1)

        y = np.sign(np.diff(closes[:n + 1]))   # +1 or -1
        X = features[:n]                        # current-bar features
        return X, y

    def _augment_dataset(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment the dataset with normalised experience replay samples."""
        if not self._agent or not self._agent._experience:
            return X, y

        # Take the most recent half of experience
        recent = self._agent._experience[-(len(self._agent._experience) // 2 + 1):]
        X_exp = np.array([s for s, _ in recent])
        y_exp = np.array([l for _, l in recent])

        # Align feature dimensions
        if X_exp.ndim == 2 and X_exp.shape[1] == X.shape[1]:
            X_aug = np.vstack([X, X_exp])
            y_aug = np.concatenate([y, y_exp])
            return X_aug, y_aug

        return X, y
