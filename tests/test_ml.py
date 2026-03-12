"""Tests for ML models, hyperparameter management, and training."""

import numpy as np
import pytest

from app.ml.models import (
    EquityHealthEnsembleModel,
    EnsembleModel,
    GradientBoostingModel,
    LinearRegressionModel,
    ModelRegistry,
    NeuralNetworkModel,
    RandomForestModel,
    _directional_accuracy,
    _mae,
    _rmse,
)
from app.ml.hyperparams import (
    BayesianOptimiser,
    HyperparamScheduler,
    HyperparamSpace,
    HyperparamSpec,
    LRScheduler,
    RandomSearch,
)
from app.ml.trainer import Trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_dataset():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.standard_normal((n, 5)).astype(np.float64)
    w = rng.standard_normal(5)
    y = X @ w + rng.standard_normal(n) * 0.05
    return X, y.astype(np.float64)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert _rmse(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 3.0])
    assert _rmse(y_true, y_pred) == pytest.approx(1.0)


def test_mae_known():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert _mae(y_true, y_pred) == pytest.approx(2/3, rel=1e-4)


def test_directional_accuracy_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _directional_accuracy(y, y) == pytest.approx(1.0)


def test_directional_accuracy_inverse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([3.0, 2.0, 1.0])
    assert _directional_accuracy(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LinearRegressionModel tests
# ---------------------------------------------------------------------------

def test_linear_regression_fits(simple_dataset):
    X, y = simple_dataset
    model = LinearRegressionModel(alpha=0.1)
    model.fit(X, y)
    assert model.is_trained


def test_linear_regression_predict_shape(simple_dataset):
    X, y = simple_dataset
    model = LinearRegressionModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_linear_regression_low_rmse(simple_dataset):
    X, y = simple_dataset
    model = LinearRegressionModel(alpha=0.001)
    model.fit(X, y)
    metrics = model.evaluate(X, y)
    assert metrics["rmse"] < 0.5


def test_linear_regression_not_trained_raises():
    model = LinearRegressionModel()
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict(np.ones((3, 5)))


# ---------------------------------------------------------------------------
# RandomForestModel tests
# ---------------------------------------------------------------------------

def test_random_forest_fits(simple_dataset):
    X, y = simple_dataset
    model = RandomForestModel(n_estimators=10)
    model.fit(X, y)
    assert model.is_trained


def test_random_forest_predict(simple_dataset):
    X, y = simple_dataset
    model = RandomForestModel(n_estimators=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_random_forest_feature_importances(simple_dataset):
    X, y = simple_dataset
    model = RandomForestModel(n_estimators=5)
    model.fit(X, y)
    imp = model.feature_importances()
    assert imp is not None
    assert len(imp) == X.shape[1]
    assert abs(sum(imp) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# GradientBoostingModel tests
# ---------------------------------------------------------------------------

def test_gradient_boosting_fits(simple_dataset):
    pytest.importorskip("xgboost")
    X, y = simple_dataset
    model = GradientBoostingModel(n_estimators=20)
    model.fit(X, y)
    assert model.is_trained


def test_gradient_boosting_predict(simple_dataset):
    pytest.importorskip("xgboost")
    X, y = simple_dataset
    model = GradientBoostingModel(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# NeuralNetworkModel tests
# ---------------------------------------------------------------------------

def test_neural_network_fits(simple_dataset):
    X, y = simple_dataset
    model = NeuralNetworkModel(hidden_layer_sizes=(16, 8), max_iter=100)
    model.fit(X, y)
    assert model.is_trained


def test_neural_network_predict(simple_dataset):
    X, y = simple_dataset
    model = NeuralNetworkModel(hidden_layer_sizes=(8,), max_iter=50)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# EnsembleModel tests
# ---------------------------------------------------------------------------

def test_ensemble_model(simple_dataset):
    X, y = simple_dataset
    m1 = LinearRegressionModel()
    m2 = RandomForestModel(n_estimators=5)
    ens = EnsembleModel([m1, m2])
    ens.fit(X, y)
    preds = ens.predict(X)
    assert len(preds) == len(X)


def test_ensemble_empty_raises():
    ens = EnsembleModel([])
    with pytest.raises(RuntimeError, match="No models"):
        ens.predict(np.ones((3, 5)))


# ---------------------------------------------------------------------------
# EquityHealthEnsembleModel tests
# ---------------------------------------------------------------------------

def test_equity_health_ensemble_weights(simple_dataset):
    X, y = simple_dataset
    m1 = LinearRegressionModel()
    m2 = RandomForestModel(n_estimators=5)
    ens = EquityHealthEnsembleModel([m1, m2])
    ens.fit(X, y)
    weights = ens.update_weights_from_equity(
        equity_curves=[
            np.array([100.0, 101.0, 103.0, 104.0]),
            np.array([100.0, 100.5, 100.0, 99.0]),
        ],
        health_factors=[np.array([2.1, 2.0, 1.9]), np.array([1.2, 1.1, 1.0])],
    )
    assert weights.shape == (2,)
    assert weights[0] > weights[1]
    preds = ens.predict(X)
    assert len(preds) == len(X)


def test_equity_health_ensemble_defaults_for_zero_scores(simple_dataset):
    X, y = simple_dataset
    m1 = LinearRegressionModel()
    m2 = RandomForestModel(n_estimators=5)
    ens = EquityHealthEnsembleModel([m1, m2])
    ens.fit(X, y)
    weights = ens.update_weights(
        returns=[np.array([]), np.array([])],
        health_factors=[np.array([]), np.array([])],
    )
    assert weights.shape == (2,)
    assert weights[0] == pytest.approx(weights[1])


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------

def test_model_registry_save_load(simple_dataset, tmp_path):
    X, y = simple_dataset
    model = LinearRegressionModel()
    model.fit(X, y)
    registry = ModelRegistry(base_dir=tmp_path / "models")
    path = registry.register(model, version="test")
    assert path.exists()
    loaded = ModelRegistry.load_from_path(path) if hasattr(ModelRegistry, "load_from_path") else model.__class__.load(path)
    assert loaded.is_trained


# ---------------------------------------------------------------------------
# LRScheduler tests
# ---------------------------------------------------------------------------

def test_lr_scheduler_warmup():
    sched = LRScheduler(base_lr=1e-3, warmup_steps=10, total_steps=100)
    lrs = [sched.step() for _ in range(10)]
    assert lrs[0] < lrs[-1]  # monotonically increasing during warmup


def test_lr_scheduler_annealing():
    sched = LRScheduler(base_lr=1e-3, warmup_steps=5, total_steps=100, min_lr=1e-6)
    # After warmup, LR should decrease
    for _ in range(6):
        sched.step()
    lr_early = sched.current_lr
    for _ in range(50):
        sched.step()
    lr_late = sched.current_lr
    assert lr_late < lr_early


def test_lr_scheduler_min_lr():
    sched = LRScheduler(base_lr=1e-3, warmup_steps=1, total_steps=10, min_lr=1e-5)
    for _ in range(1000):
        sched.step()
    assert sched.current_lr >= 1e-5


# ---------------------------------------------------------------------------
# HyperparamSpec tests
# ---------------------------------------------------------------------------

def test_hyperparam_spec_sample_in_range():
    spec = HyperparamSpec("lr", min_val=1e-4, max_val=1e-1, log_scale=True)
    for _ in range(50):
        val = spec.sample()
        assert 1e-4 <= val <= 1e-1


def test_hyperparam_spec_int():
    spec = HyperparamSpec("n", min_val=2.0, max_val=10.0, param_type="int")
    for _ in range(20):
        val = spec.sample()
        assert isinstance(val, int)
        assert 2 <= val <= 10


def test_hyperparam_spec_categorical():
    spec = HyperparamSpec("act", min_val=0, max_val=1, param_type="categorical",
                          choices=["relu", "tanh", "gelu"])
    for _ in range(20):
        assert spec.sample() in ["relu", "tanh", "gelu"]


def test_hyperparam_spec_clip():
    spec = HyperparamSpec("lr", min_val=1e-4, max_val=1e-1)
    assert spec.clip(1.0) == pytest.approx(1e-1)
    assert spec.clip(0.0) == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# HyperparamSpace tests
# ---------------------------------------------------------------------------

def test_hyperparam_space_sample():
    space = HyperparamSpace()
    space.add(HyperparamSpec("lr", 1e-4, 1e-1, log_scale=True))
    space.add(HyperparamSpec("n_estimators", 10, 200, param_type="int"))
    sample = space.sample()
    assert "lr" in sample
    assert "n_estimators" in sample
    assert isinstance(sample["n_estimators"], int)


def test_hyperparam_space_len():
    space = HyperparamSpace()
    space.add(HyperparamSpec("a", 0, 1))
    space.add(HyperparamSpec("b", 0, 1))
    assert len(space) == 2


# ---------------------------------------------------------------------------
# HyperparamScheduler tests
# ---------------------------------------------------------------------------

def test_hyperparam_scheduler_reduces_lr():
    initial = {"learning_rate": 0.1, "epochs": 20, "patience": 2}
    sched = HyperparamScheduler(initial, patience=2, factor=0.5)
    # Plateau for patience+1 epochs
    for epoch in range(5):
        sched.on_epoch_end(epoch, val_metric=1.0)  # no improvement
    params = sched.get_params()
    assert params["learning_rate"] < 0.1


def test_hyperparam_scheduler_improves_no_reduction():
    initial = {"learning_rate": 0.1, "epochs": 20}
    sched = HyperparamScheduler(initial, patience=3, factor=0.5)
    for epoch in range(5):
        sched.on_epoch_end(epoch, val_metric=float(5 - epoch))  # improving
    params = sched.get_params()
    assert params["learning_rate"] > 0.0


def test_hyperparam_scheduler_history():
    initial = {"learning_rate": 0.01, "epochs": 5}
    sched = HyperparamScheduler(initial)
    for i in range(5):
        sched.on_epoch_end(i, val_metric=1.0)
    assert len(sched.history()) == 5


# ---------------------------------------------------------------------------
# RandomSearch tests
# ---------------------------------------------------------------------------

def test_random_search_returns_best():
    space = HyperparamSpace()
    space.add(HyperparamSpec("x", -10.0, 10.0))
    rs = RandomSearch(space, n_trials=20)
    best_params, best_score = rs.run(lambda p: p["x"] ** 2)
    assert best_score >= 0.0
    assert abs(best_params["x"]) <= 10.0


def test_random_search_n_trials():
    space = HyperparamSpace()
    space.add(HyperparamSpec("x", 0.0, 1.0))
    rs = RandomSearch(space, n_trials=10)
    rs.run(lambda _: 0.0)
    assert len(rs.results()) == 10


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

def test_trainer_basic(simple_dataset, tmp_path):
    X, y = simple_dataset
    model = LinearRegressionModel(alpha=0.01)
    trainer = Trainer(
        model,
        hyperparams={"epochs": 5, "learning_rate": 0.01, "patience": 2},
        checkpoint_dir=tmp_path / "ckpt",
    )
    summary = trainer.train(X, y)
    assert "rmse" in summary
    assert summary["rmse"] >= 0.0


def test_trainer_progress_callback(simple_dataset, tmp_path):
    X, y = simple_dataset
    model = LinearRegressionModel()
    calls = []
    trainer = Trainer(
        model,
        hyperparams={"epochs": 3, "learning_rate": 0.01},
        progress_callback=lambda ep, total, m: calls.append(ep),
        checkpoint_dir=tmp_path / "ckpt2",
    )
    trainer.train(X, y)
    assert len(calls) == 3


def test_trainer_history(simple_dataset, tmp_path):
    X, y = simple_dataset
    model = LinearRegressionModel()
    trainer = Trainer(
        model,
        hyperparams={"epochs": 4},
        checkpoint_dir=tmp_path / "ckpt3",
    )
    trainer.train(X, y)
    hist = trainer.history()
    assert len(hist) == 4
    assert "epoch" in hist[0]
