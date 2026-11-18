import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_engine import backtest, data, metrics, optimizer  # noqa: E402


@pytest.fixture(scope="module")
def returns():
    return data.to_returns(data.generate_demo_prices(["A", "B", "C", "D", "E"], years=3))


def test_max_drawdown_known_series():
    r = pd.Series([0.10, -0.50, 0.20])  # 1 -> 1.1 -> 0.55 -> 0.66
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_beta_of_benchmark_is_one(returns):
    assert metrics.beta(returns["A"], returns["A"]) == pytest.approx(1.0)


def test_weights_sum_to_one_and_respect_cap(returns):
    w = optimizer.build_portfolios(returns, rf=0.02, max_weight=0.4)
    assert np.allclose(w.sum(), 1.0)
    assert (w.values >= -1e-9).all() and (w.values <= 0.4 + 1e-6).all()


def test_min_variance_beats_equal_weight_on_volatility(returns):
    mu, cov = optimizer.estimate_inputs(returns)
    w_mv = optimizer.min_variance(cov.values)
    w_eq = optimizer.equal_weight(len(mu))
    assert w_mv @ cov.values @ w_mv <= w_eq @ cov.values @ w_eq + 1e-12


def test_max_sharpe_beats_random_portfolios(returns):
    mu, cov = optimizer.estimate_inputs(returns)
    w = optimizer.max_sharpe(mu.values, cov.values, rf=0.02)
    best = optimizer.portfolio_stats(w, mu.values, cov.values, 0.02)[2]
    cloud = optimizer.random_portfolios(mu.values, cov.values, n=2000, rf=0.02)
    assert best >= cloud["Sharpe"].max() - 1e-6


def test_infeasible_cap_raises(returns):
    _, cov = optimizer.estimate_inputs(returns)
    with pytest.raises(ValueError):
        optimizer.min_variance(cov.values, max_weight=0.1)  # 5 assets * 10% < 100%


def test_walk_forward_is_out_of_sample(returns):
    oos, w = backtest.walk_forward(returns, lookback=252, rebalance=63)
    assert oos.index[0] == returns.index[252]
    assert w["Max Sharpe"].index[0] == returns.index[252]
    assert np.allclose(w["Min Variance"].sum(axis=1), 1.0)
