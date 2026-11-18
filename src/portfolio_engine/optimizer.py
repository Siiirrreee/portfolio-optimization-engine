"""Mean-variance (Modern Portfolio Theory) optimization: equal-weight, minimum-variance,
maximum-Sharpe, and the efficient frontier. Long-only with an optional per-asset cap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import TRADING_DAYS


def estimate_inputs(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Annualized expected returns (historical mean) and covariance matrix."""
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol if vol > 0 else np.nan


def _solve(objective, n: int, max_weight: float, extra_constraints=()) -> np.ndarray:
    if max_weight * n < 1 - 1e-9:
        raise ValueError(f"max_weight={max_weight} is infeasible for {n} assets (need max_weight >= {1/n:.3f}).")
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra_constraints]
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0.0, max_weight)] * n, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1 / n)


def min_variance(cov: np.ndarray, max_weight: float = 1.0) -> np.ndarray:
    return _solve(lambda w: w @ cov @ w, len(cov), max_weight)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, max_weight: float = 1.0) -> np.ndarray:
    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf) / vol
    return _solve(neg_sharpe, len(mu), max_weight)


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, points: int = 40, max_weight: float = 1.0) -> pd.DataFrame:
    """Minimum volatility for each target return between the min-variance portfolio and the best achievable return."""
    w_mv = min_variance(cov, max_weight)
    lo = float(w_mv @ mu)
    # highest achievable return under the weight cap: fill best assets up to the cap
    order, remaining, hi = np.argsort(mu)[::-1], 1.0, 0.0
    for i in order:
        take = min(max_weight, remaining)
        hi += take * mu[i]
        remaining -= take
        if remaining <= 1e-12:
            break
    rows = []
    for target in np.linspace(lo, hi * 0.999, points):
        try:
            w = _solve(lambda w: w @ cov @ w, len(mu), max_weight,
                       ({"type": "eq", "fun": lambda w, t=target: w @ mu - t},))
        except RuntimeError:
            continue
        r, v, _ = portfolio_stats(w, mu, cov)
        rows.append({"Return": r, "Volatility": v, **{f"w{i}": x for i, x in enumerate(w)}})
    return pd.DataFrame(rows)


def random_portfolios(mu: np.ndarray, cov: np.ndarray, n: int = 5000, rf: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Monte Carlo cloud of random long-only portfolios (for the frontier chart)."""
    w = np.random.default_rng(seed).dirichlet(np.ones(len(mu)), n)
    ret = w @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    return pd.DataFrame({"Return": ret, "Volatility": vol, "Sharpe": (ret - rf) / vol})


def build_portfolios(returns: pd.DataFrame, rf: float = 0.0, max_weight: float = 1.0) -> pd.DataFrame:
    """Weights for all three strategies, as a DataFrame (rows = tickers)."""
    mu, cov = estimate_inputs(returns)
    m, c = mu.values, cov.values
    return pd.DataFrame({
        "Equal Weight": equal_weight(len(m)),
        "Min Variance": min_variance(c, max_weight),
        "Max Sharpe": max_sharpe(m, c, rf, max_weight),
    }, index=returns.columns)
