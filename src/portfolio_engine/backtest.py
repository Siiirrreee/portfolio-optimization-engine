"""Walk-forward (out-of-sample) backtest.

At each rebalance date the optimizer only sees the trailing `lookback` days, then the weights
are held (drifting with prices) until the next rebalance. This avoids the look-ahead bias of
judging a portfolio on the same data used to build it. Transaction costs are charged on turnover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import equal_weight, estimate_inputs, max_sharpe, min_variance

STRATEGIES = ("Equal Weight", "Min Variance", "Max Sharpe")


def _target_weights(window: pd.DataFrame, strategy: str, rf: float, max_weight: float) -> np.ndarray:
    mu, cov = estimate_inputs(window)
    if strategy == "Equal Weight":
        return equal_weight(window.shape[1])
    if strategy == "Min Variance":
        return min_variance(cov.values, max_weight)
    if strategy == "Max Sharpe":
        return max_sharpe(mu.values, cov.values, rf, max_weight)
    raise ValueError(strategy)


def walk_forward(returns: pd.DataFrame, lookback: int = 252, rebalance: int = 63,
                 rf: float = 0.0, max_weight: float = 1.0, cost_bps: float = 10.0,
                 strategies=STRATEGIES) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Returns (daily out-of-sample strategy returns, {strategy: weights at each rebalance})."""
    if len(returns) <= lookback + rebalance:
        raise ValueError(f"Need more than {lookback + rebalance} days of data; have {len(returns)}.")
    R = returns.values
    dates = returns.index[lookback:]
    out = {s: np.zeros(len(dates)) for s in strategies}
    history = {s: [] for s in strategies}

    for s in strategies:
        w = np.zeros(R.shape[1])
        for k, t in enumerate(range(lookback, len(R))):
            cost = 0.0
            if (t - lookback) % rebalance == 0:
                target = _target_weights(returns.iloc[t - lookback:t], s, rf, max_weight)
                cost = np.abs(target - w).sum() * cost_bps / 10_000
                w = target
                history[s].append(pd.Series(w, index=returns.columns, name=returns.index[t]))
            r = R[t]
            port = float(w @ r)
            out[s][k] = port - cost
            w = w * (1 + r) / (1 + port)  # weights drift between rebalances

    weights = {s: pd.DataFrame(history[s]) for s in strategies}
    return pd.DataFrame(out, index=dates), weights
