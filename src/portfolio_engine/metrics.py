"""Performance and risk metrics for a daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    """Compound annual growth rate (CAGR)."""
    growth = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return float(growth ** (1 / years) - 1) if years > 0 else np.nan


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio using a constant annual risk-free rate."""
    excess = returns - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    var = aligned.iloc[:, 1].var(ddof=1)
    return float(aligned.cov().iloc[0, 1] / var) if var > 0 else np.nan


def summarize(returns: pd.Series, benchmark: pd.Series | None = None, rf: float = 0.0) -> dict:
    """All headline metrics for one strategy or asset."""
    cagr, mdd = annualized_return(returns), max_drawdown(returns)
    out = {
        "Annual Return": cagr,
        "Annual Volatility": annualized_volatility(returns),
        "Sharpe": sharpe_ratio(returns, rf),
        "Sortino": sortino_ratio(returns, rf),
        "Max Drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }
    if benchmark is not None:
        out["Beta"] = beta(returns, benchmark)
    return out


def summary_table(returns: pd.DataFrame, benchmark: pd.Series | None = None, rf: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({c: summarize(returns[c], benchmark, rf) for c in returns.columns}).T
