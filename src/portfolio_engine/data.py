"""Price data loading: Yahoo Finance (via yfinance) with local CSV caching, plus an offline demo generator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "SPY", "QQQ", "TLT", "GLD", "VNQ"]
DEFAULT_BENCHMARK = "SPY"


def download_prices(tickers: list[str], start: str, end: str | None = None,
                    cache_path: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download split/dividend-adjusted daily close prices. Uses a CSV cache when available."""
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            print(f"Loaded cached prices from {cache_path}")
            return cached[tickers]

    import yfinance as yf  # imported lazily so tests/demo mode don't need network

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("No data returned from Yahoo Finance. Check tickers, dates, and your internet connection.")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices[tickers]

    missing = [t for t in tickers if prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"No data for: {missing}. Remove or replace these tickers.")

    prices = prices.dropna(how="any")  # common history only, avoids mismatched windows
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
        print(f"Saved prices to {cache_path}")
    return prices


def generate_demo_prices(tickers: list[str] | None = None, years: int = 6, seed: int = 42) -> pd.DataFrame:
    """Correlated geometric-Brownian-motion prices so the pipeline runs without internet access.
    Demo output is for testing only — never publish it as real results."""
    tickers = tickers or DEFAULT_TICKERS
    rng = np.random.default_rng(seed)
    n, days = len(tickers), 252 * years
    mu = rng.uniform(0.02, 0.18, n) / 252
    vol = rng.uniform(0.12, 0.35, n) / np.sqrt(252)
    market = rng.normal(0, 1, days)
    loadings = rng.uniform(0.2, 0.9, n)
    shocks = loadings[None, :] * market[:, None] + np.sqrt(1 - loadings**2)[None, :] * rng.normal(0, 1, (days, n))
    rets = mu + vol * shocks
    idx = pd.bdate_range("2019-01-02", periods=days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna(how="all")
