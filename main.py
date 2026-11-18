"""Run the full portfolio optimization pipeline.

Examples
--------
python main.py                                  # default 10-asset universe, 2018 to today
python main.py --tickers AAPL MSFT NVDA TLT GLD --start 2019-01-01 --max-weight 0.35
python main.py --demo                           # offline synthetic data (pipeline test only)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from portfolio_engine import backtest, data, metrics, optimizer, visuals  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantitative Portfolio Optimization Engine")
    p.add_argument("--tickers", nargs="+", default=data.DEFAULT_TICKERS)
    p.add_argument("--benchmark", default=data.DEFAULT_BENCHMARK)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--rf", type=float, default=0.04, help="Annual risk-free rate, e.g. 0.04 = 4%%")
    p.add_argument("--max-weight", type=float, default=0.30, help="Max weight per asset (long-only)")
    p.add_argument("--lookback", type=int, default=252, help="Trading days used to estimate each rebalance")
    p.add_argument("--rebalance", type=int, default=63, help="Trading days between rebalances (63 = quarterly)")
    p.add_argument("--cost-bps", type=float, default=10.0, help="Transaction cost per unit turnover, in basis points")
    p.add_argument("--refresh", action="store_true", help="Ignore cached prices and re-download")
    p.add_argument("--demo", action="store_true", help="Use synthetic prices (no internet needed)")
    return p.parse_args()


def pct(x: float) -> str:
    return f"{x:.1%}"


def main() -> None:
    a = parse_args()
    tickers = list(dict.fromkeys(a.tickers + [a.benchmark]))
    out_results, out_visuals = ROOT / "results", ROOT / "visuals"
    out_results.mkdir(exist_ok=True)
    out_visuals.mkdir(exist_ok=True)

    # 1. Data
    if a.demo:
        print("DEMO MODE: synthetic prices — do not publish these numbers.")
        prices = data.generate_demo_prices(tickers)
    else:
        prices = data.download_prices(tickers, a.start, a.end, ROOT / "data" / "prices.csv", a.refresh)
    returns = data.to_returns(prices)
    bench = returns[a.benchmark]
    universe = returns[a.tickers]
    print(f"Data: {len(universe.columns)} assets, {universe.index[0]:%Y-%m-%d} to {universe.index[-1]:%Y-%m-%d} "
          f"({len(universe)} trading days)")

    # 2. In-sample optimization (full history) — the textbook MPT view
    mu, cov = optimizer.estimate_inputs(universe)
    weights = optimizer.build_portfolios(universe, a.rf, a.max_weight)
    weights.to_csv(out_results / "optimal_weights.csv", float_format="%.4f")

    asset_stats = metrics.summary_table(universe, bench, a.rf)
    asset_stats.to_csv(out_results / "asset_metrics.csv", float_format="%.4f")

    points = {}
    for s in weights.columns:
        r, v, _ = optimizer.portfolio_stats(weights[s].values, mu.values, cov.values, a.rf)
        points[s] = (v, r)
    frontier = optimizer.efficient_frontier(mu.values, cov.values, max_weight=a.max_weight)
    cloud = optimizer.random_portfolios(mu.values, cov.values, rf=a.rf)
    asset_points = pd.DataFrame({"Return": mu, "Volatility": cov.values.diagonal() ** 0.5})

    in_sample = pd.DataFrame({s: universe.values @ weights[s].values for s in weights.columns}, index=universe.index)
    in_sample[f"Benchmark ({a.benchmark})"] = bench
    in_sample_stats = metrics.summary_table(in_sample, bench, a.rf)
    in_sample_stats.to_csv(out_results / "in_sample_metrics.csv", float_format="%.4f")

    # 3. Walk-forward out-of-sample backtest — the honest view
    oos, oos_weights = backtest.walk_forward(universe, a.lookback, a.rebalance, a.rf, a.max_weight, a.cost_bps)
    oos[f"Benchmark ({a.benchmark})"] = bench.loc[oos.index]
    oos_stats = metrics.summary_table(oos, bench.loc[oos.index], a.rf)
    oos_stats.to_csv(out_results / "backtest_metrics.csv", float_format="%.4f")
    oos.to_csv(out_results / "backtest_daily_returns.csv", float_format="%.6f")
    for s, w in oos_weights.items():
        w.to_csv(out_results / f"rebalance_weights_{s.lower().replace(' ', '_')}.csv", float_format="%.4f")

    # 4. Visuals
    visuals.frontier_chart(cloud, frontier, points, asset_points, out_visuals)
    visuals.weights_chart(weights, out_visuals)
    visuals.performance_charts(oos, out_visuals, " — walk-forward backtest, after costs")
    visuals.correlation_chart(universe, out_visuals)

    # 5. Written summary
    best = oos_stats.drop(index=[c for c in oos_stats.index if c.startswith("Benchmark")])["Sharpe"].idxmax()
    shallowest = oos_stats["Max Drawdown"].idxmax()
    ins_sh, oos_sh = in_sample_stats["Sharpe"], oos_stats["Sharpe"]
    top_ms = weights["Max Sharpe"].sort_values(ascending=False).head(3)
    lines = [
        "# Results Summary",
        "",
        f"_Generated {date.today():%Y-%m-%d}{' — DEMO DATA, NOT REAL' if a.demo else ''}_",
        "",
        f"- Universe: {', '.join(a.tickers)} | Benchmark: {a.benchmark}",
        f"- Period: {universe.index[0]:%Y-%m-%d} to {universe.index[-1]:%Y-%m-%d}",
        f"- Settings: risk-free {pct(a.rf)}, max weight {pct(a.max_weight)}, lookback {a.lookback}d, "
        f"rebalance every {a.rebalance}d, costs {a.cost_bps:g} bps",
        "",
        "## Out-of-sample backtest (after costs)",
        "",
        oos_stats.to_markdown(floatfmt=".3f"),
        "",
        "## In-sample (full-period optimization)",
        "",
        in_sample_stats.to_markdown(floatfmt=".3f"),
        "",
        "## Optimal weights (full period)",
        "",
        weights.to_markdown(floatfmt=".1%"),
        "",
        "## Key findings",
        "",
        f"- Best out-of-sample Sharpe: **{best}** ({oos_stats.loc[best, 'Sharpe']:.2f}).",
        f"- Shallowest max drawdown: **{shallowest}** ({pct(oos_stats.loc[shallowest, 'Max Drawdown'])}).",
        f"- Max Sharpe (full period) concentrates in {', '.join(f'{t} {pct(w)}' for t, w in top_ms.items())}.",
        "- Sharpe, in-sample → out-of-sample: "
        + ", ".join(f"{s} {ins_sh[s]:.2f} → {oos_sh[s]:.2f}" for s in weights.columns)
        + ". A big fall for Max Sharpe signals the optimizer fit noise in past returns "
        "(note: the backtest starts after the first lookback window, so periods differ slightly).",
    ]
    (out_results / "summary.md").write_text("\n".join(lines) + "\n")

    # 6. Refresh the Results section of README.md (between the RESULTS markers)
    if not a.demo:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if start in text and end in text:
            table = oos_stats[["Annual Return", "Annual Volatility", "Sharpe", "Max Drawdown", "Beta"]].copy()
            for col in ["Annual Return", "Annual Volatility", "Max Drawdown"]:
                table[col] = table[col].map(pct)
            for col in ["Sharpe", "Beta"]:
                table[col] = table[col].map(lambda x: f"{x:.2f}")
            block = "\n".join([
                start,
                f"_Out-of-sample walk-forward backtest, after {a.cost_bps:g} bps costs, "
                f"{oos.index[0]:%b %Y} to {oos.index[-1]:%b %Y}. Full detail: [results/summary.md](results/summary.md)._",
                "",
                table.to_markdown(),
                "",
                "**Key findings**",
                "",
                *lines[-4:],
                end,
            ])
            pre, rest = text.split(start, 1)
            readme.write_text(pre + block + rest.split(end, 1)[1], encoding="utf-8")
            print("Updated README.md results section")

    print("\nOut-of-sample backtest (after costs):")
    print(oos_stats.round(3).to_string())
    print(f"\nSaved results to {out_results} and charts to {out_visuals}")


if __name__ == "__main__":
    main()
