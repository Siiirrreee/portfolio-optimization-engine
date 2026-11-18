"""Charts: interactive Plotly HTML (for exploring) and static PNGs (for README / LinkedIn)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .metrics import drawdown_series

COLORS = {"Equal Weight": "#6b7280", "Min Variance": "#2563eb", "Max Sharpe": "#d97706", "Benchmark": "#111827"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def _color(name: str) -> str:
    return COLORS.get(name, COLORS["Benchmark"] if "(" in name else "#059669")


def frontier_chart(cloud: pd.DataFrame, frontier: pd.DataFrame, points: dict[str, tuple[float, float]],
                   assets: pd.DataFrame, out_dir: Path) -> None:
    # Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cloud["Volatility"], y=cloud["Return"], mode="markers", name="Random portfolios",
                             marker=dict(size=3, color=cloud["Sharpe"], colorscale="Viridis", showscale=True,
                                         colorbar=dict(title="Sharpe")), opacity=0.5))
    fig.add_trace(go.Scatter(x=frontier["Volatility"], y=frontier["Return"], mode="lines",
                             name="Efficient frontier", line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=assets["Volatility"], y=assets["Return"], mode="markers+text", text=assets.index,
                             textposition="top center", name="Individual assets", marker=dict(size=8, color="#9ca3af")))
    for name, (v, r) in points.items():
        fig.add_trace(go.Scatter(x=[v], y=[r], mode="markers", name=name,
                                 marker=dict(size=16, symbol="star", color=_color(name), line=dict(width=1, color="white"))))
    fig.update_layout(title="Efficient Frontier (in-sample)", xaxis_title="Annual volatility",
                      yaxis_title="Annual expected return", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      template="plotly_white", height=600)
    fig.write_html(out_dir / "efficient_frontier.html", include_plotlyjs="cdn")

    # PNG
    fig2, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(cloud["Volatility"], cloud["Return"], c=cloud["Sharpe"], cmap="viridis", s=4, alpha=0.4)
    fig2.colorbar(sc, label="Sharpe ratio")
    ax.plot(frontier["Volatility"], frontier["Return"], color="black", lw=2.5, label="Efficient frontier")
    ax.scatter(assets["Volatility"], assets["Return"], color="#9ca3af", s=25, zorder=3)
    for t, row in assets.iterrows():
        ax.annotate(t, (row["Volatility"], row["Return"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    for name, (v, r) in points.items():
        ax.scatter(v, r, marker="*", s=320, color=_color(name), edgecolor="white", zorder=4, label=name)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set(xlabel="Annual volatility", ylabel="Annual expected return", title="Efficient Frontier (in-sample)")
    ax.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / "efficient_frontier.png")
    plt.close(fig2)


def weights_chart(weights: pd.DataFrame, out_dir: Path) -> None:
    long = weights.reset_index(names="Ticker").melt(id_vars="Ticker", var_name="Strategy", value_name="Weight")
    fig = px.bar(long, x="Ticker", y="Weight", color="Strategy", barmode="group", color_discrete_map=COLORS,
                 title="Portfolio Weights by Strategy", template="plotly_white")
    fig.update_layout(yaxis_tickformat=".0%")
    fig.write_html(out_dir / "weights.html", include_plotlyjs="cdn")

    ax = weights.plot.bar(figsize=(10, 5), color=[_color(c) for c in weights.columns], width=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set(title="Portfolio Weights by Strategy", xlabel="", ylabel="Weight")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "weights.png")
    plt.close()


def performance_charts(strategy_returns: pd.DataFrame, out_dir: Path, title_suffix: str = "") -> None:
    wealth = (1 + strategy_returns).cumprod()
    dd = strategy_returns.apply(drawdown_series)

    fig = go.Figure()
    for c in wealth.columns:
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth[c], name=c, line=dict(color=_color(c))))
    fig.update_layout(title=f"Growth of $1{title_suffix}", template="plotly_white", yaxis_tickprefix="$", height=500)
    fig.write_html(out_dir / "cumulative_returns.html", include_plotlyjs="cdn")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for c in wealth.columns:
        axes[0].plot(wealth.index, wealth[c], label=c, color=_color(c), lw=1.6)
        axes[1].plot(dd.index, dd[c], color=_color(c), lw=1.2)
    axes[0].set(title=f"Growth of $1{title_suffix}", ylabel="Portfolio value ($)")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Drawdown", ylabel="From peak")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_returns_drawdown.png")
    plt.close()


def correlation_chart(returns: pd.DataFrame, out_dir: Path) -> None:
    corr = returns.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation of Daily Returns", template="plotly_white")
    fig.write_html(out_dir / "correlation.html", include_plotlyjs="cdn")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iat[i, j]) > 0.6 else "black")
    ax.grid(False)
    fig.colorbar(im, fraction=0.046)
    ax.set_title("Correlation of Daily Returns")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation.png")
    plt.close()
