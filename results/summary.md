# Results Summary

_Generated 2026-11-18_

- Universe: AAPL, MSFT, JPM, XOM, JNJ, SPY, QQQ, TLT, GLD, VNQ | Benchmark: SPY
- Period: 2018-01-03 to 2026-11-17
- Settings: risk-free 4.0%, max weight 30.0%, lookback 252d, rebalance every 63d, costs 10 bps

## Out-of-sample backtest (after costs)

|                 |   Annual Return |   Annual Volatility |   Sharpe |   Sortino |   Max Drawdown |   Calmar |   Beta |
|:----------------|----------------:|--------------------:|---------:|----------:|---------------:|---------:|-------:|
| Equal Weight    |           0.185 |               0.149 |    0.944 |     1.341 |         -0.262 |    0.706 |  0.733 |
| Min Variance    |           0.100 |               0.100 |    0.600 |     0.832 |         -0.199 |    0.501 |  0.311 |
| Max Sharpe      |           0.185 |               0.145 |    0.971 |     1.368 |         -0.169 |    1.100 |  0.538 |
| Benchmark (SPY) |           0.167 |               0.192 |    0.692 |     0.971 |         -0.337 |    0.495 |  1.000 |

## In-sample (full-period optimization)

|                 |   Annual Return |   Annual Volatility |   Sharpe |   Sortino |   Max Drawdown |   Calmar |   Beta |
|:----------------|----------------:|--------------------:|---------:|----------:|---------------:|---------:|-------:|
| Equal Weight    |           0.156 |               0.152 |    0.770 |     1.088 |         -0.275 |    0.568 |  0.759 |
| Min Variance    |           0.094 |               0.098 |    0.556 |     0.786 |         -0.158 |    0.593 |  0.303 |
| Max Sharpe      |           0.206 |               0.160 |    1.002 |     1.458 |         -0.237 |    0.870 |  0.727 |
| Benchmark (SPY) |           0.137 |               0.190 |    0.560 |     0.780 |         -0.337 |    0.406 |  1.000 |

## Optimal weights (full period)

|      |   Equal Weight |   Min Variance |   Max Sharpe |
|:-----|---------------:|---------------:|-------------:|
| AAPL |          10.0% |           0.0% |        27.5% |
| MSFT |          10.0% |           0.2% |        15.9% |
| JPM  |          10.0% |           3.7% |         6.7% |
| XOM  |          10.0% |           6.2% |         7.3% |
| JNJ  |          10.0% |          18.8% |        12.7% |
| SPY  |          10.0% |          13.6% |         0.0% |
| QQQ  |          10.0% |           0.0% |         0.0% |
| TLT  |          10.0% |          30.0% |         0.0% |
| GLD  |          10.0% |          27.6% |        30.0% |
| VNQ  |          10.0% |           0.0% |         0.0% |

## Key findings

- Best out-of-sample Sharpe: **Max Sharpe** (0.97).
- Shallowest max drawdown: **Max Sharpe** (-16.9%).
- Max Sharpe (full period) concentrates in GLD 30.0%, AAPL 27.5%, MSFT 15.9%.
- Sharpe, in-sample → out-of-sample: Equal Weight 0.77 → 0.94, Min Variance 0.56 → 0.60, Max Sharpe 1.00 → 0.97. Max Sharpe held up out-of-sample, so its edge was not just hindsight. (The backtest starts after the first lookback window, so the periods differ slightly.)
