# Data

Prices come from Yahoo Finance via `yfinance` (split- and dividend-adjusted daily closes).
Running `python main.py` downloads them and caches a copy here as `prices.csv`.
Use `python main.py --refresh` to re-download.

Yahoo Finance data is free and convenient but not institutional-grade; it's fine for research and learning.
