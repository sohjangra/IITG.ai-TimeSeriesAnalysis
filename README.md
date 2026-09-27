# Statistical Arbitrage with LSTM-Enhanced Mean Reversion: ETH/UNI Crypto Pairs Trading

A quantitative research project that builds a classical statistical-arbitrage pairs trading
strategy on cryptocurrency spot markets using a deep learning forecast of the
spread's short-term dynamics.

## Overview

Pairs trading is a market-neutral strategy that exploits the tendency of two economically related
assets to revert to a stable price relationship. This project implements the full research
pipeline for such a strategy on **ETH-USD** and **UNI-USD**, sampled at 5-minute resolution:

1. **Cointegration and spread construction** — estimate a hedge ratio via OLS on log prices and
   validate mean-reversion behavior with an Augmented Dickey-Fuller test.
2. **Signal generation** — a rolling z-score of the spread drives a baseline mean-reversion
   trading rule (entry/exit thresholds, maximum holding period).
3. **Deep learning overlay** — a PyTorch LSTM is trained to forecast the next-bar spread from a
   lookback window of scaled spread values and cyclical time-of-day features, producing a
   forward-looking z-score used to filter and confirm trade entries.
4. **Backtesting and evaluation** — a walk-forward-consistent, no-lookahead backtest computes
   returns net of transaction costs, with a full performance suite (Sharpe, Sortino, CAGR, max
   drawdown, win rate, trade count).


## Architecture

```
Data Layer        ──  Binance klines API (5m OHLCV), local CSV cache per asset
Signal Layer       ──  OLS hedge ratio, log-spread, rolling z-score (ADF-validated)
Forecasting Layer  ──  2-layer LSTM (PyTorch) → next-bar spread forecast → forecast z-score
Strategy Layer      ──  Threshold-based entry/exit, max holding period, forecast-based filtering
Backtest Layer      ──  Vectorized position → return mapping with fee-adjusted turnover costs
Evaluation Layer   ──  Sharpe, Sortino, CAGR, max drawdown, win rate, per-trade P&L
```

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch (`nn.LSTM`, custom Huber loss, gradient clipping, AdamW) |
| Statistics / Econometrics | `statsmodels` (OLS hedge estimation, Augmented Dickey-Fuller test) |
| Data Handling | `pandas`, `numpy` |
| Data Source | Binance public REST API (klines endpoint) |
| Visualization | `matplotlib` |


## Methodology

### 1. Pair construction
The hedge ratio (α, β) is estimated by regressing `log(ETH)` on `log(UNI)` over the training
window. The resulting spread, `log(ETH) − α − β·log(UNI)`, is tested for stationarity with an
Augmented Dickey-Fuller test before being used as a trading signal.

### 2. Signal
A rolling z-score of the spread is computed using **only past values** (`shift(1)` applied before
the rolling window), which is the standard way to avoid look-ahead bias in a live-tradable rolling
statistic.

### 3. Forecasting model
An LSTM (2 layers, configurable hidden size, dropout regularization) consumes a fixed-length
lookback window of scaled spread values plus sine/cosine-encoded time-of-day features, and
predicts the next-bar spread. Training uses Huber loss (robust to outlier spread moves) with
gradient clipping and AdamW.

### 4. Trading rule
A position is opened when the (lagged) z-score crosses an entry threshold **and** the LSTM
forecast indicates the spread is expected to revert (smaller forecasted deviation than the current
one). Positions are closed on an exit threshold, mean-reversion confirmation, or a maximum holding
period, whichever comes first.

### 5. Backtest
Returns are computed as the position-weighted, hedge-ratio-adjusted combination of each asset's
realized return, net of a per-turnover transaction cost (in basis points). All decision variables
used to size a position at time `t` are constructed from information available strictly before
`t`, and returns realized at `t` reflect the interval `(t-1, t]`.

### 6. Evaluation
Performance is reported with annualized Sharpe and Sortino ratios (correctly annualized for
5-minute bar frequency), CAGR, maximum drawdown, trade-level win rate, and total trade count — for
both the LSTM-driven strategy and a z-score-only baseline over the same test period, for direct
comparison.

## Results

- Train Window --> 2024-01-01 to 2025-02-01
- Validation Window --> 2025-04-01 to 2025-09-01
- Test Window -->  2025-09-05 to 2026-05-29
- The spread passed the Augmented Dickey-Fuller stationarity test:

```text
ADF statistic: -3.0452
p-value: 0.030887
5% critical value: -2.8616
```

The LSTM training loss decreased from `0.0402` at epoch 0 to `0.0005` at epoch 30.

```text
Device: CPU
Training rows: 114,624
Validation rows: 44,065
Testing rows: 76,609
Final training loss: 0.0005
```

| Split      | Total Return | Ending Equity |      CAGR | Sharpe | Sortino | Max Drawdown | Win Rate | Trades |
| ---------- | -----------: | ------------: | --------: | -----: | ------: | -----------: | -------: | -----: |
| Validation |       66.54% |        1.6654 |   237.66% | 3.6112 |  5.2616 |      -10.24% |   89.03% |    319 |
| Test       |    1,003.30% |       11.0330 | 2,596.27% | 7.1425 | 11.3297 |      -10.45% |   89.23% |    520 |




## Getting Started

```bash
pip install -r requirements.txt

python pairs_trading.py
```

On first run, the script downloads 5-minute OHLCV data for both assets from Binance, caches it
locally, fits the pair and the forecasting model, runs the backtest, and writes results to
`crypto_pairs_results/`.

## Future Work

- Rolling/walk-forward re-estimation of the hedge ratio and z-score parameters.
- Extension to a multi-pair universe with proper per-pair output isolation.
- Introduce bid-ask spread and dynamic slippage models.

## License

MIT