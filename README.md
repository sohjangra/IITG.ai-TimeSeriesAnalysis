# IITG.ai-TimeSeriesAnalysis
This repository contains multiple DL volatility prediction based models we experimented with. All the different approaches as are follows:

## 1. Jump-BiLSTM Volatility Forecasting and Adaptive Trend Strategy (BTC/USDT)

This approach combines deep-learning volatility forecasting with a volatility-aware trend-following strategy for BTC/USDT.

The model uses five-minute market data, jump-related features, cross-asset volatility signals, and a 78-timestep lookback window to forecast realised volatility 30 minutes ahead. The resulting forecast is then used to adjust the trailing-stop distance in the AdaptiveTrend strategy.

### Dataset

| Split | Period |
| --- | --- |
| Training | January 2023 - June 2025 |
| Validation | July 2025 - December 2025 |
| Test | January 2026 - June 2026 |
| Data source | KuCoin via CCXT |
| Total observations | 368,333 five-minute bars |

The input features include HAR-RV lags, jump decomposition features, ETH volatility, the Deribit DVOL index, intraday seasonality, and FinBERT-based news sentiment.

### Volatility Model Comparison

| Model | QLIKE | QLIKE Reduction vs. HAR-RV |
| --- | ---: | ---: |
| HAR-RV | 0.9900 | Baseline |
| HAR-LSTM | 0.4689 | 52.6% |
| DeepVol TCN | 0.4661 | 53.1% |
| **Jump-BiLSTM with attention** | **0.4319** | **56.7%** |

The Jump-BiLSTM was the best-performing model. It combines bidirectional LSTM layers, attention over the input sequence, and explicit jump-related features.

News sentiment had minimal impact on forecasting performance, changing QLIKE by less than 0.5%. This was likely influenced by the limited news coverage in the dataset.

### AdaptiveTrend Strategy

The predicted volatility is used to scale the strategy's ATR-based trailing stop:

```text
stop_distance = k * ATR(14) * vol_factor
vol_factor    = predicted_volatility / median_training_predicted_volatility
```

The strategy operates on six-hour bars and uses EMA-based trend signals. The parameter `k = 4.0` was selected using the validation set and fixed before testing.

### Test-Period Results

| Metric | AdaptiveTrend | Buy and Hold |
| --- | ---: | ---: |
| Bar-level Sharpe ratio | **1.838** | -1.935 |
| Total return | **33.24%** | -35.79% |
| Maximum drawdown | **-20.68%** | -42.21% |
| Months outperforming buy and hold | **5 of 6** | - |
| Number of trades | 16 | - |

The strategy's bootstrap 95% confidence interval for the Sharpe ratio was `[-1.109, 4.639]`. Since the test period covers only six months and 16 trades, these results should be considered preliminary. A longer walk-forward evaluation across different market conditions would be required for stronger conclusions.



## 2. Dual-Branch Gated Volatility Network Strategy (ETH-USDT, 5m Candles)

#### 1. Overview & Strategy Logic
A deep learning trading strategy for ETH-USDT (5-minute OHLCV) that predicts volatility and dynamically switches trading rules using the **Variance Ratio (VR)**:
* **Dynamic Volatility Bands**: Calculated as $\text{SMA-20} \times (1 \pm 2\hat{\sigma})$, where $\hat{\sigma}$ is forecasted by a PyTorch model.
* **Range-Bound Regime ($VR < 1.0$)**: Mean-reversion strategy. Buys long when price drops below the lower band and shorts when price exceeds the upper band.
* **Trending Regime ($VR \ge 1.0$)**: Momentum strategy. Buys long on upside breakouts above upper band and shorts on breakdown below lower band. Liquidates back to cash upon SMA-20 reversion.

#### 2. Model Architecture (`DualBranchVolatilityNet`)
* **Branch A (Conv1D)**: Extracts spatial price shocks, liquidity gaps, and sharp regime breaks.
* **Branch B (LSTM)**: 2-layer LSTM modeling temporal memory and historical volatility persistence.
* **Gated Attention Merger**: Dynamically weights Conv1D shock features vs. LSTM memory based on market state context.
* **QLIKE Loss Function**: Custom loss function that heavily penalizes volatility under-prediction to protect against liquidation during market crashes.

#### 3. Feature Engineering (78-Bar Lookback ~ 6.5 Hours)
Engineers 10 internal features from raw OHLCV data:
* **Regime & Risk**: Variance Ratio (VR), Garman-Klass Volatility, Amihud Illiquidity, Market Fragility Index (MFI).
* **Volume Dynamics**: Volume Acceleration (z-score), Average Trade Size Proxy.
* **Time Encoded**: Cyclical Sin/Cos time-of-day features.

#### 4. Performance Matrix (Jan 2026 – May 2026)

| Metric | Dual-Branch DL Strategy | Constant Vol Baseline | Buy & Hold Benchmark |
| :--- | :---: | :---: | :---: |
| **Total Return** | **+25.72%** | -47.35% | -47.29% |
| **Max Drawdown** | **12.64%** | 55.39% | 55.39% |
| **Sharpe Ratio** | **2.29** | -1.70 | -1.69 |
| **Sortino Ratio** | **0.57** | -2.22 | -2.21 |
| **Total Trades / Fees** | 1,090 / $5,730.86 | 182 / $152.60 | 1 / $100.00 |

## 3.Statistical Arbitrage with LSTM-Enhanced Mean Reversion: ETH/UNI Crypto Pairs Trading

A classical stat-arb pairs trading strategy on **ETH-USD / UNI-USD** (5-min bars), with a PyTorch LSTM layered on top of the usual z-score mean-reversion logic to filter and confirm entries.

### Pipeline

- **Cointegration & spread**: hedge ratio via OLS on log prices, spread validated for mean-reversion via ADF test.
- **Signal**: rolling z-score of the spread (computed on lagged/shifted values only, so no look-ahead) drives entry/exit thresholds and a max holding period.
- **DL overlay**: 2-layer LSTM (Huber loss, gradient clipping, AdamW) takes a lookback window of scaled spread + sin/cos time-of-day features and forecasts the next-bar spread → forward z-score used to confirm whether the spread is actually expected to revert before entering.
- **Backtest**: walk-forward-consistent, no-lookahead, returns net of transaction costs (bps per turnover), full suite of Sharpe/Sortino/CAGR/max DD/win rate/trade count reported for both the LSTM strategy.

### Stack
PyTorch (LSTM, custom Huber loss), `statsmodels` (OLS + ADF), `pandas`/`numpy`, Binance REST API for data, `matplotlib` for plots.

### Results

**Data windows**: Train Jan'24–Feb'25 · Val Apr'25–Sep'25 · Test Sep'25–May'26

**ADF test on the spread** — stationary, passes at 5%:
```
ADF statistic: -3.0452
p-value: 0.030887
5% critical value: -2.8616
```

LSTM training loss dropped from 0.0402 (epoch 0) to 0.0005 (epoch 30), trained on 114,624 rows (CPU).

| Split      | Total Return | CAGR      | Sharpe | Sortino | Max DD  | Win Rate | Trades |
|------------|-------------:|----------:|-------:|--------:|--------:|---------:|-------:|
| Validation | 66.54%       | 237.66%   | 3.61   | 5.26    | -10.24% | 89.03%   | 319    |
| Test       | **1,003.30%**| **2,596.27%** | **7.14** | **11.33** | -10.45% | 89.23%   | 520    |


### Running it

```bash
pip install -r requirements.txt
python pairs_trading.py
```

First run pulls 5m OHLCV for both assets from Binance, caches locally, fits the pair + LSTM, backtests, and dumps results to `crypto_pairs_results/`.


##4. Multi-Branch MODWT-CNN-LSTM Volatility Engine and Execution Sandbox

### Project Overview
An end-to-end quantitative trading and volatility forecasting framework designed for high-frequency cryptocurrency markets (Binance ETH/USDT 10-minute candles). The system combines time-frequency wavelet decomposition, spatial-temporal deep learning, and a fee-aware execution engine to detect ultra-low volatility compression ("squeezes") before explosive directional breakouts occur.

### Technical Architecture
* Wavelet Signal Filtering: Applies a Maximal Overlap Discrete Wavelet Transform (MODWT) using Daubechies 4 (db4) wavelets to separate macro trends from microstructural noise across six market features (log returns, Garman-Klass volatility, log range, volume change, body gap, and relative close position).
* Spatial Encoding: Transforms wavelet outputs into Gramian Angular Field (GAF) polar matrices, converting 1D price time series into 2D spatial feature tensors ($32 \times 32 \times 12$).
* Dual-Branch Fusion Network: Integrates a 2D CNN branch (extracting spatial compression patterns from GAF tensors) with a parallel LSTM branch (processing continuous sequence momentum). Outputs next-bar volatility predictions using an asymmetric QLIKE loss function that heavily penalizes volatility underestimation.
* Execution Infrastructure: Modular Python architecture featuring vectorized batch inference, paginated data processing via the Binance API, and realistic market friction modeling (0.05% taker / 0.02% maker fees).

### Model Training and Out-of-Sample Performance
The system was trained on full 2024 to 2025 Binance ETH/USDT data (~105,120 10-minute bars) and evaluated out-of-sample on the first six months of 2026 (25,550 bars).

#### Out-of-Sample Strategy Results (H1 2026)

| Metric | Strategy 1 (Clean Taker) | Strategy 2 (Post-Maker) | Strategy 3 (Anti-Chop) |
| :--- | :---: | :---: | :---: |
| Net Return | **+14.06%** | +12.86% | +5.43% |
| Max Drawdown | -10.00% | -7.94% | **-3.95%** |
| Sharpe Ratio (Annualized) | 1.55 | **1.76** | 1.04 |
| Win Rate | 42.4% | 43.2% | 38.6% |
| Total Trades | 66 | 44 | 44 |
| Total Fees Paid | 8.22% (0.05% Taker) | 2.23% (0.02% Maker) | 3.85% (0.05% Taker) |
| Primary Execution | Market Orders | Limit Orders | Dynamic Sizing |

### Key Findings and Operational Takeaways

1. The Ensemble Paradox: Combining the Neural Network with a GARCH(1,1) econometric model achieved the best statistical score (lowest QLIKE loss of 0.6207). However, the standalone Neural Network delivered higher net backtest returns (+14.06%). The GARCH component introduced smoothing lag into rolling percentile rankings, whereas the neural network detected raw, sharp compression points for better entry timing.
2. Market Taker Superiority: Strategy 1 (Clean Taker) proved to be the most viable production setup. Paying the 0.05% taker fee guaranteed instant entries on fast breakout candles, allowing outsized trend runners to easily absorb transaction fee drag.
3. Execution Mechanics: Passive limit orders (Strategy 2) suffered from adverse selection in live conditions, missing fast winning breakouts while getting filled on bad setups. Defensive position scaling (Strategy 3) successfully reduced max drawdown to -3.95%, but hurt overall mathematical expectancy by catching major trend runners at half leverage.

### Production Targets
* Model Gate: Standalone GAF-CNN-LSTM Neural Network
* Execution Mode: Strategy 1 Market Taker Orders
* Performance Profile: +14.06% Net Return, 1.55 Sharpe Ratio, -10.00% Max Drawdown
