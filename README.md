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

