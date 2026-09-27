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

**Volatility-Regime-Switching Strategy — Technical Report (Short Version)

This covers the BTC/ETH regime-switching strategy: how it picks a regime, what each engine trades, how positions are sized and closed, and what changed across the four versions of the script.

1. How the strategy works

The strategy uses the volatility model's predicted volatility (pred_vol) to decide, candle by candle, whether the market is calm or chaotic compared with its own recent history. It then routes each candle to one of two engines:
- Momentum engine (calm markets): trades trend continuation using VWAP and RSI.
- Reversal engine (volatile markets): fades overextended moves back toward the mean using Bollinger Bands.

1.1 Regime switch

pred_sigma = sqrt(clip(pred_vol, 0) / 6)
z = rolling_zscore(pred_vol, lookback=672, min_periods=96)   # 7 days of 15m bars
regime = "reversal" if z >= 0.8, "momentum" if z < 0.8, "insufficient_data" (hold cash) if z is NaN

The z-score window carries over from validation into test, so the test period gets a proper warm-up.

1.2 Momentum engine
- Long: close > VWAP and RSI >= adaptive upper threshold.
- Short: close < VWAP and RSI <= adaptive lower threshold.
- Adaptive thresholds are RSI's rolling mean ± k·std, replacing fixed 70/30 levels.
- Exit: after the minimum hold, exit on the first RSI pullback from its peak (long) or trough (short) since entry.

1.3 Reversal engine
- Short: close > upper Bollinger Band. Long: close < lower band.
- Optional filters (BTC only): trade only if the price move is already slowing (gradient deceleration), and skip if more than 80% of recent candles moved the same way (choppiness gate).
- Exit: a hard ATR-based stop-loss at any time. After the minimum hold, also exit on a snapback to the band mid-line, or when the volatility z-score falls back below 0.8.

1.4 Position sizing
position = direction * min(target_risk_frac / pred_sigma, max_leverage_per_asset)
Higher predicted volatility means a smaller position, so each trade risks roughly the same amount. Leverage is capped at 1.5x.

1.5 State machine
Each asset runs its own state machine. An open trade is always exited with the rules of the engine that opened it. When flat, the current regime decides which entry rules apply.

One quirk to be aware of: exit signals always fire, but the amount actually sold depends on price.
- A momentum exit sells 90% of the position only if price is above entry (for a long). Otherwise it sells nothing, but the trade is still marked closed.
- Reversal snapback and vol-reversion exits follow the same rule, with a full close.
- The reversal stop-loss always closes the whole position.
So an EXIT signal does not always mean the quantity changed.

2. What changed across versions

V1 → V2
- Momentum exit: removed the VWAP-cross exit, so RSI pullback is now the only exit. VWAP crosses happen on ordinary chop, so this lets winning trades run longer.
- Reversal exit: added an exit when the volatility z-score drops back below the threshold. If volatility has calmed, the reason for the trade no longer holds, so the trade closes early instead of sitting open.
- RSI period cut from 14 to 7 to 5 (CLI default), so RSI reacts faster at entry and exit. rsi_threshold_k was raised from 1.9 to 2.0 to offset the extra noise.

V2 → V3
- Reversal bands switched from Keltner Channels (EMA(20) ± ATR multiple) to Bollinger Bands (SMA(20) ± 2 std of close). ATR still sets the stop distance.
- This changes which breakouts count as fade-worthy, not when the reversal regime is active. The effect depends on the data; treat it as an experiment.

V3 → V4
- Transaction fees are finally applied. Before this, trading_cost was hardcoded to 0, so versions 1–3 were all zero-fee backtests whatever fee_bps was set to.
- Sizing is scaled up by 1 / (1 − fee_rate) so risk after fees still matches the target.
- The cost fee_rate × |trade_units| is charged on every actual transaction, per asset, before the 50/50 BTC/ETH combination.
- fee_bps still defaults to 0, so you must set it explicitly to see fees.

3. Key parameters (final version)

lookback_candles 672 | z_min_periods 96 | z_threshold 0.8
vwap_reset_period 96 | bb_n 20 | bb_num_std 2.0
atr_n 14 | stop_atr_mult 1.0 | min_hold_candles 3
target_risk_frac 0.004 | max_leverage_per_asset 1.5 | fee_bps 0
rsi_n 5 | rsi_threshold_lookback 96 | rsi_threshold_k 2.0
momentum_exit_close_frac 0.9 | gradient_n 10
choppiness_n 10 | choppiness_uniformity_th 0.8
rvol_n 14 and roc_n 12 are for diagnostics only.

BTC vs ETH: BTC's reversal engine uses the gradient filter, the choppiness filter and the profit gate. ETH uses none of them and trades a plain Bollinger fade. This has been the same in all four versions.

4. Bottom line

- Momentum trades are held longer and exited closer to the real turn.
- Reversal trades now exit once the high-volatility reason for entering is gone.
- The Keltner-to-Bollinger switch could help or hurt depending on the data.
- Version 4 is the first with real trading costs. To compare it fairly with versions 1–3, re-run those with the same nonzero fee_bps and the fee fix applied.

strategy.py — pure signal generation: indicators (VWAP, ATR, RVOL, Bollinger, RSI, ROC), volatility-regime z-score classifier, momentum/reversal entry rules, and the position state machine. No P&L, no plotting.
backtest_engine.py — pure measurement: diagnostics, zero-fee mark-to-market backtesting (single-asset, per-regime, combined portfolio), and all plotting.
pipeline.py — runs strategy.py + backtest_engine.py across BTC/ETH × validation/test, plus cache load/save and output-writing (this replaces the old monolithic run_volatility_regime_pipeline/save_backtest_outputs).
config.py — every tunable parameter as dataclasses (DataConfig, ModelConfig, TrainConfig, StrategyConfig per-asset, BacktestConfig, all bundled in PipelineConfig).
main.py — the entry point tying it all together: data_preprocessing.py → model_architecture.py/training_script.py (training + prediction caching) → pipeline.py (strategy + backtest), driven entirely by config.py.
model_architecure.py - contains the architecture of model used for forecasting the volatility from given data
training.py - contains the training script for the given model
