# Real-Time AI Trading Strategy (ETH-USDT)

A Python-based deep learning trading system for ETH-USDT on 5-minute candles. It uses a **Dual-Branch PyTorch Neural Network** to predict market volatility and dynamically adapt between mean-reversion and momentum breakout strategies with dual long and short position capabilities.

---

## 1. Overall Strategy

The strategy dynamically adapts its trading logic based on market volatility regimes:

* **Market Regime Detection**: Uses the **Variance Ratio (VR)** to measure whether the market is range-bound ($VR < 1.0$) or trending ($VR \ge 1.0$).
* **Dynamic Volatility Bands**: Calculates volatility bands around a 20-period Moving Average ($\text{SMA-20}$) using the model's predicted volatility ($\hat{\sigma}$):
  $$\text{Bands} = \text{SMA-20} \times (1 \pm 2\hat{\sigma})$$
* **Trading Logic (Long & Short)**:
  * **Range-Bound Regime ($VR < 1.0$)**: Mean-reversion trading. Buys long when price drops below the lower band (`1.1 * lower_band`) and opens short positions when price reaches the upper band (`0.9 * upper_band`).
  * **Trending Regime ($VR \ge 1.0$)**: Momentum trading. Buys long on upside breakouts above the upper band (`0.9 * upper_band`) and opens short positions on downside breakdowns below the lower band (`1.1 * lower_band`).
  * **Trailing Liquidations**: Positions exit back to cash when price reverts to SMA-20.

---

## 2. Technical Details & Model Architecture

### Feature Engineering
From raw **5-minute OHLCV data**, 10 features are computed over a **78-bar lookback window** (~6.5 hours of history):
* **Variance Ratio (VR)**: Market regime classifier.
* **Garman-Klass Volatility**: Intraday volatility estimator.
* **Amihud Illiquidity & Market Fragility Index (MFI)**: Measures market liquidity and structural stress.
* **Average Trade Size & Volume Acceleration**: Tracks volume and trade flow dynamics.
* **Sin/Cos Time Encodings**: Captures cyclical time-of-day patterns.

### Dual-Branch Neural Network (`DualBranchVolatilityNet`)
* **Branch A (Conv1D)**: 1D CNN that extracts spatial price shocks and sudden liquidity gaps.
* **Branch B (LSTM)**: 2-layer LSTM that models long-term volatility clustering and temporal trends.
* **Gated Attention Merger**: Dynamically weights the Conv1D and LSTM outputs using the Variance Ratio ($VR$). High $VR$ favors Conv1D shock features; low $VR$ favors LSTM memory.
* **Custom QLIKE Loss**: Penalizes under-predicting volatility to protect against crash liquidations:
  $$\text{Loss} = \frac{y}{\hat{y}} - \ln\left(\frac{y}{\hat{y}}\right) - 1$$

---

## 3. Performance Results (Jan 2026 – May 2026)

| Metric | Dual-Branch DL Model | Constant Vol Baseline | Buy & Hold Benchmark |
| :--- | :---: | :---: | :---: |
| **Final Value** | **$125,722.92** | $52,648.20 | $52,714.83 |
| **Total Return** | **+25.72%** | -47.35% | -47.29% |
| **Max Drawdown** | **12.64%** | 55.39% | 55.39% |
| **Sharpe Ratio** | **2.29** | -1.70 | -1.69 |
| **Sortino Ratio** | **0.57** | -2.22 | -2.21 |
| **Total Trades** | 1,090 | 182 | 1 (Entry) |
| **Total Fees Paid** | $5,730.86 | $152.60 | $100.00 |

---

## 4. How to Run

### Installation
Set up a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Backtest
Simulate strategy performance on historical data and generate comparative performance plots:
```bash
python backtest.py
```

### Run Live Execution Module
Initialize the live bar buffer and start real-time inference readiness:
```bash
python execution.py
```
