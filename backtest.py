import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your custom architecture
from src.architecture import DualBranchVolatilityNet

def load_test_data(filepath):
    print(f"Loading test data from {filepath}...")
    df = pd.read_csv(filepath)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    return df

def run_backtest(df, model, device, initial_capital=100000.0, seq_len=78, sma_period=20, benchmark_mode=None):
    """
    Simulates the trading strategy.
    Modes:
      - None: Uses the Deep Learning Dual-Branch Volatility forecast.
      - 'constant_vol': Bypasses the DL model and uses standard rolling standard deviation.
    """
    mode_desc = "Dual-Branch DL Model" if not benchmark_mode else "Constant Volatility Baseline"
    print(f"Starting backtest for [{mode_desc}] with ${initial_capital:,.2f} initial capital...")
    
    cash = initial_capital
    inventory = 0.0  
    equity_curve = []
    
    # Tracking metrics
    total_trades = 0
    total_fees_paid = 0.0
    fee_rate = 0.001 # Binance Spot 0.1%
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume_asset', 
        'sin_time', 'cos_time', 'variance_ratio', 
        'mfi', 'vol_acceleration'
    ]
    
    features_array = df[feature_cols].values
    close_prices = df['close'].values
    vr_values = df['variance_ratio'].values
    
    # Pre-compute rolling standard deviation percentage for Constant Volatility baseline
    if benchmark_mode == 'constant_vol':
        pct_returns = pd.Series(close_prices).pct_change()
        rolling_std_pct = pct_returns.rolling(window=sma_period).std().values
    
    for i in range(seq_len, len(df)):
        current_price = close_prices[i]
        current_vr = vr_values[i]
        
        # 1. Volatility Prediction / Estimation
        if benchmark_mode == 'constant_vol':
            # Classic statistical approach: current standard deviation of past returns
            predicted_vol = rolling_std_pct[i] if not np.isnan(rolling_std_pct[i]) else 0.01
        else:
            # Deep Learning approach: Inference pass through Dual-Branch model
            seq = features_array[i - seq_len : i]
            x_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_rv, _ = model(x_tensor)
            predicted_vol = torch.sqrt(predicted_rv).item()
        
        # 2. Dynamic Bands Calculation
        sma_20 = np.mean(close_prices[i - sma_period : i])
        upper_band = sma_20 * (1 + (2 * predicted_vol))
        lower_band = sma_20 * (1 - (2 * predicted_vol))
        
        # 3. Execution Logic
        if current_vr < 1.0:  # Mean-Reverting Regime
            if current_price <= 1.1 * lower_band:
                capital_to_risk = cash * 0.05
                qty = round(capital_to_risk / current_price, 4) 
                fee = (qty * current_price) * fee_rate
                cost = (qty * current_price) + fee
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    inventory += qty
                    total_fees_paid += fee
                    total_trades += 1
                    
            elif current_price >= 0.9 * upper_band and inventory > 0:
                fee = (inventory * current_price) * fee_rate
                revenue = (inventory * current_price) - fee
                cash += revenue
                total_fees_paid += fee
                total_trades += 1
                inventory = 0.0
                
        else:  # Breakout / Trending Regime
            if current_price >= 0.9 * upper_band:
                capital_to_risk = cash * 0.10
                qty = round(capital_to_risk / current_price, 4)
                fee = (qty * current_price) * fee_rate
                cost = (qty * current_price) + fee
                
                if qty > 0 and cash >= cost:
                    cash -= cost
                    inventory += qty
                    total_fees_paid += fee
                    total_trades += 1
                    
            elif current_price <= 1.1 * sma_20 and inventory > 0:
                fee = (inventory * current_price) * fee_rate
                revenue = (inventory * current_price) - fee
                cash += revenue
                total_fees_paid += fee
                total_trades += 1
                inventory = 0.0
                
        # 4. State Tracking
        current_equity = cash + (inventory * current_price)
        equity_curve.append(current_equity)

    # Final Liquidation Window
    if inventory > 0:
        final_price = close_prices[-1]
        fee = (inventory * final_price) * fee_rate
        cash += (inventory * final_price) - fee
        total_fees_paid += fee
        total_trades += 1
        inventory = 0.0
        equity_curve[-1] = cash
        
    return pd.Series(equity_curve, index=df.index[seq_len:]), cash, total_trades, total_fees_paid

def calculate_metrics(equity_series, initial_cap):
    returns = equity_series.pct_change().dropna()
    bars_per_year = 365 * 24 * 12 # 5-minute chunks
    
    total_return = ((equity_series.iloc[-1] - initial_cap) / initial_cap) * 100
    
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min()) * 100
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(bars_per_year) if std_ret > 0 else 0.0
    
    downside_returns = returns[returns < 0]
    down_std = downside_returns.std()
    sortino = (mean_ret / down_std) * np.sqrt(bars_per_year) if down_std > 0 else 0.0
    
    return total_return, max_drawdown, sharpe, sortino

def generate_buy_and_hold_curve(df, initial_capital, seq_len=78):
    close_prices = df['close'].iloc[seq_len:].values
    fee_rate = 0.001
    net_capital = initial_capital * (1 - fee_rate)
    shares_held = net_capital / close_prices[0]
    bh_equity = shares_held * close_prices
    return pd.Series(bh_equity, index=df.index[seq_len:])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_path = "data/processed/test_processed.csv"
    weights_path = "models/dual_branch_weights.pth"
    
    seq_len = 78
    num_features = 10
    initial_cap = 100000.0
    
    # 1. Setup Data Environment
    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found at {test_data_path}.")
        sys.exit(1)
    df_test = load_test_data(test_data_path)
    
    # 2. Setup Deep Learning Engine
    print("Loading Dual-Branch Volatility Network...")
    model = DualBranchVolatilityNet(num_features=num_features, seq_len=seq_len)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device).eval()
    else:
        print(f"Error: Weights not found at {weights_path}.")
        sys.exit(1)
        
    # 3. Process Strategy and Benchmark Variants
    # Run active model strategy
    strat_equity, strat_final, strat_trades, strat_fees = run_backtest(
        df_test, model, device, initial_capital=initial_cap
    )
    
    # Run statistical constant volatility strategy
    cvol_equity, cvol_final, cvol_trades, cvol_fees = run_backtest(
        df_test, model, device, initial_capital=initial_cap, benchmark_mode='constant_vol'
    )
    
    # Run baseline Buy & Hold tracker
    bh_equity = generate_buy_and_hold_curve(df_test, initial_capital=initial_cap, seq_len=seq_len)
    
    # 4. Metric Processing Engine
    strat_ret, strat_dd, strat_sharpe, strat_sortino = calculate_metrics(strat_equity, initial_cap)
    cvol_ret, cvol_dd, cvol_sharpe, cvol_sortino = calculate_metrics(cvol_equity, initial_cap)
    bh_ret, bh_dd, bh_sharpe, bh_sortino = calculate_metrics(bh_equity, initial_cap)
    
    # 5. Output Institutional Performance Ledger
    print("\n" + "="*85)
    print("                COMPLETE SYSTEM EVALUATION MATRIX (Jan 2026 - May 2026)")
    print("="*85)
    print(f"Metric          |  Dual-Branch DL Model  |  Constant Vol Baseline  |  Buy & Hold Benchmark")
    print("-"*85)
    print(f"Final value     |  ${strat_final:,.2f}          |  ${cvol_final:,.2f}            |  ${bh_equity.iloc[-1]:,.2f}")
    print(f"Total Return    |  {strat_ret:.2f}%                  |  {cvol_ret:.2f}%                    |  {bh_ret:.2f}%")
    print(f"Max Drawdown    |  {strat_dd:.2f}%                  |  {cvol_dd:.2f}%                    |  {bh_dd:.2f}%")
    print(f"Sharpe Ratio    |  {strat_sharpe:.2f}                     |  {cvol_sharpe:.2f}                      |  {bh_sharpe:.2f}")
    print(f"Sortino Ratio   |  {strat_sortino:.2f}                     |  {cvol_sortino:.2f}                      |  {bh_sortino:.2f}")
    print("-"*85)
    print(f"Total Trades    |  {strat_trades}                     |  {cvol_trades}                      |  1 (Entry)")
    print(f"Total Fees Paid |  ${strat_fees:,.2f}               |  ${cvol_fees:,.2f}                 |  ${(initial_cap * 0.001):,.2f}")
    print("="*85)
    
    # 6. Comparative Visual Generation
    plt.figure(figsize=(14, 7))
    plt.plot(strat_equity.index, strat_equity.values, label='Dual-Branch DL Strategy', color='blue', linewidth=1.8)
    plt.plot(cvol_equity.index, cvol_equity.values, label='Constant Volatility Baseline', color='purple', linestyle='-.', alpha=0.8)
    plt.plot(bh_equity.index, bh_equity.values, label='Buy & Hold Benchmark', color='gray', linestyle='--', alpha=0.6)
    
    plt.title("Performance Contrast: Machine Learning Model vs. Comparative Baselines", fontsize=12, fontweight='bold')
    plt.ylabel("Account Balance Value (USD)", fontsize=11)
    plt.xlabel("Historical Date Timeline", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("backtest_performance_comparison.png", dpi=300)
    print("\nSaved high-resolution performance comparison plot to 'backtest_performance_comparison.png'.")
    plt.show()