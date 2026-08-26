import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Import custom modules
from src.architecture import DualBranchVolatilityNet
from src.features import FeatureEngineer
from src.data_loader import DataLoader

# ==========================================
# Global State Management
# ==========================================
class ExecutionEnvironment:
    """
    Maintains the state required for the Dual-Branch Volatility strategy.
    Initialized once when the C++ framework loads the Python module.
    """
    def __init__(self):
        print("[INIT] Booting Dual-Branch Execution Environment...")
        
        # 1. Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Strategy Parameters
        self.seq_len = 78
        self.num_features = 10
        self.held_inventory = 0  # Track our position to prevent naked shorting
        self.sma_period = 20
        
        # 3. Load Static Weights
        self.model = DualBranchVolatilityNet(num_features=self.num_features, seq_len=self.seq_len)
        weights_path = os.path.join(os.path.dirname(__file__), 'models', 'dual_branch_weights.pth')
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"[INIT] Model weights loaded successfully from {weights_path}.")
        else:
            raise FileNotFoundError(f"CRITICAL: Cannot find weights at {weights_path}")
            
        # 4. Initialize Feature Engineer
        self.engineer = FeatureEngineer(vr_q=5, rolling_window=self.seq_len)
        
        # 5. Warm up the buffer with live data 
        # (This grabs the exact 78 bars needed before the C++ queue takes over)
        self.data_loader = DataLoader(symbol="ETH/USDT", timeframe="5m")
        print("[INIT] Fetching 100 recent bars for buffer warmup...")
        self.buffer_df = self.data_loader.fetch_recent_crypto(lookback=100)
        
        print("[INIT] Execution Environment Ready.")

    def update_buffer(self, current_price):
        """
        Accumulates ticks into the current 5-minute bar, and appends a new bar
        only when a 5-minute boundary is crossed. This maintains the 5-minute time scale.
        """
        now = pd.Timestamp.now(tz='UTC').tz_localize(None)  # Timezone-naive UTC timestamp
        current_bar_time = now.floor('5min')
        
        last_timestamp = self.buffer_df.index[-1]
        
        if current_bar_time == last_timestamp:
            # Update the current 5-minute bar in place
            self.buffer_df.loc[last_timestamp, 'close'] = current_price
            self.buffer_df.loc[last_timestamp, 'high'] = max(self.buffer_df.loc[last_timestamp, 'high'], current_price)
            self.buffer_df.loc[last_timestamp, 'low'] = min(self.buffer_df.loc[last_timestamp, 'low'], current_price)
            # Volume is kept at its initialized proxy value to avoid accumulation overflow
        else:
            # Boundary crossed, finalize previous bar and create new bar
            last_close = self.buffer_df['close'].iloc[-1]
            
            new_row = {
                'open': last_close,
                'high': max(last_close, current_price),
                'low': min(last_close, current_price),
                'close': current_price,
                # Initial volume proxy using the rolling average volume of past bars
                'volume': self.buffer_df['volume'].rolling(10).mean().iloc[-1]
            }
            
            new_df = pd.DataFrame([new_row], index=[current_bar_time])
            self.buffer_df = pd.concat([self.buffer_df, new_df])
            self.buffer_df = self.buffer_df.iloc[-100:] # Keep last 100 rows

# Initialize the global environment
env = ExecutionEnvironment()

# ==========================================
# The Required Execution Function
# ==========================================
def execute(current_price: float, money_remaining: float) -> dict:
    """
    Core function called by the C++ high-speed queue.
    Takes ONLY current_price and money_remaining.
    Returns a dictionary of whole numbers: {"buy": int, "sell": int}.
    """
    global env
    
    # Default action (Do nothing)
    trade_action = {"buy": 0, "sell": 0}
    
    try:
        # 1. Update state with the new tick
        env.update_buffer(current_price)
        
        # 2. Engineer features for the current state
        features_df = env.engineer.build_feature_set(env.buffer_df)
        
        if len(features_df) < env.seq_len:
            # Not enough data for a full forward pass yet
            return trade_action
            
        # 3. Prepare PyTorch Input (Grab the last 78 bars)
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume', 
            'sin_time', 'cos_time', 'variance_ratio', 
            'mfi', 'vol_acceleration'
        ]
        
        seq_array = features_df[feature_cols].iloc[-env.seq_len:].values
        x_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(env.device)
        
        # 4. Model Inference
        with torch.no_grad():
            predicted_rv, gate_weight = env.model(x_tensor)
            
        # Convert variance to standard deviation (volatility proxy)
        predicted_vol = torch.sqrt(predicted_rv).item()
        current_vr = features_df['variance_ratio'].iloc[-1]
        
        # 5. Dynamic Bands Calculation (around SMA-20)
        sma_20 = features_df['close'].iloc[-env.sma_period:].mean()
        upper_band = sma_20 * (1 + (2 * predicted_vol))
        lower_band = sma_20 * (1 - (2 * predicted_vol))
        
        # 6. Execution Logic (Original Cushioned Flat Strategy)
        if current_vr < 1.0:  # Mean-Reverting Regime
            if current_price <= 1.1 * lower_band:
                # Buy with 5% of remaining capital
                capital_to_risk = money_remaining * 0.05
                qty = int(capital_to_risk // current_price)
                if qty > 0:
                    trade_action["buy"] = qty
                    env.held_inventory += qty
                    
            elif current_price >= 0.9 * upper_band and env.held_inventory > 0:
                # Sell all inventory
                trade_action["sell"] = env.held_inventory
                env.held_inventory = 0
                
        else:  # Breakout / Trending Regime
            if current_price >= 0.9 * upper_band:
                # Buy with 10% of remaining capital
                capital_to_risk = money_remaining * 0.10
                qty = int(capital_to_risk // current_price)
                if qty > 0:
                    trade_action["buy"] = qty
                    env.held_inventory += qty
                    
            elif current_price <= 1.1 * sma_20 and env.held_inventory > 0:
                # Sell all inventory
                trade_action["sell"] = env.held_inventory
                env.held_inventory = 0

    except Exception as e:
        print(f"[EXECUTION ERROR] {e}")
        # On structural failure, fail safely by holding positions and halting buys
        
    return trade_action