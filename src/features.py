import numpy as np
import pandas as pd

class FeatureEngineer:
    """
    Computes structural market features for the Dual-Branch Volatility Network.
    Designed to process standard OHLCV DataFrames.
    """
    
    def __init__(self, vr_q=5, rolling_window=78):
        self.vr_q = vr_q
        self.window = rolling_window

    def add_time_encoding(self, df):
        """
        Calculates sin/cos encodings for the time of day.
        Helps the network recognize intraday cyclicality (e.g., market open vs. lunch hours).
        """
        # Get minute of the day (0 to 1439)
        minutes_of_day = df.index.hour * 60 + df.index.minute
        
        # 2 * pi / 1440 minutes in a day
        df['sin_time'] = np.sin(2 * np.pi * minutes_of_day / 1440)
        df['cos_time'] = np.cos(2 * np.pi * minutes_of_day / 1440)
        return df

    def calculate_variance_ratio(self, df):
        """
        Implements Ernest Chan's Variance Ratio test.
        VR < 1: Mean-reverting regime.
        VR > 1: Trending/Breakout regime.
        """
        # 1-period log returns
        log_ret_1 = np.log(df['close'] / df['close'].shift(1))
        
        # q-period log returns
        log_ret_q = np.log(df['close'] / df['close'].shift(self.vr_q))
        
        # Rolling variances
        var_1 = log_ret_1.rolling(window=self.window).var()
        var_q = log_ret_q.rolling(window=self.window).var()
        
        # Variance Ratio
        df['variance_ratio'] = var_q / (self.vr_q * var_1)
        
        # Fill initial NaNs safely
        df['variance_ratio'] = df['variance_ratio'].fillna(1.0) 
        return df

    def calculate_mfi(self, df):
        """
        Calculates the Market Fragility Index (MFI).
        Isolates empty order book panic spikes from stable liquidations.
        """
        # 1. Garman-Klass Volatility
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        gk_vol = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        # 2. Amihud Illiquidity (Absolute log return / Volume)
        log_ret = np.abs(np.log(df['close'] / df['close'].shift(1)))
        amihud = log_ret / (df['volume_asset'] + 1e-8)
        
        # 3. Average Trade Size Proxy
        # Note: Standard OHLCV APIs (like CCXT fetch_ohlcv) do not return 'trade_count'.
        # If your C++ queue streams trade count, replace 'volume' with (volume / trade_count).
        # Here we use a rolling volume smoothing as a proxy to prevent division by zero.
        avg_trade_size_proxy = df['volume_asset'].rolling(window=10).mean().fillna(df['volume_asset'])
        
        # Calculate MFI
        df['mfi'] = (gk_vol * amihud) / (avg_trade_size_proxy + 1e-8)
        
        # Smooth out extreme mathematical anomalies
        df['mfi'] = df['mfi'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return df

    def calculate_volume_acceleration(self, df):
        """
        Measures how the current volume deviates from recent norms,
        acting as a shock-sensor for the 1D-CNN branch.
        """
        rolling_vol_mean = df['volume_asset'].rolling(window=self.window).mean()
        rolling_vol_std = df['volume_asset'].rolling(window=self.window).std()
        
        # Z-score of current volume
        df['vol_acceleration'] = (df['volume_asset'] - rolling_vol_mean) / (rolling_vol_std + 1e-8)
        df['vol_acceleration'] = df['vol_acceleration'].fillna(0.0)
        return df

    def build_feature_set(self, df):
        """
        Master function to execute the pipeline. 
        Takes raw OHLCV and returns the final feature matrix for the Neural Network.
        """
        df = df.copy()
        
        # Calculate base features
        df = self.add_time_encoding(df)
        df = self.calculate_variance_ratio(df)
        df = self.calculate_mfi(df)
        df = self.calculate_volume_acceleration(df)
        
        # Calculate target: Realized Volatility (RV) or Log Returns for the NEXT period
        # Since the proposal asks for log-returns or volatility, we create the target here.
        df['target_log_return'] = np.log(df['close'].shift(-1) / df['close'])
        
        # Drop the rows that don't have enough data for the rolling window
        df.dropna(inplace=True)
        
        return df

# ==========================================
# Example Usage / Testing Block
# ==========================================
if __name__ == "__main__":
    # Generate mock 5-minute data to test the pipeline
    dates = pd.date_range("2026-01-01 09:30:00", periods=200, freq="5min")
    mock_data = pd.DataFrame({
        'open': np.random.uniform(3000, 3100, 200),
        'high': np.random.uniform(3100, 3200, 200),
        'low': np.random.uniform(2900, 3000, 200),
        'close': np.random.uniform(3000, 3100, 200),
        'volume': np.random.uniform(10, 500, 200)
    }, index=dates)

    engineer = FeatureEngineer(vr_q=5, rolling_window=78)
    processed_df = engineer.build_feature_set(mock_data)

    print(f"Original Shape: {mock_data.shape}")
    print(f"Processed Shape (after dropping NaNs from rolling windows): {processed_df.shape}")
    print("\nSample Output (Last 3 rows):")
    print(processed_df[['variance_ratio', 'mfi', 'vol_acceleration', 'sin_time', 'cos_time']].tail(3))