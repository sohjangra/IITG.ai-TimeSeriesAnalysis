import numpy as np
import pandas as pd

class FeatureEngineer:
    """Computes technical indicators for regime switching and volatility prediction."""
    
    def __init__(self, vr_q=5, rolling_window=78):
        self.vr_q = vr_q
        self.window = rolling_window

    def add_time_encoding(self, df):
        # Sin/Cos encoding captures cyclical intraday patterns
        minutes_of_day = df.index.hour * 60 + df.index.minute
        df['sin_time'] = np.sin(2 * np.pi * minutes_of_day / 1440)
        df['cos_time'] = np.cos(2 * np.pi * minutes_of_day / 1440)
        return df

    def calculate_variance_ratio(self, df):
        # Variance Ratio: VR < 1 indicates mean-reversion, VR > 1 indicates trending
        log_ret_1 = np.log(df['close'] / df['close'].shift(1))
        log_ret_q = np.log(df['close'] / df['close'].shift(self.vr_q))
        
        var_1 = log_ret_1.rolling(window=self.window).var()
        var_q = log_ret_q.rolling(window=self.window).var()
        
        df['variance_ratio'] = (var_q / (self.vr_q * var_1)).fillna(1.0)
        return df

    def calculate_mfi(self, df):
        # Market Fragility Index combines Garman-Klass volatility and Amihud illiquidity
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        gk_vol = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        log_ret = np.abs(np.log(df['close'] / df['close'].shift(1)))
        amihud = log_ret / (df['volume_asset'] + 1e-8)
        avg_trade_size_proxy = df['volume_asset'].rolling(window=10).mean().fillna(df['volume_asset'])
        
        df['mfi'] = ((gk_vol * amihud) / (avg_trade_size_proxy + 1e-8)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return df

    def calculate_volume_acceleration(self, df):
        # Volume z-score detects sudden volume acceleration shocks
        rolling_vol_mean = df['volume_asset'].rolling(window=self.window).mean()
        rolling_vol_std = df['volume_asset'].rolling(window=self.window).std()
        df['vol_acceleration'] = ((df['volume_asset'] - rolling_vol_mean) / (rolling_vol_std + 1e-8)).fillna(0.0)
        return df

    def build_feature_set(self, df):
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        if 'volume_asset' not in df.columns and 'volume' in df.columns:
            df['volume_asset'] = df['volume']
        if 'volume' not in df.columns and 'volume_asset' in df.columns:
            df['volume'] = df['volume_asset']
        if 'volume_usdt' not in df.columns:
            df['volume_usdt'] = df['volume_asset'] * df['close'] if 'volume_asset' in df.columns else 0.0
        
        df = self.add_time_encoding(df)
        df = self.calculate_variance_ratio(df)
        df = self.calculate_mfi(df)
        df = self.calculate_volume_acceleration(df)
        df['target_log_return'] = np.log(df['close'].shift(-1) / df['close'])
        df.dropna(inplace=True)
        return df


if __name__ == "__main__":
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
    print(f"Features engineered successfully. Output shape: {processed_df.shape}")