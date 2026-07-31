import pandas as pd
import ccxt
import time
from datetime import datetime, timezone

class DataLoader:
    """
    Handles data fetching strictly from Binance for both the historical 
    training environment and the live execution pipeline.
    """
    
    def __init__(self, symbol="ETH/USDT", timeframe="5m"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange_id = "binance"
        
        # Initialize Crypto Exchange via ccxt
        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True, # Crucial for fetching massive historical datasets without IP bans
        })

    def fetch_historical_crypto(self, since_str, limit=1000, end_str=None):
        """
        Fetches large-scale historical 5-minute data for training.
        Includes pagination to bypass API limits and an optional end date.
        """
        print(f"Fetching historical {self.timeframe} data for {self.symbol} from Binance...")
        
        # Convert date strings to UNIX timestamps in milliseconds
        since_timestamp = self.exchange.parse8601(since_str)
        end_timestamp = self.exchange.parse8601(end_str) if end_str else None
        
        all_ohlcv = []
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since_timestamp, limit)
                
                if len(ohlcv) == 0:
                    break
                    
                # Filter out data that goes beyond the requested end_date
                if end_timestamp:
                    ohlcv = [candle for candle in ohlcv if candle[0] <= end_timestamp]
                    if len(ohlcv) == 0:
                        break

                all_ohlcv.extend(ohlcv)
                
                # Update 'since' to the last timestamp + 1 millisecond to paginate forward
                since_timestamp = ohlcv[-1][0] + 1 
                
                # Stop if we fetched less than the limit (meaning we hit the current time or end date)
                if len(ohlcv) < limit or (end_timestamp and since_timestamp > end_timestamp):
                    break
                    
                # Respect Binance rate limits to avoid temporary IP bans during massive pulls
                time.sleep(0.1) 
                
            except Exception as e:
                print(f"Error fetching data: {e}. Retrying in 5 seconds...")
                time.sleep(5)
                
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df

    def fetch_recent_crypto(self, lookback=78):
        """
        Fetches the exact recent sequence required to warm up the model state.
        Usage: execution.py (to build the initial 78-bar sequence before the C++ queue takes over).
        """
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=lookback)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df


# ==========================================
# Example Usage / Testing Block
# ==========================================
if __name__ == "__main__":
    loader = DataLoader(symbol="ETH/USDT", timeframe="5m")
    
    # 1. Test fetching a specific historical window
    print("Testing Historical Fetch (Jan 1, 2026 to Jan 2, 2026)...")
    historical_data = loader.fetch_historical_crypto(
        since_str="2026-01-01T00:00:00Z", 
        end_str="2026-01-02T00:00:00Z"
    )
    print(f"Historical Shape: {historical_data.shape}")
    print(historical_data.head(3))
    
    # 2. Test grabbing the exact 78-bar lookback for live execution
    print("\nTesting Live Execution Warm-up Fetch...")
    recent_data = loader.fetch_recent_crypto(lookback=78)
    print(f"Recent Shape: {recent_data.shape}")
    print(recent_data.tail(3))