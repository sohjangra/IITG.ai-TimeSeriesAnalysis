import pandas as pd
import ccxt
import time

class DataLoader:
    """Fetches Binance OHLCV data for training and live execution warmup."""
    
    def __init__(self, symbol="ETH/USDT", timeframe="5m"):
        self.symbol = symbol
        self.timeframe = timeframe
        exchange_class = getattr(ccxt, "binance")
        self.exchange = exchange_class({'enableRateLimit': True})

    def fetch_historical_crypto(self, since_str, limit=1000, end_str=None):
        since_timestamp = self.exchange.parse8601(since_str)
        end_timestamp = self.exchange.parse8601(end_str) if end_str else None
        all_ohlcv = []
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since_timestamp, limit)
                if not ohlcv:
                    break
                if end_timestamp:
                    ohlcv = [candle for candle in ohlcv if candle[0] <= end_timestamp]
                    if not ohlcv:
                        break

                all_ohlcv.extend(ohlcv)
                since_timestamp = ohlcv[-1][0] + 1
                if len(ohlcv) < limit or (end_timestamp and since_timestamp > end_timestamp):
                    break
                time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching data: {e}. Retrying in 5s...")
                time.sleep(5)
                
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def fetch_recent_crypto(self, lookback=78):
        # Fetches recent bars to warm up the 78-bar lookback sequence
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=lookback)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df


if __name__ == "__main__":
    loader = DataLoader(symbol="ETH/USDT", timeframe="5m")
    recent_data = loader.fetch_recent_crypto(lookback=78)
    print(f"Fetched {len(recent_data)} recent bars.")