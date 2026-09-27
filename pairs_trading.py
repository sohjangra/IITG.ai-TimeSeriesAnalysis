from __future__ import annotations

import json
from pathlib import Path
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass
import requests
import statsmodels.api as sm
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import DataLoader, TensorDataset


def backtest_dates() ->dict[str,str]:
    return {
        "download_start":"2023-12-30",
        "train_start":"2024-01-01",
        "train_end":"2025-02-01",
        "val_start":"2025-04-01",
        "val_end":"2025-09-01",
        "test_start":"2025-09-05",
        "test_end" : "2026-05-29",
        "download_end":"2026-05-31",
    }

_DATES = backtest_dates()

@dataclass
class Config:
    # Crypto Assets Selected based on ADF separately over a universe of coins in training phase
    asset_a : str = "ETH-USD"
    asset_b : str = "UNI-USD"

    # Timelines (train,val,test)
    download_start: str = _DATES["download_start"]
    train_start: str = _DATES["train_start"]
    train_end: str = _DATES["train_end"]
    val_start: str = _DATES["val_start"]
    val_end: str = _DATES["val_end"]
    test_start: str = _DATES["test_start"]
    test_end: str = _DATES["test_end"]
    download_end: str = _DATES["download_end"]

    # Look-back window lenghts
    z_window :int = 480
    sequence_length: int = 24
    forecast_horizon:int = 1

    # LSTM parameters
    hidden_size :int = 16
    num_layers:int = 2
    dropout:int = 0.3

    # Model Training Parameters
    epochs:int = 40
    batch_size: int = 128
    learning_rate: float = 0.0005
    weight_decay: float = 0.00001
    huber_delta: float = 0.7

    # Trading Strategy
    entry_z :float = 0.90
    exit_z:float = 0.30
    max_holding_days:int = 15

    # Assumed defaults
    fee_bps: float = 5.0
    adf_level: float = 0.05
    seed: int = 42


def set_seed(seed):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)


class SpreadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              dropout=dropout,
              batch_first=True,
          )
        # Sending the output of LSTM to a linear layer, hidden denotes the neurons in it
        hidden = 8
        self.head = nn.Sequential(nn.Linear(hidden_size,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self,x) ->torch.Tensor:
          output,_ = self.lstm(x)
          return self.head(output[:,-1,:])


def flatten_download(raw : pd.DataFrame,ticker:str) -> pd.DataFrame:
      if isinstance(raw.columns, pd.MultiIndex):
          if ticker in raw.columns.get_level_values(-1):
              raw = raw.xs(ticker, axis=1, level=-1)
          else:
              raw.columns = raw.columns.get_level_values(0)
      raw = raw.rename_axis("Date").sort_index()
      raw.index = pd.to_datetime(raw.index).tz_localize(None)
      if "Adj Close" not in raw.columns and "Close" in raw.columns:
          raw["Adj Close"] = raw["Close"]
      fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
      return raw[[f for f in fields if f in raw.columns]].dropna(how="all")


def _binance_symbol(ticker: str) -> str:
      return ticker.replace("-USD", "USDT")


def fetch_binance_klines(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
      symbol = _binance_symbol(ticker)
      start_ms = int(pd.Timestamp(start).timestamp() * 1000)
      # Treat the configured end date as inclusive.
      end_ms = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000) - 1
      rows = []
      url = "https://api.binance.us/api/v3/klines"

      while start_ms < end_ms:
          params = {
              "symbol": symbol,
              "interval": interval,
              "startTime": start_ms,
              "endTime": end_ms,
              "limit": 1000,
          }
          response = requests.get(url, params=params, timeout=30)
          response.raise_for_status()
          data = response.json()
          if not data:
              break
          rows.extend(data)
          start_ms = data[-1][0] + 1

      columns = [
          "Open time", "Open", "High", "Low", "Close", "Volume",
          "Close time", "Quote asset volume", "Number of trades",
          "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore",
      ]
      frame = pd.DataFrame(rows, columns=columns)
      frame["Date"] = pd.to_datetime(frame["Open time"], unit="ms")
      frame = frame.set_index("Date")
      for col in ["Open", "High", "Low", "Close", "Volume"]:
          frame[col] = frame[col].astype(float)
      frame["Adj Close"] = frame["Close"]
      return frame[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def fetch_data(config: Config, cache_dir: Path, refresh: bool = False) -> dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    market = {}
    for ticker in (config.asset_a, config.asset_b):
        cache_path = cache_dir / f"{ticker}.parquet"
        if cache_path.exists() and not refresh:
            df = pd.read_parquet(cache_path)
        else:
            df = fetch_binance_klines(
                ticker,
                config.download_start,
                config.download_end,
                interval="5m",  
            )
            df.to_parquet(cache_path)
        market[ticker] = df
    return market

# Sine & Cosine encodings of the minute of the day - Imporves performance of TimeSeries
def add_time_cycle(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    minute_of_day = df.index.hour * 60 + df.index.minute

    df["hour_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    df["hour_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    return df


def build_pair(config:Config, market: dict[str,pd.DataFrame]):
    # Get the assets from the initial config
      a,b = config.asset_a, config.asset_b

    # Concatenate the respective prices & write as a matching time series in two columns
      prices = pd.concat({a: market[a]["Adj Close"], b: market[b]["Adj Close"]}, axis=1).dropna()
      train_prices = prices.loc[config.train_start : config.train_end]

    # Fitting OLS on the log of train prices only.
      hedge = sm.OLS(np.log(train_prices[a]), sm.add_constant(np.log(train_prices[b]))).fit()
      alpha, beta = float(hedge.params.iloc[0]), float(hedge.params.iloc[1])

    # Calculated the spread, rolling mean & rolling std
      spread = np.log(prices[a]) - alpha - beta * np.log(prices[b])
      rolling_mean = spread.shift(1).rolling(config.z_window).mean()
      rolling_std = spread.shift(1).rolling(config.z_window).std()
      zscore = (spread - rolling_mean) / rolling_std

    # Used ADF test to verfiy cointegration
      adf_stat, adf_p, _, _, crit, _ = adfuller(spread.loc[config.train_start : config.train_end].dropna())

      frame = pd.DataFrame({
          f"{a}_price": prices[a],
          f"{b}_price": prices[b],
          "ret_a": prices[a].pct_change(),
          "ret_b": prices[b].pct_change(),
          "spread": spread,
          "rolling_mean": rolling_mean,
          "rolling_std": rolling_std,
          "zscore": zscore,
      }).dropna()

      info = {
          "hedge_alpha": alpha,
          "hedge_beta": beta,
          "spread_adf_stat": float(adf_stat),
          "spread_adf_pvalue": float(adf_p),
          "spread_adf_critical_5pct": float(crit["5%"]),
      }


      print(f"Spread ADF: {adf_stat:.4f}   p={adf_p:.6f}   (5% crit {crit['5%']:.4f})")

      return frame, info, beta



def make_sequences(values: pd.DataFrame, start, end, seq_len, horizon, target_col="spread_scaled"):
    """
    This function converts the time-indexed DataFrame
    into supervised-learning sequences for the LSTM.
    """

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    n = len(values)
    feature_cols = values.columns.tolist()
    x_rows, y_rows, dates = [], [], []
    for i in range(seq_len, n):
        target_idx = i - 1 + horizon
        if target_idx >= n:
            break
        prediction_date = values.index[i]
        if prediction_date < start_ts: continue
        if prediction_date > end_ts:   break
        if values.index[target_idx] > end_ts: break

        x = values.iloc[i - seq_len : i][feature_cols].to_numpy()
        y = values.iloc[target_idx][target_col]
        x_rows.append(x)
        y_rows.append([y])
        dates.append(prediction_date)

    return np.array(x_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32), pd.DatetimeIndex(dates)

def train_lstm(x_train,y_train,config:Config,device):

        # Train the LSTM Model using AdamW Optimizer & Huber Loss for metric to avoid over-penalization of outliers
          input_size = x_train.shape[-1]
          model = SpreadLSTM(input_size, config.hidden_size, config.num_layers, config.dropout).to(device)

          loader = DataLoader(
          TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
          batch_size=config.batch_size, shuffle=True,
          )

          optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay = config.weight_decay)
          loss_fn = nn.HuberLoss(delta=config.huber_delta)

          losses = []

          model.train()

          for i in range(config.epochs):
              batch_losses  = []
              for batch_x, batch_y in loader:
                  batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                  optimizer.zero_grad()
                  loss = loss_fn(model(batch_x), batch_y)
                  loss.backward()
                  nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                  optimizer.step()
                  batch_losses.append(float(loss.detach().cpu()))
              losses.append(float(np.mean(batch_losses)))
              if(i % 10 == 0):
                  print(f"Loss @ epoch {i}= {losses[i]:.4f}")
          return model,losses


def predict_lstm(model, x, device,batch_size):
          loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
          predictions  = []
          model.eval()
          with torch.no_grad():
              for(batch_x,) in loader:
                  predictions.append(model(batch_x.to(device)).cpu().numpy())

          return np.vstack(predictions).ravel()


def lstm_forecast(frame: pd.DataFrame, config: Config, device):
    train_spread = frame["spread"].loc[config.train_start : config.train_end]
    mean, std = float(train_spread.mean()), float(train_spread.std())

    features = add_time_cycle(frame)
    features["spread_scaled"] = (features["spread"] - mean) / std

    feature_cols = ["spread_scaled", "hour_sin", "hour_cos"]
    model_input = features[feature_cols]

    x_train, y_train, _ = make_sequences(
        model_input,
        config.train_start,
        config.train_end,
        config.sequence_length,
        config.forecast_horizon,
    )
    x_val, _, val_dates = make_sequences(
        model_input,
        config.val_start,
        config.val_end,
        config.sequence_length,
        config.forecast_horizon,
    )
    x_test, _, test_dates = make_sequences(
        model_input,
        config.test_start,
        config.test_end,
        config.sequence_length,
        config.forecast_horizon,
    )

    model, losses = train_lstm(x_train, y_train, config, device)

    val_forecast_scaled = predict_lstm(model, x_val, device, config.batch_size)
    test_forecast_scaled = predict_lstm(model, x_test, device, config.batch_size)

    val_forecast = pd.Series(
        val_forecast_scaled * std + mean,
        index=val_dates,
        name="forecast_spread",
    )
    test_forecast = pd.Series(
        test_forecast_scaled * std + mean,
        index=test_dates,
        name="forecast_spread",
    )

    return val_forecast, test_forecast, losses

# Using the rolling stats from before to normalize z_score
def forecast_z(frame: pd.DataFrame, forecast_spread: pd.Series) -> pd.Series:
      aligned = frame.reindex(forecast_spread.index)
      return ((forecast_spread - aligned["rolling_mean"]) / aligned["rolling_std"]).rename("forecast_zscore")



def generate_positions(
    frame: pd.DataFrame,
    config: Config,
    forecast_z: pd.Series,
    start: str,
    end: str,
):
      """
    The strategy trades mean reversion using the previous period’s z-score to avoid look-ahead bias.
    It enters a short position when the z-score exceeds the positive entry threshold and a long position when it falls below the negative threshold.
    It uses the LSTM z_spread forecasts to ensure that the trades are opened only if the predicted z-score moves closer to zero, confirming expected reversion.
    Positions are closed when the z-score reaches the exit threshold or the maximum holding period is exceeded (which would mean diversion from mean).
      """

      dates = frame.loc[start : end].index
      if forecast_z is not None:
          dates = dates.intersection(forecast_z.index)
      z_prev = frame["zscore"].shift(1)

      positions = pd.Series(0.0, index=dates, name="position")
      position, holding = 0.0, 0

      for d in dates:
          z = float(z_prev.loc[d])

          reverting = True
          if forecast_z is not None:
              fz = float(forecast_z.loc[d])
              reverting = np.isfinite(fz) and abs(fz) < abs(z)

          if position == 0.0:
              if z >= config.entry_z and reverting:
                  position, holding = -1.0, 0
              elif z <= -config.entry_z and reverting:
                  position, holding = 1.0, 0
          else:
              holding += 1
              exit_long = position > 0 and z >= -config.exit_z
              exit_short = position < 0 and z <= config.exit_z
              if exit_long or exit_short or holding >= config.max_holding_days * 24 * 12:
                  position, holding = 0.0, 0

          positions.loc[d] = position

      return positions

# This function determines the returns from the strategy, while also accounting for trading fee (in bps)
def pair_returns(frame: pd.DataFrame, positions: pd.Series, beta: float, fee_bps: float):
      aligned = frame.reindex(positions.index)
      long_spread_return = (aligned["ret_a"] - beta * aligned["ret_b"]) / (1.0 + abs(beta))
      turnover = positions.diff().abs()
      if len(turnover):
          turnover.iloc[0] = abs(positions.iloc[0])
      costs = turnover.fillna(0.0) * fee_bps / 10000.0
      return (positions * long_spread_return - costs).fillna(0.0).rename("strategy_return")


def trade_returns(strategy_returns: pd.Series, positions: pd.Series) -> list[float]:
      trades, cumulative, in_trade, previous = [], 1.0, False, 0.0
      for d, position in positions.items():
          daily_return = float(strategy_returns.loc[d])
          if previous == 0.0 and position != 0.0:
              cumulative, in_trade = 1.0 + daily_return, True
          elif in_trade:
              cumulative *= 1.0 + daily_return
          if in_trade and previous != 0.0 and position == 0.0:
              trades.append(cumulative - 1.0)
              in_trade, cumulative = False, 1.0
          previous = position
      if in_trade:
          trades.append(cumulative - 1.0)
      return trades

def performance_metrics(returns: pd.Series,positions: pd.Series | None = None ,periods_per_year: int = 12 * 24 * 365) -> dict[str, float | int]:
    returns = returns.dropna().astype(float)


    # Ensure positions and returns use exactly the same timestamps.
    if positions is not None:
        positions = positions.reindex(returns.index).fillna(0.0)

    equity = (1.0 + returns).cumprod()
    ending_equity = float(equity.iloc[-1])

    elapsed_seconds = (returns.index[-1] - returns.index[0]).total_seconds()

    seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
    years = elapsed_seconds / seconds_per_year

    mean_return = float(returns.mean())
    volatility = float(returns.std(ddof=0))

    downside_returns = np.minimum(returns.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))

    annualization_factor = np.sqrt(periods_per_year)

    annual_volatility = volatility * annualization_factor

    sharpe = (mean_return / volatility) * annualization_factor

    sortino = (mean_return / downside_deviation) * annualization_factor


    drawdown = equity / equity.cummax() - 1.0

    if positions is None:
        # Buy-and-hold is treated as one continuous trade.
        trades = [ending_equity - 1.0]
    else:
        trades = trade_returns(returns, positions)

    winning_trades = sum(trade_return > 0.0 for trade_return in trades)

    win_rate = float(winning_trades / len(trades))


    cagr = float(ending_equity ** (1.0 / years) - 1.0)


    return {
        "total_return": ending_equity - 1.0,
        "ending_equity": ending_equity,
        "CAGR": cagr,
        "annual_volatility": float(annual_volatility),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
        "win_rate": win_rate,
        "number_of_trades": len(trades),
    }

def run(config: Config, output_dir: Path, refresh: bool = False, device_name: str = "cpu"):
      set_seed(config.seed)
      device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
      output_dir.mkdir(parents=True, exist_ok=True)

      print("Initializing...")
      print(f"Train Window --> {config.train_start} to {config.train_end}")
      print(f"Validation Window --> {config.val_start} to {config.val_end}")
      print(f"Test Window -->  {config.test_start} to {config.test_end}")

      market = fetch_data(config, output_dir / "data_cache", refresh)
      frame, info, beta = build_pair(config,market)

      val_forecast_spread, test_forecast_spread, losses = lstm_forecast(frame, config, device)

      val_z_score = forecast_z(frame, val_forecast_spread)
      test_z_score = forecast_z(frame, test_forecast_spread)

      val_positions = generate_positions(
          frame,
          config,
          val_z_score,
          config.val_start,
          config.val_end,
      )
      test_positions = generate_positions(
          frame,
          config,
          test_z_score,
          config.test_start,
          config.test_end,
      )

      val_returns = pair_returns(frame, val_positions, beta, config.fee_bps)
      test_returns = pair_returns(frame, test_positions, beta, config.fee_bps)

      strategy_name = f"PyTorch LSTM spread strategy ({config.forecast_horizon * 5}m horizon)"
      metrics = pd.DataFrame.from_dict(
          {
              "validation": performance_metrics(val_returns, val_positions),
              "test": performance_metrics(test_returns, test_positions),
          },
          orient="index",
      )
      metrics.index.name = "split"
      metrics.insert(0, "strategy", strategy_name)
      metrics.to_csv(output_dir / "performance_metrics.csv")

      def build_daily_output(
          returns: pd.Series,
          forecast_spread: pd.Series,
          z_score: pd.Series,
          positions: pd.Series,
      ) -> pd.DataFrame:
          daily = frame.reindex(returns.index).copy()
          daily["forecast_spread"] = forecast_spread.reindex(daily.index)
          daily["forecast_zscore"] = z_score.reindex(daily.index)
          daily["lstm_position"] = positions.reindex(daily.index)
          daily["lstm_return"] = returns
          daily["lstm_equity"] = (1.0 + returns).cumprod()
          daily.index.name = "Date"
          return daily

      val_daily = build_daily_output(
          val_returns,
          val_forecast_spread,
          val_z_score,
          val_positions,
      )
      test_daily = build_daily_output(
          test_returns,
          test_forecast_spread,
          test_z_score,
          test_positions,
      )

      val_daily.to_csv(output_dir / "validation_backtest.csv")
      test_daily.to_csv(output_dir / "test_backtest.csv")
      pd.concat(
          {"validation": val_daily, "test": test_daily},
          names=["split", "Date"],
      ).to_csv(output_dir / "daily_backtest.csv")

      summary = {
          "config": asdict(config),
          "device": str(device),
          "pair_info": info,
          "train_rows": int(len(frame.loc[config.train_start : config.train_end])),
          "val_rows": int(len(val_returns)),
          "test_rows": int(len(test_returns)),
          "final_training_loss": float(losses[-1]),
          "validation_metrics": metrics.loc["validation"].drop("strategy").to_dict(),
          "test_metrics": metrics.loc["test"].drop("strategy").to_dict(),
      }
      with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
          json.dump(summary, handle, indent=2)

      print("\n")
      print("----RESULTS----")
      print(f"Device: {device}")
      print(f"LSTM final training loss: {losses[-1]:.4f}")
      print(
          f"Training rows: {summary['train_rows']:,}; "
          f"validation rows: {summary['val_rows']:,}; "
          f"testing rows: {summary['test_rows']:,}"
      )
      print("\n")
      columns = [
          "total_return",
          "ending_equity",
          "CAGR",
          "Sharpe",
          "Sortino",
          "max_drawdown",
          "win_rate",
          "number_of_trades",
      ]
      print(metrics[columns].to_string(float_format=lambda v: f"{v:.4f}"))

settings = Config()
parent_dir = Path.cwd()
run(settings, parent_dir/ "crypto_pairs_results")
print()
print("-----Happy Trading!-----")