"""
config.py

Single source of truth for every tunable parameter in the pipeline:
data loading, model architecture, training, per-asset strategy rules, and
backtesting. `main.py` builds one `PipelineConfig` and threads it through
`data_preprocessing.py` -> `model_architecture.py` / `training_script.py`
-> `strategy.py` / `backtest_engine.py` (via `pipeline.py`).

Everything here has the same default as the original scripts/CLI, so
`PipelineConfig()` reproduces the original behavior; override individual
fields (or whole sub-configs) for experiments.
"""

from dataclasses import dataclass, field
from typing import Optional


# ==============================================================
# 1. DATA
# ==============================================================

@dataclass
class DataConfig:
    btc_path: str = "/content/btc_15m_data_2018_to_2025.csv"
    eth_path: str = "/content/ETHUSDT 15M.csv"
    btc_attention_path: str = "/content/df_cropped_processed_BTC.csv"
    eth_attention_path: str = "/content/df_cropped_processed_ETH.csv"
    test_size: int = 10_000
    val_size: int = 10_000
    verbose: bool = True


# ==============================================================
# 2. MODEL ARCHITECTURE
# ==============================================================

@dataclass
class ModelConfig:
    seq_len: int = 10
    lora_rank: int = 8
    lstm_hidden: int = 64


# ==============================================================
# 3. TRAINING
# ==============================================================

@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 64
    eval_batch_size: int = 256
    vol_penalty_multiplier: float = 0.5   # exponential_asymmetric_loss penalty for underestimating vol
    grad_clip_norm: float = 1.0

    # XGBoost returns/direction model
    xgb_num_class: int = 3
    xgb_eta: float = 0.1
    xgb_max_depth: int = 5
    xgb_num_boost_round: int = 200
    xgb_focal_alpha: tuple = (0.4, 0.2, 0.4)   # class weights: Down, Chop, Up
    xgb_focal_gamma: float = 2.0

    run_shap: bool = True
    save_normalized_csvs: bool = True
    random_seed: int = 42


# ==============================================================
# 4. STRATEGY (one instance per asset -- BTC and ETH can diverge)
# ==============================================================

@dataclass
class StrategyConfig:
    # Regime switch (rolling z-score of predicted volatility)
    lookback_candles: int = 672          # 7 days of 15m candles
    z_min_periods: int = 96
    z_threshold: float = 0.8

    # Indicators
    rvol_n: int = 14
    vwap_reset_period: int = 96
    bb_n: int = 20
    bb_num_std: float = 2.0
    atr_n: int = 14
    gradient_n: int = 10
    choppiness_n: int = 10
    choppiness_uniformity_th: float = 0.8
    roc_n: int = 12
    rsi_n: int = 7
    rsi_threshold_lookback: int = 96
    rsi_threshold_k: float = 1.9

    # Reversal engine filters
    reversal_gradient_filter: bool = True
    reversal_choppiness_filter: bool = True
    reversal_profit_gate: bool = True

    # Position sizing / risk
    stop_atr_mult: float = 1.0
    min_hold_candles: int = 3
    target_risk_frac: float = 0.004
    max_leverage_per_asset: float = 1.5
    momentum_exit_close_frac: float = 0.9


def default_btc_strategy_config() -> StrategyConfig:
    """BTC defaults: gradient-confirmed + choppiness-gated reversal entries
    + profit-gated reversal snapback exit (matches original CLI defaults)."""
    return StrategyConfig(
        reversal_gradient_filter=True,
        reversal_choppiness_filter=True,
        reversal_profit_gate=True,
    )


def default_eth_strategy_config() -> StrategyConfig:
    """ETH defaults: Bollinger-only reversal entries (no gradient/choppiness
    gate) + unconditional reversal snapback exit."""
    return StrategyConfig(
        reversal_gradient_filter=False,
        reversal_choppiness_filter=False,
        reversal_profit_gate=False,
    )


# ==============================================================
# 5. BACKTEST
# ==============================================================

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000
    fee_bps: float = 0                # frictionless by default
    slip_base_bps: float = 0
    slip_vol_scaler: float = 0
    periods_per_year: int = 365 * 24 * 4   # 15m candles/year, for annualized Sharpe


# ==============================================================
# 6. TOP-LEVEL PIPELINE CONFIG
# ==============================================================

@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    btc_strategy: StrategyConfig = field(default_factory=default_btc_strategy_config)
    eth_strategy: StrategyConfig = field(default_factory=default_eth_strategy_config)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    seq_len: int = 10                  # kept in sync with model.seq_len by __post_init__
    window_size_candles: int = 200     # candles per windowed plot chunk
    cache_path: str = "cache.pkl"
    output_dir: str = "vol_regime_strategy_output"
    plot: bool = True

    def __post_init__(self):
        self.seq_len = self.model.seq_len
