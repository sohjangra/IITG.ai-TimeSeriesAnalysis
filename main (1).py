"""
main.py

End-to-end entry point for the BTC/ETH volatility-forecasting +
regime-switch trading pipeline. Ties together:

    data_preprocessing.py  -> raw data loading, alpha feature engineering
    model_architecture.py  -> VolatilityModel (PatchTST+LoRA -> LSTM -> MLP)
    training_script.py     -> losses, XGBoost returns model, evaluation
    strategy.py             -> indicators, regime classification, entry
                                signals, position state machine
    backtest_engine.py      -> diagnostics, mark-to-market backtest, plots
    pipeline.py              -> wires strategy.py + backtest_engine.py
                                across BTC/ETH and validation/test splits
    config.py                -> every tunable parameter, in one place

Two modes:
    --mode train        Load raw data, train VolatilityModel + the XGBoost
                         returns model, predict on validation/test, cache
                         the predictions to disk, then run the strategy +
                         backtest.
    --mode from-cache    Skip training entirely and run the strategy +
                         backtest directly from a previously-saved cache
                         (see --cache).

Usage:
    python main.py --mode train --cache cache.pkl --output-dir out/
    python main.py --mode from-cache --cache cache.pkl --output-dir out/
"""

import argparse
import sys

import numpy as np
import torch
import torch.optim as optim
import xgboost as xgb
from torch.utils.data import TensorDataset, DataLoader

import data_preprocessing as dp
import training_script as ts
from model_architecture import VolatilityModel
import pipeline as pl
from config import PipelineConfig


# ==============================================================
# 1. TRAINING STAGE -- builds the cache pipeline.py needs
# ==============================================================

def train_models(cfg: PipelineConfig):
    """
    Trains VolatilityModel (PyTorch) + the returns/direction model
    (XGBoost, focal-loss objective) using `data_preprocessing.py` and
    `model_architecture.py`, following the same recipe as
    `training_script.main()` but driven by `cfg` instead of hardcoded
    constants, and without the SHAP/plotting side effects (those still
    live in `training_script.py` if you want the full diagnostic run).

    Returns a `cache` dict shaped exactly as `pipeline.py` expects:
        cache["results"][split]["alpha_btc" | "alpha_eth"]  -> feature df
        cache["calibrated"][split]["btc" | "eth"]            -> pred vol array
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Booting Environment on {device} ---")

    seq_len = cfg.model.seq_len

    # ---- Data ----
    df_train, df_val, df_test = dp.load_raw_data(
        btc_path=cfg.data.btc_path, eth_path=cfg.data.eth_path,
        btc_attention_path=cfg.data.btc_attention_path, eth_attention_path=cfg.data.eth_attention_path,
        test_size=cfg.data.test_size, val_size=cfg.data.val_size, verbose=cfg.data.verbose)

    print("1. Engineering Alpha Matrix (train)...")
    df_train_alpha, feature_cols, feature_cols_ret = dp.build_alpha_feature_matrix(df_train)

    print("\n2. Generating Sequences and Scaling (train)...")
    X_train_vol, y_vol_train, _, feature_scaler_vol = dp.prepare_pytorch_data(
        df_train_alpha, feature_cols, seq_length=seq_len, fit_scaler=None)

    X_train_ret_flat = df_train_alpha[feature_cols_ret].values.astype(np.float32)
    y_train_ret_flat = np.clip(np.round(df_train_alpha["target_ret"].values), 0, 2).astype(int)

    X_train_vol_tensor = torch.tensor(X_train_vol).float().to(device)
    y_vol_train_tensor = torch.tensor(y_vol_train).float().to(device)
    train_loader = DataLoader(TensorDataset(X_train_vol_tensor, y_vol_train_tensor),
                               batch_size=cfg.train.batch_size, shuffle=True)

    # ---- VolatilityModel ----
    print("\n3. Initializing VolatilityModel (Hybrid LoRA-LSTM)...")
    vol_model = VolatilityModel(num_features=len(feature_cols), seq_length=seq_len,
                                 lora_rank=cfg.model.lora_rank, lstm_hidden=cfg.model.lstm_hidden).to(device)
    vol_model.transformer_with_lora.print_trainable_parameters()

    vol_optimizer = optim.Adam(vol_model.parameters(), lr=cfg.train.learning_rate)

    print("\n4. Training VolatilityModel...")
    for epoch in range(cfg.train.epochs):
        vol_model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            vol_optimizer.zero_grad()
            pred_vol = vol_model(batch_x)
            loss = ts.exponential_asymmetric_loss(
                pred_vol, batch_y, penalty_multiplier=cfg.train.vol_penalty_multiplier)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vol_model.parameters(), max_norm=cfg.train.grad_clip_norm)
            vol_optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{cfg.train.epochs} | Vol Loss (Asymmetric): {epoch_loss / len(train_loader):.4f}")

    # ---- Returns model (XGBoost, focal-loss objective) ----
    print("\n5. Training ReturnModel (XGBoost, Focal-Loss Objective)...")
    dtrain_ret = xgb.DMatrix(X_train_ret_flat, label=y_train_ret_flat)
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": cfg.train.xgb_num_class,
        "eta": cfg.train.xgb_eta,
        "max_depth": cfg.train.xgb_max_depth,
        "tree_method": "hist",
        "disable_default_eval_metric": 1,
    }
    ret_model = xgb.train(
        xgb_params, dtrain_ret, num_boost_round=cfg.train.xgb_num_boost_round,
        obj=lambda preds, dtr: ts.focal_loss_xgb_multiclass(
            preds, dtr, alpha=list(cfg.train.xgb_focal_alpha), gamma=cfg.train.xgb_focal_gamma,
            num_class=cfg.train.xgb_num_class),
        custom_metric=lambda preds, dtr: ts.focal_eval_metric_xgb(
            preds, dtr, alpha=list(cfg.train.xgb_focal_alpha), gamma=cfg.train.xgb_focal_gamma,
            num_class=cfg.train.xgb_num_class),
        evals=[(dtrain_ret, "train")], verbose_eval=20)

    print("\n--- Training complete. Predicting on validation and test sets. ---")

    # ---- Predict on validation / test, and assemble the strategy cache ----
    results = {}
    calibrated = {}
    for split_name, df_raw in (("validation", df_val), ("test", df_test)):
        df_alpha, _, _ = dp.build_alpha_feature_matrix(df_raw)
        pred_vol, _, _, _ = ts.evaluate(
            df_raw, split_name, vol_model, ret_model, feature_cols, feature_cols_ret,
            feature_scaler_vol, seq_len, device, num_class=cfg.train.xgb_num_class,
            batch_size=cfg.train.eval_batch_size)

        # NOTE: the provided training pipeline predicts a single trailing
        # volatility target derived from BTC's own returns (there is no
        # separate ETH-specific volatility target/model upstream). Until a
        # dedicated ETH volatility head exists, the same BTC-trained
        # prediction is reused to size the ETH leg -- this is a documented
        # simplification inherited from the given data/model pipeline, not
        # a modeling claim that BTC and ETH volatility are identical.
        results[split_name] = {"alpha_btc": df_alpha, "alpha_eth": df_alpha}
        calibrated[split_name] = {"btc": pred_vol, "eth": pred_vol}

    cache = {"results": results, "calibrated": calibrated}
    return cache


# ==============================================================
# 2. CLI
# ==============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="BTC/ETH volatility-forecasting + regime-switch strategy pipeline.")
    parser.add_argument("--mode", choices=["train", "from-cache"], default="from-cache",
                         help="'train': run data->model->cache->strategy->backtest end to end. "
                              "'from-cache': skip training, load an existing cache (default).")
    parser.add_argument("--cache", default="cache.pkl",
                         help="Path to read (from-cache) or write (train) the pickled prediction cache.")
    parser.add_argument("--output-dir", default="vol_regime_strategy_output",
                         help="Directory to write result CSVs + PNGs to.")
    parser.add_argument("--window-size-candles", type=int, default=200,
                         help="Candles per window for the price+position+volatility+RSI plots.")
    parser.add_argument("--no-show", action="store_true",
                         help="Don't call plt.show() interactively -- still saves PNGs.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    cfg = PipelineConfig()
    cfg.cache_path = args.cache
    cfg.output_dir = args.output_dir
    cfg.window_size_candles = args.window_size_candles
    cfg.plot = not args.no_show

    if args.mode == "train":
        cache = train_models(cfg)
        pl.save_cache(cache, cfg.cache_path)
        print(f"Saved prediction cache to: {cfg.cache_path}")
    else:
        try:
            cache = pl.load_cache(cfg.cache_path)
        except FileNotFoundError:
            print(f"ERROR: cache file not found at '{cfg.cache_path}'. "
                  f"Run with --mode train to generate one, or point --cache at an "
                  f"existing file.", file=sys.stderr)
            sys.exit(1)

    output = pl.run_volatility_regime_pipeline(cache, cfg, plot=cfg.plot)
    summary_df = pl.save_backtest_outputs(
        output, output_dir=cfg.output_dir, window_size_candles=cfg.window_size_candles)

    print("\n=== Final backtest summary ===")
    print(summary_df)
    return output


if __name__ == "__main__":
    main()
