"""
pipeline.py

Wires `strategy.py` (signal generation) and `backtest_engine.py`
(backtesting/diagnostics/plotting) together across BTC + ETH and the
validation/test splits, using the model predictions cached by the
training stage (see `main.py`).

    cache["results"][split]["alpha_btc"]   -> alpha feature df (has 'open','high','low','close', optionally 'volume')
    cache["results"][split]["alpha_eth"]   -> alpha feature df (has 'eth_open','eth_high','eth_low','eth_close', optionally 'eth_volume')
    cache["calibrated"][split]["btc"]      -> calibrated predicted trailing volatility (np array)
    cache["calibrated"][split]["eth"]      -> calibrated predicted trailing volatility (np array)
"""

import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

import strategy as strat
import backtest_engine as bte


# ==============================================================
# 1. RUN ONE ASSET, ONE SPLIT
# ==============================================================

def _run_asset_split(alpha_df, close_col, high_col, low_col, volume_col,
                      pred_vol, prior_pred_vol, strategy_cfg):
    volume = alpha_df[volume_col] if volume_col in alpha_df.columns else None
    strategy_df, features, z, regime = strat.generate_strategy_for_asset(
        pred_vol, alpha_df[close_col], alpha_df[high_col], alpha_df[low_col],
        volume, prior_pred_vol, strategy_cfg)
    return strategy_df


# ==============================================================
# 2. FULL PIPELINE: BTC + ETH, validation + test
# ==============================================================

def run_volatility_regime_pipeline(cache, cfg, plot=True):
    """
    `cfg` is a `PipelineConfig` (see config.py) bundling a `StrategyConfig`
    per asset (`cfg.btc_strategy`, `cfg.eth_strategy`), plus backtest and
    plotting knobs (`cfg.backtest`, `cfg.window_size_candles`).
    """
    results, calibrated = cache["results"], cache["calibrated"]
    seq_len = cfg.seq_len

    strategy_results = {}
    backtest_results = {}       # {"validation": {"btc": (df, summary), "eth": (...)},"test": {...}}
    combined_result = {}        # {"validation": (df, summary), "test": (df, summary)}
    regime_breakdown = {}       # {"validation": {"btc": {...}, "eth": {...}}, "test": {...}}
    bt_dfs_by_asset = {"btc": {}, "eth": {}}

    prior_pred = {"btc": None, "eth": None}

    for split in ("validation", "test"):
        alpha_btc = results[split]["alpha_btc"].reset_index(drop=True).iloc[seq_len:].reset_index(drop=True)
        alpha_eth = results[split]["alpha_eth"].reset_index(drop=True).iloc[seq_len:].reset_index(drop=True)
        pred_btc = calibrated[split]["btc"]
        pred_eth = calibrated[split]["eth"]

        btc_strat = _run_asset_split(
            alpha_btc, "close", "high", "low", "volume",
            pred_btc, prior_pred["btc"], cfg.btc_strategy)
        eth_strat = _run_asset_split(
            alpha_eth, "eth_close", "eth_high", "eth_low", "eth_volume",
            pred_eth, prior_pred["eth"], cfg.eth_strategy)

        prior_pred["btc"], prior_pred["eth"] = pred_btc, pred_eth
        strategy_results[split] = {"btc": btc_strat, "eth": eth_strat}

        print(f"\nBTC {split} signal counts:\n{btc_strat['signal'].value_counts()}")
        print(f"\nETH {split} signal counts:\n{eth_strat['signal'].value_counts()}")

        title_suffix = split.capitalize()

        bte.diagnose_strategy(btc_strat, asset_name=f"BTC ({split})")
        bte.diagnose_strategy(eth_strat, asset_name=f"ETH ({split})")

        bcfg = cfg.backtest
        btc_bt_df, btc_summary = bte.backtest_regime_switch_strategy(
            btc_strat, initial_capital=bcfg.initial_capital, fee_bps=bcfg.fee_bps)
        eth_bt_df, eth_summary = bte.backtest_regime_switch_strategy(
            eth_strat, initial_capital=bcfg.initial_capital, fee_bps=bcfg.fee_bps)
        backtest_results[split] = {"btc": (btc_bt_df, btc_summary), "eth": (eth_bt_df, eth_summary)}
        bt_dfs_by_asset["btc"][split] = btc_bt_df
        bt_dfs_by_asset["eth"][split] = eth_bt_df

        combined_df, combined_summary = bte.run_combined_backtest(
            btc_strat, eth_strat, initial_capital=bcfg.initial_capital, fee_bps=bcfg.fee_bps)
        combined_result[split] = (combined_df, combined_summary)

        regime_breakdown[split] = {
            "btc": bte.backtest_by_regime(btc_strat, initial_capital=bcfg.initial_capital, fee_bps=bcfg.fee_bps),
            "eth": bte.backtest_by_regime(eth_strat, initial_capital=bcfg.initial_capital, fee_bps=bcfg.fee_bps),
        }

        print(f"\n=== Vol-regime-switch strategy backtest ({title_suffix} set, independent legs) ===")
        print(pd.DataFrame([{"asset": "BTC", **btc_summary}, {"asset": "ETH", **eth_summary}]).set_index("asset"))
        print(f"\n=== Combined (50/50 capital, unweighted) portfolio backtest ({title_suffix}) ===")
        print(pd.Series(combined_summary))
        print(f"\n=== Momentum-only vs. Reversal-only breakdown ({title_suffix} set) ===")
        print(pd.DataFrame([
            {"asset": "BTC", "regime": "momentum", **regime_breakdown[split]["btc"]["momentum"][1]},
            {"asset": "BTC", "regime": "reversal", **regime_breakdown[split]["btc"]["reversal"][1]},
            {"asset": "ETH", "regime": "momentum", **regime_breakdown[split]["eth"]["momentum"][1]},
            {"asset": "ETH", "regime": "reversal", **regime_breakdown[split]["eth"]["reversal"][1]},
        ]).set_index(["asset", "regime"]))

        if plot:
            bte.plot_backtest_results(backtest_results[split], title_suffix=title_suffix)
            bte.plot_regime_breakdown(regime_breakdown[split], title_suffix=title_suffix)

    if plot:
        bte.plot_positions_and_price_windowed(
            bt_dfs_by_asset["btc"].get("validation"), bt_dfs_by_asset["btc"].get("test"),
            asset_name="BTC", window_size=cfg.window_size_candles)
        bte.plot_positions_and_price_windowed(
            bt_dfs_by_asset["eth"].get("validation"), bt_dfs_by_asset["eth"].get("test"),
            asset_name="ETH", window_size=cfg.window_size_candles)

    return {
        "strategy_results": strategy_results,
        "backtest": backtest_results,
        "combined_backtest": combined_result,
        "regime_breakdown": regime_breakdown,
    }


# ==============================================================
# 3. CACHE LOAD + OUTPUT SAVING
# ==============================================================

def load_cache(cache_path):
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    required_keys = {"results", "calibrated"}
    missing = required_keys - set(cache.keys())
    if missing:
        raise ValueError(
            f"Loaded cache is missing required key(s): {missing}. "
            f"Expected a dict with 'results' and 'calibrated'."
        )
    return cache


def save_cache(cache, cache_path):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def save_backtest_outputs(output, output_dir="vol_regime_strategy_output", window_size_candles=200):
    os.makedirs(output_dir, exist_ok=True)

    for split, per_asset in output["strategy_results"].items():
        for asset in ("btc", "eth"):
            per_asset[asset].to_csv(os.path.join(output_dir, f"strategy_{asset}_{split}.csv"), index=False)

    summary_rows = []
    for split, per_asset in output["backtest"].items():
        for asset, (bt_df, summary) in per_asset.items():
            bt_df.to_csv(os.path.join(output_dir, f"backtest_{asset}_{split}.csv"), index=False)
            summary_rows.append({"asset": f"{asset.upper()}_{split.upper()}", **summary})

    for split, (combined_df, combined_summary) in output.get("combined_backtest", {}).items():
        combined_df.to_csv(os.path.join(output_dir, f"combined_backtest_{split}.csv"), index=False)
        summary_rows.append({"asset": f"COMBINED_{split.upper()}", **combined_summary})

    for split, per_asset_regime in output.get("regime_breakdown", {}).items():
        for asset, per_regime in per_asset_regime.items():
            for regime_name, (rdf, rsummary) in per_regime.items():
                rdf.to_csv(os.path.join(output_dir, f"regime_{asset}_{regime_name}_{split}.csv"), index=False)
                summary_rows.append({"asset": f"{asset.upper()}_{regime_name.upper()}_ONLY_{split.upper()}", **rsummary})

    summary_df = pd.DataFrame(summary_rows).set_index("asset")
    summary_df.to_csv(os.path.join(output_dir, "backtest_summary.csv"))

    for split, per_asset in output["backtest"].items():
        title_suffix = split.capitalize()

        fig = bte.plot_backtest_results(per_asset, title_suffix=title_suffix)
        fig.savefig(os.path.join(output_dir, f"equity_curves_{split}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        if output.get("regime_breakdown", {}).get(split) is not None:
            breakdown_fig = bte.plot_regime_breakdown(output["regime_breakdown"][split], title_suffix=title_suffix)
            breakdown_fig.savefig(os.path.join(output_dir, f"regime_breakdown_{split}.png"), dpi=150, bbox_inches="tight")
            plt.close(breakdown_fig)

    for asset in ("btc", "eth"):
        bt_df_val, _ = output["backtest"].get("validation", {}).get(asset, (None, None))
        bt_df_test, _ = output["backtest"].get("test", {}).get(asset, (None, None))
        figs = bte.plot_positions_and_price_windowed(
            bt_df_val, bt_df_test, asset_name=asset.upper(), window_size=window_size_candles)
        for idx, wfig in enumerate(figs):
            wfig.savefig(os.path.join(output_dir, f"positions_price_{asset}_window_{idx:04d}.png"),
                         dpi=150, bbox_inches="tight")
            plt.close(wfig)

    print(f"\nSaved strategy CSVs, backtest CSVs, backtest_summary.csv, equity_curves_{{split}}.png, "
          f"regime_breakdown_{{split}}.png, and windowed positions_price_{{btc,eth}}_window_*.png "
          f"to: {output_dir}/")
    return summary_df
