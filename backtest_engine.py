"""
backtest_engine.py

BACKTEST + DIAGNOSTICS + PLOTTING engine for the volatility-regime-switch
strategy (BTC + ETH).

This module owns everything about *measuring what happened*: sanity-check
diagnostics on a strategy's signal/exit output, zero-fee signed-exposure
mark-to-market backtesting (single asset, per-regime, and combined
portfolio), and all plotting.

It consumes the strategy dataframe produced by `strategy.py`
(`run_regime_switch_strategy`'s output) but has no opinion about how that
dataframe was generated -- it only requires the columns:
    close, pred_sigma, z_score, roc, rsi, live_regime, position_regime,
    signal, trade_units, position_units, state, reason, exit_action
(plus, for the windowed plots, rsi_upper_th / rsi_lower_th if you want the
adaptive-threshold overlay -- optional).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================
# 1. DIAGNOSTICS
# ==============================================================

def diagnose_strategy(df_strategy, asset_name="asset"):
    df = df_strategy.reset_index(drop=True)
    max_pos = df["position_units"].abs().max()

    entries = df.index[df["signal"].isin(["ENTRY_LONG", "ENTRY_SHORT"])]
    entry_regimes = df.loc[entries, "position_regime"].value_counts()

    exits = df.index[df["signal"] == "EXIT"]
    n_exits = len(exits)
    # Reasons for exits carry appended detail after "||" (profit-gate status, buy/sell
    # action, P&L) -- group by the base reason code only for a meaningful value_counts().
    exit_reasons = df.loc[exits, "reason"].apply(
        lambda r: r.split("||", 1)[0] if isinstance(r, str) else r
    ).value_counts()

    print(f"--- Diagnostics: {asset_name} ---")
    print(f"Max |position_units| observed: {max_pos:.3f} (should be <= max_leverage_per_asset)")
    print(f"Entries by regime:\n{entry_regimes}")
    print(f"Total exits: {n_exits}\nExit reasons:\n{exit_reasons}")
    print()

    return {"max_position": max_pos, "entry_regimes": entry_regimes.to_dict(),
            "n_exits": n_exits, "exit_reasons": exit_reasons.to_dict()}


# ==============================================================
# 2. ZERO-FEE SIGNED-EXPOSURE MARK-TO-MARKET BACKTEST
# ==============================================================

def backtest_regime_switch_strategy(df_strategy, initial_capital=10_000, fee_bps=0,
                                     slip_base_bps=0, slip_vol_scaler=0,
                                     periods_per_year=365 * 24 * 4):
    """
    Fees and slippage are completely removed for zero-friction backtesting.
    """
    df = df_strategy.copy().reset_index(drop=True)

    signed_exposure = df["position_units"].shift(1).fillna(0.0)  # avoid lookahead
    price_return = df["close"].pct_change().fillna(0.0)
    gross_return = signed_exposure * price_return

    trading_cost = 0.0

    df["strategy_return"] = gross_return - trading_cost
    df["equity"] = initial_capital * (1.0 + df["strategy_return"]).cumprod()

    total_return = df["equity"].iloc[-1] / initial_capital - 1.0
    running_max = df["equity"].cummax()
    max_drawdown = (df["equity"] / running_max - 1.0).min()
    ret_mean, ret_std = df["strategy_return"].mean(), df["strategy_return"].std()
    sharpe = (ret_mean / ret_std) * np.sqrt(periods_per_year) if ret_std > 0 else np.nan
    n_trades = int((df["trade_units"] != 0).sum())

    summary = {
        "final_equity": float(df["equity"].iloc[-1]),
        "total_return_pct": total_return * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "sharpe_ratio_annualized": sharpe,
        "n_trades": n_trades,
        "n_candles": len(df),
    }
    return df, summary


def backtest_by_regime(df_strategy, initial_capital=10_000, fee_bps=0,
                        slip_base_bps=0, slip_vol_scaler=0,
                        periods_per_year=365 * 24 * 4):
    df = df_strategy.reset_index(drop=True)
    out = {}

    for regime_name in ("momentum", "reversal"):
        masked = df.copy()
        in_regime = (masked["position_regime"] == regime_name)
        masked["position_units"] = np.where(in_regime, masked["position_units"], 0.0)
        masked["trade_units"] = np.where(in_regime | (masked["position_regime"].shift(1) == regime_name),
                                          masked["trade_units"], 0.0)

        bt_df, summary = backtest_regime_switch_strategy(
            masked, initial_capital=initial_capital, fee_bps=fee_bps,
            slip_base_bps=slip_base_bps, slip_vol_scaler=slip_vol_scaler,
            periods_per_year=periods_per_year)
        out[regime_name] = (bt_df, summary)

    return out


def run_combined_backtest(btc_strat, eth_strat, initial_capital=10_000, fee_bps=0,
                           slip_base_bps=0, slip_vol_scaler=0,
                           periods_per_year=365 * 24 * 4):
    """
    Fees and slippage are completely removed for zero-friction backtesting.
    """
    btc = btc_strat.copy().reset_index(drop=True)
    eth = eth_strat.copy().reset_index(drop=True)
    n = len(btc)

    def leg_returns(exp, close, sigma):
        signed = pd.Series(exp).shift(1).fillna(0.0)
        price_ret = pd.Series(close).pct_change().fillna(0.0)
        gross_ret = signed * price_ret
        cost = 0.0
        return gross_ret - cost

    btc_ret = leg_returns(btc["position_units"].values, btc["close"].values, btc["pred_sigma"].values)
    eth_ret = leg_returns(eth["position_units"].values, eth["close"].values, eth["pred_sigma"].values)

    combined_ret = 0.5 * btc_ret + 0.5 * eth_ret
    equity = initial_capital * (1.0 + combined_ret).cumprod()

    total_return = equity.iloc[-1] / initial_capital - 1.0
    running_max = equity.cummax()
    max_drawdown = (equity / running_max - 1.0).min()
    ret_mean, ret_std = combined_ret.mean(), combined_ret.std()
    sharpe = (ret_mean / ret_std) * np.sqrt(periods_per_year) if ret_std > 0 else np.nan
    n_trades = int((btc["trade_units"] != 0).sum() + (eth["trade_units"] != 0).sum())

    summary = {
        "final_equity": float(equity.iloc[-1]), "total_return_pct": total_return * 100,
        "max_drawdown_pct": max_drawdown * 100, "sharpe_ratio_annualized": sharpe,
        "n_trades": n_trades, "n_candles": n,
    }
    combined_df = pd.DataFrame({"timestamp_idx": np.arange(n), "btc_exposure": btc["position_units"].values,
                                 "eth_exposure": eth["position_units"].values, "combined_return": combined_ret,
                                 "equity": equity})
    return combined_df, summary


# ==============================================================
# 3. PLOTTING
# ==============================================================

def plot_backtest_results(backtest_results, title_suffix="Test"):
    """
    Equity curves only (continuous, whole test set). Positions + price
    (+ predicted volatility + RSI) are handled separately by
    `plot_positions_and_price_windowed()` in windowed chunks.
    """
    btc_bt_df, btc_summary = backtest_results["btc"]
    eth_bt_df, eth_summary = backtest_results["eth"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    axes[0].plot(btc_bt_df["equity"], color="tab:blue")
    axes[0].set_title(
        f"BTC Vol-Regime-Switch Equity ({title_suffix}) | "
        f"Total Return: {btc_summary['total_return_pct']:.2f}% | "
        f"Sharpe: {btc_summary['sharpe_ratio_annualized']:.2f} | "
        f"Max DD: {btc_summary['max_drawdown_pct']:.2f}% | "
        f"Trades: {btc_summary['n_trades']}"
    )
    axes[0].set_ylabel("Equity ($)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(eth_bt_df["equity"], color="tab:orange")
    axes[1].set_title(
        f"ETH Vol-Regime-Switch Equity ({title_suffix}) | "
        f"Total Return: {eth_summary['total_return_pct']:.2f}% | "
        f"Sharpe: {eth_summary['sharpe_ratio_annualized']:.2f} | "
        f"Max DD: {eth_summary['max_drawdown_pct']:.2f}% | "
        f"Trades: {eth_summary['n_trades']}"
    )
    axes[1].set_ylabel("Equity ($)")
    axes[1].set_xlabel("Time Steps")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def _stagger_marker_offsets(sorted_idxs, min_gap_candles):
    """
    Given entry/exit candle indices IN THIS WINDOW, sorted ascending, return a
    parallel list of small integer stagger levels (0, +1, -1, +2, -2, ...) so
    that markers which land within `min_gap_candles` of the previous one get
    pushed alternately above/below the price line instead of drawing directly
    on top of each other.
    """
    levels = []
    last_idx = None
    run_len = 0
    for idx in sorted_idxs:
        if last_idx is not None and (idx - last_idx) <= min_gap_candles:
            run_len += 1
        else:
            run_len = 0
        magnitude = (run_len // 2) + 1
        sign = 1 if run_len % 2 == 0 else -1
        levels.append(0 if run_len == 0 else sign * magnitude)
        last_idx = idx
    return levels


_EXIT_REASON_EXPLANATIONS = {
    "vwap_invalidation": "momentum exit -- price crossed back over VWAP against the position, thesis invalidated",
    "rsi_peak_reversal": "momentum exit -- RSI ticked off its running peak/trough since entry, first sign the trend is stalling (exits near the extreme instead of waiting for a full reversion)",
    "stop_loss": "reversal exit -- emergency ATR-based stop hit, cutting the fade trade to control risk",
    "take_profit_snapback": "reversal exit -- price snapped back to the Bollinger mid-line (rolling SMA), fade thesis played out",
    "vol_regime_reverted": "reversal exit -- predicted-volatility z-score dropped back below the regime threshold, reversal thesis no longer supported",
}


def _explain_event(signal, reason):
    if signal == "ENTRY_LONG":
        return f"LONG entry -- {reason}"
    if signal == "ENTRY_SHORT":
        return f"SHORT entry -- {reason}"
    if signal == "EXIT":
        if isinstance(reason, str) and "||" in reason:
            base_code, detail = reason.split("||", 1)
        else:
            base_code, detail = reason, ""
        base_sentence = _EXIT_REASON_EXPLANATIONS.get(base_code, base_code)
        return f"EXIT -- {base_sentence} [{detail}]" if detail else f"EXIT -- {base_sentence}"
    return f"{signal} -- {reason}"


def _render_window_column(fig, axes_col, chunk, start, end, asset_name, split_label,
                           w, n_windows):
    """
    Draws one split's (validation OR test) 4-panel stack -- price+position,
    predicted volatility + regime z-score, RSI, plain-language explanation
    -- into the given column of axes for window `w`.
    """
    ax_price, ax_vol, ax_rsi, ax_text = axes_col

    if chunk is None:
        for ax in (ax_price, ax_vol, ax_rsi, ax_text):
            ax.axis("off")
        ax_price.text(0.5, 0.5, f"{split_label}: no data for this window",
                       ha="center", va="center", fontsize=10, transform=ax_price.transAxes)
        return

    has_signal = "signal" in chunk.columns
    has_reason = "reason" in chunk.columns
    has_zscore = "z_score" in chunk.columns
    has_rsi = "rsi" in chunk.columns

    # ---- Panel 1: price + position, with entry/exit markers ----
    ax_price.plot(chunk.index, chunk["close"], color="black", linewidth=1.1, label="Price", zorder=2)
    ax_price.set_ylabel("Price", fontsize=9)
    ax_price.grid(alpha=0.3)
    ax_price.tick_params(labelsize=8)

    if has_signal:
        price_range = chunk["close"].max() - chunk["close"].min()
        offset_unit = max(price_range * 0.045, 1e-9)
        window_span = max(end - start, 1)
        min_gap_candles = max(1, int(window_span * 0.02))

        marker_specs = [
            ("ENTRY_LONG", "^", "tab:green", "Long entry"),
            ("ENTRY_SHORT", "v", "tab:red", "Short entry"),
            ("EXIT", "x", "black", "Exit"),
        ]
        for sig_name, marker, color, label in marker_specs:
            idxs = sorted(chunk.index[chunk["signal"] == sig_name].tolist())
            if not idxs:
                continue
            levels = _stagger_marker_offsets(idxs, min_gap_candles)
            xs, ys = [], []
            for idx, level in zip(idxs, levels):
                true_price = chunk.loc[idx, "close"]
                plotted_y = true_price + level * offset_unit
                xs.append(idx)
                ys.append(plotted_y)
                if level != 0:
                    ax_price.plot([idx, idx], [true_price, plotted_y],
                                  color=color, alpha=0.5, linewidth=0.8, zorder=3)
            ax_price.scatter(xs, ys, marker=marker, color=color, s=80,
                              edgecolors="white", linewidths=0.6, zorder=5, label=label)

    ax_pos = ax_price.twinx()
    ax_pos.fill_between(chunk.index, chunk["position_units"], step="mid",
                         color="tab:blue", alpha=0.25, label="Position")
    ax_pos.axhline(0, color="tab:blue", linewidth=0.6, alpha=0.5)
    ax_pos.set_ylabel("Position (+long/-short)", color="tab:blue", fontsize=8)
    ax_pos.tick_params(axis="y", colors="tab:blue", labelsize=8)

    ax_price.set_title(
        f"{asset_name} -- {split_label}: Price & Position -- candles {start}-{end - 1} "
        f"(window {w + 1}/{n_windows})",
        fontsize=10, pad=10
    )

    lines_1, labels_1 = ax_price.get_legend_handles_labels()
    lines_2, labels_2 = ax_pos.get_legend_handles_labels()
    ax_price.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=7, ncol=2, framealpha=0.9)

    # ---- Panel 2: predicted volatility for the same window (+ regime z-score) ----
    ax_vol.plot(chunk.index, chunk["pred_sigma"], color="tab:orange", linewidth=1.1, label="Predicted sigma")
    ax_vol.set_ylabel("Pred. sigma", color="tab:orange", fontsize=8)
    ax_vol.tick_params(axis="y", colors="tab:orange", labelsize=8)
    ax_vol.tick_params(axis="x", labelsize=8)
    ax_vol.grid(alpha=0.3)
    ax_vol.set_title(f"{asset_name} -- {split_label}: Predicted volatility -- candles {start}-{end - 1}",
                      fontsize=9, pad=10)

    if has_zscore:
        ax_z = ax_vol.twinx()
        ax_z.plot(chunk.index, chunk["z_score"], color="tab:purple", linewidth=0.8, alpha=0.6, label="Rolling z-score")
        ax_z.axhline(0.8, color="tab:purple", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_z.set_ylabel("Z-score\n(dashed=threshold)", color="tab:purple", fontsize=7, labelpad=8)
        ax_z.tick_params(axis="y", colors="tab:purple", labelsize=7)

    # ---- Panel 3: RSI indicator for the same window with adaptive thresholds ----
    if has_rsi:
        ax_rsi.plot(chunk.index, chunk["rsi"], color="tab:green", linewidth=1.0, label="RSI")
        if "rsi_upper_th" in chunk.columns and "rsi_lower_th" in chunk.columns:
            ax_rsi.plot(chunk.index, chunk["rsi_upper_th"], color="tab:red", linestyle="--", linewidth=0.8, alpha=0.6, label="Adaptive Upper")
            ax_rsi.plot(chunk.index, chunk["rsi_lower_th"], color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.6, label="Adaptive Lower")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI", fontsize=8)
        ax_rsi.tick_params(labelsize=8)
        ax_rsi.grid(alpha=0.3)
        ax_rsi.legend(loc="upper left", fontsize=6, ncol=3, framealpha=0.9)
        ax_rsi.set_title(f"{asset_name} -- {split_label}: RSI -- candles {start}-{end - 1}",
                          fontsize=9, pad=10)
        ax_rsi.set_xlabel("Candle index", fontsize=9)
    else:
        ax_rsi.axis("off")

    # ---- Panel 4: plain-language "why" explanation for every entry/exit ----
    ax_text.axis("off")
    ax_text.set_title(
        f"{asset_name} -- {split_label}: Why each position was entered/exited -- candles {start}-{end - 1}",
        fontsize=9, loc="left", pad=8)

    explanation_lines = []
    if has_signal and has_reason:
        events = chunk[chunk["signal"].isin(["ENTRY_LONG", "ENTRY_SHORT", "EXIT"])]
        for candle_idx, row in events.iterrows():
            explanation_lines.append(f"Candle {candle_idx}: {_explain_event(row['signal'], row['reason'])}")
    if not explanation_lines:
        explanation_lines = ["No entries or exits fired in this window."]

    ax_text.text(0.01, 0.97, "\n\n".join(explanation_lines), transform=ax_text.transAxes,
                 fontsize=8, family="monospace", va="top", ha="left", linespacing=1.5,
                 wrap=True)


def plot_positions_and_price_windowed(bt_df_val, bt_df_test, asset_name="asset", window_size=100,
                                       max_windows=None):
    """
    Chunks BOTH the validation and test backtests into `window_size`-candle
    windows and renders them side by side, window index by window index.
    Pass `None` for either `bt_df_val` or `bt_df_test` if that split isn't
    available/wanted -- the other split still renders on its own.

    Returns a list of Figures in chronological window order.
    """
    splits = []
    if bt_df_val is not None:
        splits.append(("Validation", bt_df_val.reset_index(drop=True)))
    if bt_df_test is not None:
        splits.append(("Test", bt_df_test.reset_index(drop=True)))
    if not splits:
        return []

    n_windows_total = max(int(np.ceil(len(df) / window_size)) for _, df in splits)
    n_windows = n_windows_total if max_windows is None else min(max_windows, n_windows_total)
    if max_windows is not None and n_windows_total > n_windows:
        print(f"{asset_name}: rendering {n_windows} of {n_windows_total} windows ({window_size}-candle each) "
              f"-- capped by max_windows={max_windows}. Pass a larger/None max_windows to see the rest.")

    ncols = len(splits)
    figs = []

    for w in range(n_windows):
        col_chunks = []
        for label, df in splits:
            n = len(df)
            start = w * window_size
            end = min(start + window_size, n)
            if start >= n:
                col_chunks.append((label, None, start, end))
            else:
                col_chunks.append((label, df.iloc[start:end], start, end))

        def _n_events(chunk):
            if chunk is None or "signal" not in chunk.columns:
                return 0
            return int(chunk["signal"].isin(["ENTRY_LONG", "ENTRY_SHORT", "EXIT"]).sum())

        max_span = max((end - start for _, chunk, start, end in col_chunks if chunk is not None), default=window_size)
        max_events = max((_n_events(chunk) for _, chunk, _, _ in col_chunks), default=0)

        col_width = max(9.0, max_span * 0.09) + min(4.0, max_events * 0.2)
        fig_width = col_width * ncols

        max_lines = max((_n_events(chunk) for _, chunk, _, _ in col_chunks), default=0)
        max_lines = max(max_lines, 1)
        text_ratio = min(3.0, max(0.8, 0.18 * max_lines))
        fig_height = 10.0 + text_ratio * 1.7

        fig, axes = plt.subplots(
            4, ncols, figsize=(fig_width, fig_height),
            gridspec_kw={"height_ratios": [2.4, 1.3, 1.0, text_ratio], "hspace": 0.7, "wspace": 0.28},
            squeeze=False)

        for col_i, (label, chunk, start, end) in enumerate(col_chunks):
            axes_col = [axes[row][col_i] for row in range(4)]
            _render_window_column(fig, axes_col, chunk, start, end, asset_name, label, w, n_windows)

        fig.suptitle(f"{asset_name}: Validation vs. Test -- window {w + 1}/{n_windows}", fontsize=13, y=0.995)
        fig.subplots_adjust(top=0.94, bottom=0.04, left=0.05, right=0.97)
        plt.show()
        figs.append(fig)

    return figs


def plot_regime_breakdown(regime_breakdown, title_suffix="Test"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    layout = [("btc", "momentum", "tab:blue"), ("btc", "reversal", "tab:cyan"),
              ("eth", "momentum", "tab:orange"), ("eth", "reversal", "tab:red")]

    for ax, (asset, regime, color) in zip(axes.flat, layout):
        df, summary = regime_breakdown[asset][regime]
        ax.plot(df["equity"], color=color)
        ax.axhline(summary["final_equity"] / (1 + summary["total_return_pct"] / 100), color="black",
                   linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(
            f"{asset.upper()} -- {regime.capitalize()}-only ({title_suffix}) | "
            f"Return: {summary['total_return_pct']:.2f}% | "
            f"Sharpe: {summary['sharpe_ratio_annualized']:.2f} | "
            f"Max DD: {summary['max_drawdown_pct']:.2f}% | "
            f"Trades: {summary['n_trades']}"
        )
        ax.set_ylabel("Equity ($)")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig
