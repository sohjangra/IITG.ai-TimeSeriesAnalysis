"""
strategy.py

Volatility-regime-switching STRATEGY LOGIC (BTC + ETH): Momentum vs. Reversal.

This module owns everything about *deciding what to do*: technical
indicators, the volatility-regime classifier, entry-signal rules for the
Momentum and Reversal engines, and the position state machine that turns
those signals into a signed exposure series with an auditable reason for
every entry/exit.

It deliberately contains NO backtesting, P&L accounting, or plotting --
that lives in `backtest_engine.py`. `pipeline.py` is what wires this
module's output into `backtest_engine.py` per split (validation/test) and
per asset (BTC/ETH).

Candle timeframe is assumed to be 15 minutes throughout (lookback windows,
annualization factor, etc. are all expressed in 15m-candle units).
"""

import numpy as np
import pandas as pd


# ==============================================================
# 1. INDICATORS (VWAP, ATR, RVOL, Bollinger Bands, ROC, RSI, ...)
# ==============================================================

def compute_atr(high, low, close, n=14):
    """Wilder ATR. Causal (uses only the current + past bars)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def compute_rvol(volume, close=None, high=None, low=None, n=14):
    """
    Relative Volume (RVOL) -- MODIFIED replacement for ADX in the Momentum
    engine, used in the same structural role (a "strong AND accelerating"
    strength/acceleration filter).

    RVOL(t) = volume(t) / mean(volume, trailing n bars, EXCLUDING bar t)

    The trailing average is shifted by one bar so the current bar's own
    volume never contributes to its own baseline -- causal by construction
    and not tautologically close to 1.0.

    If `volume` is None (no volume column available upstream), falls back
    to a range-based proxy for market activity, (high - low) / close,
    evaluated relative to ITS OWN trailing n-bar average in the same way.
    This is a documented fallback, NOT true RVOL -- supply real volume if
    you have it.

    NOTE: RVOL is currently unused by the entry/exit decision logic (see
    `compute_momentum_signal`) -- it is retained purely as a diagnostic
    feature column.
    """
    if volume is not None:
        activity = pd.Series(volume).reset_index(drop=True).astype(float)
    else:
        close_s = pd.Series(close).reset_index(drop=True)
        high_s = pd.Series(high).reset_index(drop=True)
        low_s = pd.Series(low).reset_index(drop=True)
        activity = ((high_s - low_s) / close_s.replace(0, np.nan)).abs()

    baseline = activity.rolling(n, min_periods=n).mean().shift(1)
    rvol = activity / baseline.replace(0, np.nan)
    return rvol


def compute_vwap(close, high, low, volume=None, reset_period=96):
    """
    VWAP anchored every `reset_period` candles (default 96 = 1 day of 15m
    candles). If `volume` is None, falls back to an equal-weighted rolling
    mean of typical price over the same window -- flagged clearly since it
    is NOT a true VWAP.
    """
    typical_price = (high + low + close) / 3.0
    n = len(close)
    group = pd.Series(np.arange(n) // reset_period, index=close.index)

    if volume is None:
        vwap = typical_price.groupby(group).transform(lambda s: s.expanding().mean())
        return vwap

    volume = pd.Series(volume).reset_index(drop=True)
    volume.index = close.index
    pv = typical_price * volume
    cum_pv = pv.groupby(group).cumsum()
    cum_vol = volume.groupby(group).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def compute_price_gradient(close, n=5):
    """
    Price gradient -- used as an optional confirmation filter on top of
    the Bollinger Band breakout in the Reversal engine.

        gradient(t) = (close(t) - close(t - n)) / n

    Positive gradient -> price has been rising over the window; negative ->
    falling. This is NOT the slope of a fitted trendline; it's the direct
    trailing difference, which is causal, cheap, and gives the same
    qualitative "is the move accelerating or decelerating" signal when
    compared candle-to-candle.

    NOTE: this is an ABSOLUTE per-candle price difference (used internally
    as the Reversal engine's deceleration filter), NOT a percentage. See
    `compute_roc` below for the classic percentage Rate-of-Change indicator
    used purely for plotting/diagnostics.
    """
    close = pd.Series(close).reset_index(drop=True)
    return (close - close.shift(n)) / n


def compute_roc(close, n=12):
    """
    Classic Rate-of-Change (ROC) indicator, expressed as a percentage:

        ROC(t) = (close(t) - close(t - n)) / close(t - n) * 100

    Causal (trailing-only, uses shift(n)); NaN for the first n candles.
    Purely a diagnostic indicator threaded through the strategy dataframe /
    CSV outputs; it does NOT feed into any entry/exit logic.
    """
    close = pd.Series(close).reset_index(drop=True)
    prior = close.shift(n)
    return (close - prior) / prior.replace(0, np.nan) * 100.0


def compute_rsi(close, n=14):
    """
    Wilder's RSI (Relative Strength Index), causal by construction.

        delta(t)   = close(t) - close(t-1)
        gain(t)    = max(delta(t), 0)
        loss(t)    = max(-delta(t), 0)
        avg_gain(t)= EWM(gain, alpha=1/n)
        avg_loss(t)= EWM(loss, alpha=1/n)
        RS(t)      = avg_gain(t) / avg_loss(t)
        RSI(t)     = 100 - (100 / (1 + RS(t)))

    Edge cases (kept explicit rather than silently propagating NaN/inf):
        - avg_loss == 0 and avg_gain > 0  -> RSI = 100 (pure uptrend so far)
        - avg_loss == 0 and avg_gain == 0 -> RSI = 50  (no movement at all
          yet, e.g. the very first candle)

    Used here as a momentum ENTRY GATE (see `compute_momentum_signal`).
    """
    close = pd.Series(close).reset_index(drop=True)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    no_loss = avg_loss == 0
    rsi = rsi.where(~no_loss, 100.0)
    rsi = rsi.where(~(no_loss & (avg_gain == 0)), 50.0)

    return rsi


def compute_rsi_adaptive_thresholds(rsi, lookback=96, min_periods=20, k=1.0,
                                     upper_bound=95.0, lower_bound=5.0):
    """
    Rolling (trailing, causal) ADAPTIVE RSI thresholds -- replaces fixed
    70/30 levels used to gate momentum entries.

        upper(t) = rolling_mean(RSI, trailing lookback, up to t-1) + k * rolling_std
        lower(t) = rolling_mean(RSI, trailing lookback, up to t-1) - k * rolling_std

    The rolling window is shifted by one candle so a candle's own RSI value
    never contributes to the threshold it's being compared against.
    `upper`/`lower` are clipped to `[lower_bound, upper_bound]` (default
    5/95) since RSI is itself bounded 0-100.

    Before a full `lookback`-candle window of history exists (governed by
    `min_periods`), both thresholds are NaN -- comparisons against NaN
    thresholds evaluate to False, so the momentum entry gate simply
    doesn't fire until real, well-supported thresholds are available.

    Returns (upper, lower), each a pandas Series aligned to `rsi`.
    """
    rsi = pd.Series(rsi).reset_index(drop=True)
    roll_mean = rsi.rolling(lookback, min_periods=min_periods).mean().shift(1)
    roll_std = rsi.rolling(lookback, min_periods=min_periods).std().shift(1)

    upper = (roll_mean + k * roll_std).clip(lower=lower_bound, upper=upper_bound)
    lower = (roll_mean - k * roll_std).clip(lower=lower_bound, upper=upper_bound)
    return upper, lower


def compute_diff_uniformity(close, n=10):
    """
    Rolling directional uniformity of candle-to-candle price changes over
    the trailing `n` candles -- used as an optional "choppiness" gate on
    the Reversal engine's entries.

        pos_frac(t) = (# positive diffs in the trailing n diffs) / n

    - pos_frac near 1.0 or 0.0 -> uniformly trending (up or down).
    - pos_frac near 0.5 -> choppy / directionless recent price action.

    See `compute_choppiness_gate` for how this becomes a boolean filter.
    """
    close = pd.Series(close).reset_index(drop=True)
    diff = close.diff()
    pos_frac = (diff > 0).astype(float).rolling(n, min_periods=n).mean()
    return pos_frac


def compute_choppiness_gate(diff_pos_frac, uniformity_th=0.8):
    """
    Turns the rolling positive-diff fraction into a boolean "is choppy
    enough to fade" gate.

    Blocked (too uniform/trending) when `pos_frac >= uniformity_th` or
    `pos_frac <= 1 - uniformity_th`. Allowed otherwise.
    """
    pos_frac = pd.Series(diff_pos_frac).reset_index(drop=True)
    is_choppy = (pos_frac > (1.0 - uniformity_th)) & (pos_frac < uniformity_th)
    return is_choppy.fillna(False)


def compute_bollinger_bands(close, n=20, num_std=2.0):
    """
    Bollinger Bands from a rolling mean/std of closing price over the
    trailing `n` candles. `min_periods=n` so all three are NaN until a
    full window of history exists.

    This is the Reversal engine's band definition. ATR is still computed
    separately (see `compute_atr`) since it drives the reversal engine's
    hard stop-loss distance independently of the band type.
    """
    close = pd.Series(close).reset_index(drop=True)
    middle = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


# ==============================================================
# 2. VOLATILITY REGIME SWITCH (rolling z-score)
# ==============================================================

def compute_rolling_zscore_with_warmup(pred_vol_current_split, pred_vol_prior_split=None,
                                        lookback=672, min_periods=96):
    """
    Rolling (trailing, causal) z-score of predicted volatility against its
    own recent history. `pred_vol_prior_split` lets the rolling window
    "warm up" using the tail of the previous split (e.g. validation's tail
    feeding test's early z-scores) instead of restarting cold every split.
    """
    prior = pd.Series(pred_vol_prior_split).reset_index(drop=True) if pred_vol_prior_split is not None else pd.Series([], dtype=float)
    current = pd.Series(pred_vol_current_split).reset_index(drop=True)
    combined = pd.concat([prior, current], ignore_index=True)

    roll_mean = combined.rolling(lookback, min_periods=min_periods).mean()
    roll_std = combined.rolling(lookback, min_periods=min_periods).std()
    z_full = (combined - roll_mean) / roll_std.replace(0, np.nan)

    return z_full.iloc[len(prior):].reset_index(drop=True)


def classify_regime(z, z_threshold=0.8):
    """Z < threshold -> 'momentum'; Z >= threshold -> 'reversal';
    NaN (not enough history yet) -> 'insufficient_data' (hold cash)."""
    z = pd.Series(z).reset_index(drop=True)
    regime = np.where(z.isna(), "insufficient_data",
                       np.where(z >= z_threshold, "reversal", "momentum"))
    return pd.Series(regime)


# ==============================================================
# 3. FEATURES (one asset, causal by construction)
# ==============================================================

def compute_regime_features(pred_vol, close, high, low, volume=None,
                             vwap_reset_period=96, rvol_n=14,
                             bb_n=20, bb_num_std=2.0, atr_n=14,
                             gradient_n=5, choppiness_n=10, roc_n=12, rsi_n=14,
                             rsi_threshold_lookback=96, rsi_threshold_min_periods=20,
                             rsi_threshold_k=1.0, rsi_threshold_upper_bound=95.0,
                             rsi_threshold_lower_bound=5.0):
    close = pd.Series(close).reset_index(drop=True)
    high = pd.Series(high).reset_index(drop=True)
    low = pd.Series(low).reset_index(drop=True)
    pred_vol = pd.Series(pred_vol).reset_index(drop=True)
    volume = pd.Series(volume).reset_index(drop=True) if volume is not None else None

    pred_sigma = np.sqrt(pred_vol.clip(lower=0) / 6.0)

    vwap = compute_vwap(close, high, low, volume, reset_period=vwap_reset_period)
    rvol = compute_rvol(volume, close=close, high=high, low=low, n=rvol_n)
    bb_mid, bb_upper, bb_lower = compute_bollinger_bands(close, n=bb_n, num_std=bb_num_std)
    atr = compute_atr(high, low, close, atr_n)
    price_gradient = compute_price_gradient(close, n=gradient_n)
    diff_pos_frac = compute_diff_uniformity(close, n=choppiness_n)
    roc = compute_roc(close, n=roc_n)
    rsi = compute_rsi(close, n=rsi_n)
    rsi_upper_th, rsi_lower_th = compute_rsi_adaptive_thresholds(
        rsi, lookback=rsi_threshold_lookback, min_periods=rsi_threshold_min_periods,
        k=rsi_threshold_k, upper_bound=rsi_threshold_upper_bound, lower_bound=rsi_threshold_lower_bound)

    return pd.DataFrame({
        "close": close, "high": high, "low": low,
        "pred_vol": pred_vol, "pred_sigma": pred_sigma,
        "vwap": vwap, "rvol": rvol,
        "bb_mid": bb_mid, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "atr": atr, "price_gradient": price_gradient,
        "diff_pos_frac": diff_pos_frac, "roc": roc, "rsi": rsi,
        "rsi_upper_th": rsi_upper_th, "rsi_lower_th": rsi_lower_th,
    })


# ==============================================================
# 4. ENTRY SIGNALS -- Momentum and Reversal
# ==============================================================

def compute_momentum_signal(features):
    """
    RSI (Wilder) gates every momentum entry: RSI >= adaptive upper flags
    an upward trend and is required for a momentum LONG (also requires
    price above VWAP); RSI <= adaptive lower flags a downward trend and is
    required for a momentum SHORT (also requires price below VWAP).
    """
    f = features
    rsi_upper_trend = f["rsi"] >= f["rsi_upper_th"]   # upward-trend flag
    rsi_lower_trend = f["rsi"] <= f["rsi_lower_th"]   # downward-trend flag

    entry_long = (f["close"] > f["vwap"]) & rsi_upper_trend
    entry_short = (f["close"] < f["vwap"]) & rsi_lower_trend
    return entry_long.fillna(False), entry_short.fillna(False)


def compute_reversal_signal(features, use_gradient_filter=False,
                             use_choppiness_filter=False, choppiness_uniformity_th=0.8):
    """
    Bollinger Band fade: price above the upper band -> SHORT candidate,
    price below the lower band -> LONG candidate. Optionally confirmed by
    gradient deceleration and/or a choppiness (non-uniformity) gate.
    """
    f = features
    raw_short = f["close"] > f["bb_upper"]
    raw_long = f["close"] < f["bb_lower"]

    if use_gradient_filter and "price_gradient" in f.columns:
        grad = f["price_gradient"]
        grad_prev = grad.shift(1)
        short_decelerating = (grad < grad_prev).fillna(False)
        long_decelerating = (grad > grad_prev).fillna(False)
        entry_short = raw_short & short_decelerating
        entry_long = raw_long & long_decelerating
    else:
        entry_short = raw_short
        entry_long = raw_long

    if use_choppiness_filter and "diff_pos_frac" in f.columns:
        is_choppy = compute_choppiness_gate(f["diff_pos_frac"], uniformity_th=choppiness_uniformity_th)
        entry_short = entry_short & is_choppy
        entry_long = entry_long & is_choppy

    return entry_long.fillna(False), entry_short.fillna(False)


# ==============================================================
# 5. THE STATE MACHINE (turns signals into a signed position series)
# ==============================================================

def run_regime_switch_strategy(
    features, z, regime_label,
    mom_entry_long, mom_entry_short, rev_entry_long, rev_entry_short,
    target_risk_frac=0.004, max_leverage_per_asset=1.5, stop_atr_mult=1.0,
    min_hold_candles=3, reversal_profit_gate=False,
    reversal_uses_gradient_filter=False, reversal_uses_choppiness_filter=False,
    momentum_exit_close_frac=0.9, z_threshold=0.8
):
    """
    Momentum engine exit: RSI peak-reversal is the ONLY momentum exit
    trigger (unconditional, fires once `min_hold_candles` has elapsed,
    regardless of P&L). The TRANSACTED SIZE at that exit is
    price-conditional: closing a long only sells `momentum_exit_close_frac`
    of the position if current price > entry price (else sells nothing);
    closing a short only buys back that fraction if current price < entry
    price (else buys back nothing).

    Reversal engine exit: stop-loss (unconditional ATR-based hard stop) OR,
    once `min_hold_candles` has elapsed, EITHER a Bollinger mid-line
    snapback OR the predicted-volatility z-score reverting back below
    `z_threshold` -- these two are OR'd together. The transacted-size logic
    at exit is the same price-conditional pattern as momentum (with
    `close_frac` fixed at 1.0 when price is favorable).
    """
    n = len(features)
    close = features["close"].values
    vwap = features["vwap"].values
    bb_mid = features["bb_mid"].values
    atr = features["atr"].values
    high = features["high"].values
    low = features["low"].values
    sigma_v = features["pred_sigma"].values
    roc = features["roc"].values if "roc" in features.columns else np.full(n, np.nan)
    rsi = features["rsi"].values if "rsi" in features.columns else np.full(n, np.nan)

    regime_v = pd.Series(regime_label).values
    z_v = pd.Series(z).values

    mom_el, mom_es = pd.Series(mom_entry_long).values, pd.Series(mom_entry_short).values
    rev_el, rev_es = pd.Series(rev_entry_long).values, pd.Series(rev_entry_short).values

    STATE_FLAT, STATE_LONG, STATE_SHORT = "FLAT", "LONG", "SHORT"
    state = STATE_FLAT
    position = 0.0            # signed fraction of capital: + long, - short
    position_regime = None    # "momentum" or "reversal" -- locked in at entry
    stop_price = None         # reversal-engine hard stop only
    entry_bar = None
    entry_price = None
    entry_extreme_rsi = None

    signals_out = [None] * n
    trade_units = [0.0] * n
    positions = [0.0] * n
    states = [STATE_FLAT] * n
    regimes_out = [None] * n
    reasons = [None] * n
    exit_actions = [None] * n

    for i in range(n):
        px = close[i]

        # ---- Step 1: manage an open position -- exits first, per its own locked-in regime ----
        if state != STATE_FLAT:
            cur_dir = 1.0 if state == STATE_LONG else -1.0
            held = i - entry_bar
            past_cooldown = held >= min_hold_candles
            exited = False

            if position_regime == "momentum":
                cur_rsi = rsi[i]
                rsi_dropping = False
                if pd.notna(cur_rsi) and pd.notna(entry_extreme_rsi):
                    if cur_dir > 0:
                        rsi_dropping = cur_rsi < entry_extreme_rsi
                    else:
                        rsi_dropping = cur_rsi > entry_extreme_rsi

                trigger = rsi_dropping
                fire = past_cooldown and trigger

                if fire:
                    reasons[i] = "rsi_peak_reversal"
                    exited = True
                elif pd.notna(cur_rsi):
                    if entry_extreme_rsi is None:
                        entry_extreme_rsi = cur_rsi
                    elif cur_dir > 0:
                        entry_extreme_rsi = max(entry_extreme_rsi, cur_rsi)
                    else:
                        entry_extreme_rsi = min(entry_extreme_rsi, cur_rsi)

            elif position_regime == "reversal":
                hit_stop = (cur_dir < 0 and px > stop_price) or (cur_dir > 0 and px < stop_price)
                snapback = (cur_dir < 0 and px <= bb_mid[i]) or (cur_dir > 0 and px >= bb_mid[i])
                cur_z = z_v[i]
                vol_regime_reverted = pd.notna(cur_z) and cur_z < z_threshold

                if hit_stop:
                    reasons[i] = "stop_loss"
                    exited = True
                elif past_cooldown and (snapback or vol_regime_reverted):
                    reasons[i] = "take_profit_snapback" if snapback else "vol_regime_reverted"
                    exited = True

            # ---- Decide the TRANSACTED quantity + closing buy/sell action ----
            if exited:
                if position_regime in ["momentum", "reversal"]:
                    if cur_dir > 0:
                        transact = px > entry_price
                    else:
                        transact = px < entry_price

                    if position_regime == "momentum":
                        close_frac = momentum_exit_close_frac if transact else 0.0
                    else:
                        close_frac = 1.0 if transact else 0.0
                else:
                    close_frac = 1.0

                qty_closed = position * close_frac

                if close_frac > 0:
                    pnl_pct = ((px / entry_price) - 1.0) * 100.0 if cur_dir > 0 else ((entry_price / px) - 1.0) * 100.0
                    price_relation = ">" if px > entry_price else ("<" if px < entry_price else "==")
                    if cur_dir > 0:
                        exit_actions[i] = "SELL"
                        action_note = (f"bought at entry ${entry_price:.2f}; current price ${px:.2f} {price_relation} entry "
                                       f"-> SELL {close_frac * 100:.0f}% of position to close")
                    else:
                        exit_actions[i] = "BUY_TO_COVER"
                        action_note = (f"sold (opened short) at entry ${entry_price:.2f}; current price ${px:.2f} {price_relation} entry "
                                       f"-> BUY BACK (cover) {close_frac * 100:.0f}% of position to close")
                else:
                    pnl_pct = 0.0
                    exit_actions[i] = "NO_ACTION"
                    if cur_dir > 0:
                        action_note = (f"bought at entry ${entry_price:.2f}; current price ${px:.2f} <= entry "
                                       f"-> NO SELL (price not favorable), position bookkept as closed")
                    else:
                        action_note = (f"sold (opened short) at entry ${entry_price:.2f}; current price ${px:.2f} >= entry "
                                       f"-> NO BUY BACK (price not favorable), position bookkept as closed")

                if position_regime in ["momentum", "reversal"]:
                    gate_note = ("trigger unconditional (fires regardless of P&L); transacted size is "
                                 "price-conditional -- see action note")
                    reasons[i] = f"{reasons[i]}||{gate_note}; {action_note}; P&L {pnl_pct:+.2f}%"
                else:
                    reasons[i] = f"{reasons[i]}||{action_note}; P&L {pnl_pct:+.2f}%"

                trade_units[i] = -qty_closed
                signals_out[i] = "EXIT"
                position = 0.0
                position_regime = None
                stop_price = None
                entry_bar = None
                entry_price = None
                entry_extreme_rsi = None
                state = STATE_FLAT

        # ---- Step 2/3/4: if flat, route by this candle's regime and look for a new entry ----
        if state == STATE_FLAT:
            reg = regime_v[i]
            sigma = sigma_v[i]
            entered = False

            if reg == "momentum" and pd.notna(sigma) and sigma > 0:
                if mom_el[i] or mom_es[i]:
                    direction = 1.0 if mom_el[i] else -1.0
                    raw_size = target_risk_frac / sigma
                    position = direction * min(raw_size, max_leverage_per_asset)
                    position_regime = "momentum"
                    state = STATE_LONG if direction > 0 else STATE_SHORT
                    entry_bar = i
                    entry_price = px
                    entry_extreme_rsi = rsi[i] if pd.notna(rsi[i]) else None
                    entered = True
                    signals_out[i] = "ENTRY_LONG" if direction > 0 else "ENTRY_SHORT"
                    rsi_note = "RSI>=adaptive_upper (upward trend)" if direction > 0 else "RSI<=adaptive_lower (downward trend)"
                    reasons[i] = f"momentum entry (VWAP-aligned, {rsi_note}), size={position:.3f}"

            elif reg == "reversal" and pd.notna(sigma) and sigma > 0:
                if rev_el[i] or rev_es[i]:
                    direction = 1.0 if rev_el[i] else -1.0
                    raw_size = target_risk_frac / sigma
                    position = direction * min(raw_size, max_leverage_per_asset)
                    position_regime = "reversal"
                    state = STATE_LONG if direction > 0 else STATE_SHORT
                    entry_extreme = low[i] if direction > 0 else high[i]
                    stop_price = entry_extreme - stop_atr_mult * atr[i] if direction > 0 else entry_extreme + stop_atr_mult * atr[i]
                    entry_bar = i
                    entry_price = px
                    entered = True
                    signals_out[i] = "ENTRY_LONG" if direction > 0 else "ENTRY_SHORT"
                    gradient_note = " + gradient deceleration confirmed" if reversal_uses_gradient_filter else ""
                    choppiness_note = " + choppiness (non-uniform recent price action) confirmed" if reversal_uses_choppiness_filter else ""
                    reasons[i] = f"reversal entry (Bollinger fade{gradient_note}{choppiness_note}), size={position:.3f}"

            if entered:
                trade_units[i] += position
            elif signals_out[i] is None:
                signals_out[i] = "HOLD_CASH"
                reasons[i] = reasons[i] or (
                    "insufficient volatility history" if reg == "insufficient_data" else "no confirmed signal -- noise"
                )
        elif signals_out[i] is None:
            signals_out[i] = "HOLD_POSITION"
            reasons[i] = reasons[i] or "mid-trade"

        positions[i] = position
        states[i] = state
        regimes_out[i] = position_regime if state != STATE_FLAT else regime_v[i]

    return pd.DataFrame({
        "close": close, "pred_sigma": sigma_v, "z_score": z_v, "roc": roc, "rsi": rsi,
        "live_regime": regime_v, "position_regime": regimes_out,
        "signal": signals_out, "trade_units": trade_units, "position_units": positions,
        "state": states, "reason": reasons, "exit_action": exit_actions,
    })


# ==============================================================
# 6. PER-ASSET, PER-SPLIT SIGNAL-GENERATION CONVENIENCE WRAPPER
# ==============================================================

def generate_strategy_for_asset(pred_vol, close, high, low, volume, prior_pred_vol,
                                 strategy_cfg):
    """
    One-asset convenience wrapper that runs the full strategy stack
    (features -> z-score/regime -> entry signals -> state machine) using a
    `StrategyConfig`-like object (see config.py). Returns
    (strategy_df, features_df, z_score_series, regime_series) so callers
    (e.g. `pipeline.py`) can carry the z-score forward as the next split's
    warmup.
    """
    cfg = strategy_cfg

    features = compute_regime_features(
        pred_vol, close, high, low, volume=volume,
        vwap_reset_period=cfg.vwap_reset_period, rvol_n=cfg.rvol_n,
        bb_n=cfg.bb_n, bb_num_std=cfg.bb_num_std, atr_n=cfg.atr_n,
        gradient_n=cfg.gradient_n, choppiness_n=cfg.choppiness_n, roc_n=cfg.roc_n,
        rsi_n=cfg.rsi_n, rsi_threshold_lookback=cfg.rsi_threshold_lookback,
        rsi_threshold_k=cfg.rsi_threshold_k)

    z = compute_rolling_zscore_with_warmup(
        pred_vol, prior_pred_vol, cfg.lookback_candles, cfg.z_min_periods)
    regime = classify_regime(z, cfg.z_threshold)

    mom_el, mom_es = compute_momentum_signal(features)
    rev_el, rev_es = compute_reversal_signal(
        features, use_gradient_filter=cfg.reversal_gradient_filter,
        use_choppiness_filter=cfg.reversal_choppiness_filter,
        choppiness_uniformity_th=cfg.choppiness_uniformity_th)

    strategy_df = run_regime_switch_strategy(
        features, z, regime, mom_el, mom_es, rev_el, rev_es,
        target_risk_frac=cfg.target_risk_frac, max_leverage_per_asset=cfg.max_leverage_per_asset,
        stop_atr_mult=cfg.stop_atr_mult, min_hold_candles=cfg.min_hold_candles,
        reversal_profit_gate=cfg.reversal_profit_gate,
        reversal_uses_gradient_filter=cfg.reversal_gradient_filter,
        reversal_uses_choppiness_filter=cfg.reversal_choppiness_filter,
        momentum_exit_close_frac=cfg.momentum_exit_close_frac, z_threshold=cfg.z_threshold)

    return strategy_df, features, z, regime
