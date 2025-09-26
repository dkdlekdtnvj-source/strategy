"""Python backtest implementation mirroring the Pine strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import Trade, aggregate_metrics


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=max(length, 1), adjust=False).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    length = max(length, 1)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return _rma(tr, length)


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1.0 / length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1.0 / length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stoch(df: pd.DataFrame, length: int = 14) -> pd.Series:
    lowest = df["low"].rolling(length).min()
    highest = df["high"].rolling(length).max()
    k = 100 * (df["close"] - lowest) / (highest - lowest)
    return k.rolling(3).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return pd.DataFrame({"macd": macd_line, "signal": signal_line})


def _estimate_tick(prices: pd.Series) -> float:
    diffs = prices.diff().abs()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float(prices.iloc[-1]) * 1e-6 if len(prices) else 0.01
    return float(diffs.min())


def _align_htf(index: pd.Index, htf_df: pd.DataFrame, ema_len: int) -> pd.Series:
    if htf_df is None or htf_df.empty:
        return pd.Series(False, index=index, dtype=bool)
    aligned = htf_df["close"].reindex(index, method="ffill")
    confirmed = aligned.notna()
    aligned = aligned.fillna(method="ffill")
    trend = _ema(aligned, ema_len)
    return confirmed & (aligned >= trend)


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = ha_close.copy()
    if not ha_open.empty:
        ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
        for i in range(1, len(ha_open)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({
        "open": ha_open,
        "high": ha_high,
        "low": ha_low,
        "close": ha_close,
    })


def _linreg(series: pd.Series, length: int) -> pd.Series:
    length = max(length, 1)
    if length == 1:
        return series
    idx = np.arange(length)
    denom = ((idx - idx.mean()) ** 2).sum()

    if denom == 0:
        return series

    def calc(window: np.ndarray) -> float:
        y = window
        x = idx
        y_mean = y.mean()
        x_mean = x.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        intercept = y_mean - slope * x_mean
        return intercept + slope * x[-1]

    return series.rolling(length).apply(calc, raw=True)


def _directional_flux(
    df: pd.DataFrame, length: int, smooth: int, use_heikin: bool
) -> pd.Series:
    length = max(length, 1)
    smooth = max(smooth, 1)
    source = _heikin_ashi(df) if use_heikin else df
    high = source["high"]
    low = source["low"]
    close = source["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > 0) & (up_move > down_move), up_move, 0.0)
    minus_dm = np.where((down_move > 0) & (down_move > up_move), down_move, 0.0)
    plus_series = pd.Series(plus_dm, index=df.index)
    minus_series = pd.Series(minus_dm, index=df.index)

    atr_df = pd.DataFrame({"high": high, "low": low, "close": close})
    atr = _atr(atr_df, length).replace(0, np.nan)

    flux = (_rma(plus_series, length) - _rma(minus_series, length)) / atr
    flux = flux.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if smooth > 1:
        flux = flux.rolling(smooth).mean()
    return flux.fillna(0.0)


def _parse_event_windows(value: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if not value:
        return windows
    for segment in value.split(","):
        seg = segment.strip()
        if not seg:
            continue
        if "/" not in seg:
            continue
        start_str, end_str = seg.split("/", 1)
        try:
            start = pd.to_datetime(start_str.strip(), utc=True)
            end = pd.to_datetime(end_str.strip(), utc=True)
        except ValueError:
            continue
        if pd.isna(start) or pd.isna(end):
            continue
        if end < start:
            start, end = end, start
        windows.append((start, end))
    return windows


def _pivot_series(series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
    if left < 1 or right < 1:
        return pd.Series(np.nan, index=series.index)
    length = len(series)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    values = series.values
    for idx in range(left, length - right):
        window = values[idx - left : idx + right + 1]
        center = window[left]
        if is_high and center == window.max():
            result.iloc[idx + right] = center
        elif not is_high and center == window.min():
            result.iloc[idx + right] = center
    return result.ffill()


@dataclass
class BacktestContext:
    """Holds state for the running backtest."""

    position: int = 0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    mfe: float = 0.0
    mae: float = 0.0
    bars_held: int = 0
    highest_price: float = 0.0
    lowest_price: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, float | bool],
    fees: Dict[str, float],
    risk: Dict[str, float],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError("DataFrame must contain OHLCV columns")

    osc_len = int(params.get("len", params.get("wtLen", 14)))
    sig_len = int(params.get("sig", 3))
    bb_len = int(params.get("bbLen", params.get("sqz_bbLen", 20)))
    kc_len = int(params.get("kcLen", params.get("sqz_kcLen", 18)))
    bb_mult = float(params.get("bbMult", params.get("sqz_bbMult", 1.4)))
    kc_mult = float(params.get("kcMult", params.get("sqz_kcMult", 1.0)))
    flux_len = int(params.get("dfl", 14))
    flux_smooth = int(params.get("dfSmoothLen", 1))
    use_heikin = bool(params.get("dfh", params.get("useHeikin", True)))

    use_sym_threshold = bool(params.get("useSymThreshold", False))
    use_dynamic_threshold = bool(params.get("useDynamicThresh", True))
    stat_threshold = float(params.get("statThreshold", 38.0))
    buy_threshold = float(params.get("buyThreshold", 36.0))
    sell_threshold = float(params.get("sellThreshold", 36.0))
    dyn_len = int(params.get("dynLen", 21))
    dyn_mult = float(params.get("dynMult", 1.1))

    exit_opposite = bool(params.get("exitOpposite", True))
    use_mom_fade = bool(params.get("useMomFade", False))
    mom_fade_bars = int(params.get("momFadeBars", 1))
    mom_fade_len = int(params.get("momFadeLen", max(1, osc_len)))
    fade_mode = str(params.get("fadeMode", "VN")).upper()

    atr_mult = float(params.get("atrMult", 1.5))
    use_htf = bool(params.get("useHTF", params.get("useHtfTrend", False)))
    use_rsi = bool(params.get("useRSI", False))
    use_macd = bool(params.get("useMACD", False))
    use_stoch = bool(params.get("useStoch", False))
    htf_ema_len = int(params.get("htfMaLen", params.get("htfEmaLen", 20)))
    use_event_filter = bool(params.get("useEventFilter", False))
    event_windows_raw = str(params.get("eventWindows", "")) if use_event_filter else ""
    use_time_stop = bool(params.get("useTimeStop", False))
    max_hold_param = int(params.get("maxHoldBars", 0))
    use_breakeven = bool(params.get("useBreakevenStop", False))
    breakeven_mult = float(params.get("breakevenMult", 1.0))
    use_atr_trail = bool(params.get("useAtrTrail", False))
    atr_trail_len = int(params.get("atrTrailLen", max(1, osc_len)))
    atr_trail_mult = float(params.get("atrTrailMult", 2.0))
    use_pivot_stop = bool(params.get("usePivotStop", False))
    pivot_len = int(params.get("pivotLen", 5))
    use_atr_stop = bool(params.get("useAtrStop", True))
    use_fixed_stop = bool(params.get("useFixedStop", False))
    fixed_stop_pct = float(params.get("fixedStopPct", 0.0)) / 100.0
    use_stop_loss = bool(params.get("useStopLoss", False))
    stop_lookback = int(params.get("stopLookback", 5))

    commission = float(fees.get("commission_pct", 0.0))
    slippage_ticks = float(fees.get("slippage_ticks", 0.0))

    leverage = float(risk.get("leverage", 1.0))
    qty_pct = float(risk.get("qty_pct", 10.0)) / 100.0
    liq_buffer_pct = float(risk.get("liq_buffer_pct", 0.0)) / 100.0

    trade_penalty = float(risk.get("penalty_trade", 1.0))
    hold_penalty = float(risk.get("penalty_hold", 1.0))
    loss_penalty = float(risk.get("penalty_consecutive_loss", 1.0))

    min_trades = int(risk.get("min_trades", min_trades if min_trades is not None else 100))
    min_hold_bars = int(risk.get("min_hold_bars", 1))
    max_consec_losses = int(risk.get("max_consecutive_losses", 5))

    atr = _atr(df, kc_len).bfill().replace(0, np.nan)
    hl2 = (df["high"] + df["low"]) / 2.0
    bb_basis = hl2.rolling(bb_len).mean()
    kc_range = _atr(df, kc_len) * kc_mult
    normalized = (df["close"] - bb_basis) / kc_range.replace(0, np.nan)
    normalized = normalized.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    osc_raw = _linreg(normalized, osc_len) * 100.0
    osc = osc_raw.fillna(0.0)
    if sig_len > 1:
        osc_signal = osc.rolling(sig_len).mean()
    else:
        osc_signal = osc.copy()

    flux = _directional_flux(df, flux_len, flux_smooth, use_heikin)

    dyn_series = osc.rolling(max(1, dyn_len)).std() * dyn_mult
    base_level = abs(stat_threshold)
    buy_level = pd.Series(-abs(buy_threshold), index=df.index)
    sell_level = pd.Series(abs(sell_threshold), index=df.index)
    if use_dynamic_threshold:
        dyn_level = dyn_series.fillna(base_level)
        buy_level = -dyn_level
        sell_level = dyn_level
    elif use_sym_threshold:
        buy_level = pd.Series(-base_level, index=df.index)
        sell_level = pd.Series(base_level, index=df.index)

    fade_window = max(1, mom_fade_len)
    if fade_mode == "LEGACY":
        fade_source = osc
        fade_metric = fade_source.abs()
        fade_long_series = (fade_source > 0) & (fade_metric <= fade_metric.shift())
        fade_short_series = (fade_source < 0) & (fade_metric <= fade_metric.shift())
    else:
        fade_source = osc.rolling(fade_window).mean()
        fade_metric = fade_source.abs().rolling(max(1, mom_fade_bars)).mean()
        fade_mag_down = fade_metric <= fade_metric.shift()
        fade_long_series = (fade_source > 0) & fade_mag_down
        fade_short_series = (fade_source < 0) & fade_mag_down

    cross_up = (osc > osc_signal) & (osc.shift(1) <= osc_signal.shift(1))
    cross_down = (osc < osc_signal) & (osc.shift(1) >= osc_signal.shift(1))
    long_signal_series = cross_up & (osc <= buy_level) & (flux > 0)
    short_signal_series = cross_down & (osc >= sell_level) & (flux < 0)

    signal_frame = pd.DataFrame(
        {
            "osc": osc,
            "osc_signal": osc_signal,
            "flux": flux,
            "long_signal": long_signal_series.fillna(False).astype(bool),
            "short_signal": short_signal_series.fillna(False).astype(bool),
            "fade_long": fade_long_series.fillna(False).astype(bool),
            "fade_short": fade_short_series.fillna(False).astype(bool),
        }
    )

    rsi = _rsi(df["close"]) if use_rsi else pd.Series(index=df.index, dtype=float)
    macd = _macd(df["close"]) if use_macd else pd.DataFrame(index=df.index)
    stoch = _stoch(df) if use_stoch else pd.Series(index=df.index, dtype=float)
    htf_ok = (
        _align_htf(df.index, htf_df, htf_ema_len)
        if use_htf
        else pd.Series(True, index=df.index, dtype=bool)
    )
    if use_htf and (htf_df is None or htf_df.empty):
        raise ValueError("useHTF=True requires htf_df data to be provided")

    event_windows = _parse_event_windows(event_windows_raw)
    event_mask = pd.Series(False, index=df.index, dtype=bool)
    if event_windows:
        idx_ns = df.index.view("int64")
        mask_array = np.zeros(len(df.index), dtype=bool)
        for start, end in event_windows:
            start_ns = pd.Timestamp(start).value
            end_ns = pd.Timestamp(end).value
            if end_ns < start_ns:
                start_ns, end_ns = end_ns, start_ns
            mask_array |= (idx_ns >= start_ns) & (idx_ns <= end_ns)
        event_mask = pd.Series(mask_array, index=df.index)

    atr_trail_series = (
        _atr(df, max(1, atr_trail_len))
        if (use_atr_trail or use_breakeven)
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    if use_pivot_stop:
        pivot_low_series = _pivot_series(df["low"], pivot_len, pivot_len, is_high=False)
        pivot_high_series = _pivot_series(df["high"], pivot_len, pivot_len, is_high=True)
    else:
        pivot_low_series = pd.Series(np.nan, index=df.index, dtype=float)
        pivot_high_series = pd.Series(np.nan, index=df.index, dtype=float)

    if use_stop_loss and stop_lookback > 0:
        swing_low_series = df["low"].rolling(stop_lookback).min()
        swing_high_series = df["high"].rolling(stop_lookback).max()
    else:
        swing_low_series = pd.Series(np.nan, index=df.index, dtype=float)
        swing_high_series = pd.Series(np.nan, index=df.index, dtype=float)

    tick_size = _estimate_tick(df["close"])
    slippage_value = tick_size * slippage_ticks

    returns = pd.Series(0.0, index=df.index)
    trades: List[Trade] = []
    ctx = BacktestContext()
    max_hold_bars = max(0, max_hold_param if use_time_stop else 0)

    def close_position(exit_time: pd.Timestamp, exit_price: float, reason: str) -> None:
        if ctx.position == 0 or ctx.entry_time is None:
            return
        nonlocal returns
        direction = "long" if ctx.position > 0 else "short"
        adj_exit = exit_price - slippage_value if ctx.position > 0 else exit_price + slippage_value
        notional = qty_pct * leverage
        price_diff = (adj_exit - ctx.entry_price) / ctx.entry_price
        gross_pct = price_diff * ctx.position * leverage
        equity_return = gross_pct * qty_pct
        equity_return -= commission * notional * 2
        returns.loc[exit_time] += equity_return
        trades.append(
            Trade(
                entry_time=ctx.entry_time,
                exit_time=exit_time,
                direction=direction,
                size=notional,
                entry_price=ctx.entry_price,
                exit_price=adj_exit,
                profit=equity_return,
                return_pct=gross_pct,
                mfe=ctx.mfe,
                mae=ctx.mae,
                bars_held=ctx.bars_held,
                reason=reason,
            )
        )
        ctx.position = 0
        ctx.entry_price = 0.0
        ctx.entry_time = None
        ctx.mfe = 0.0
        ctx.mae = 0.0
        ctx.bars_held = 0
        ctx.highest_price = 0.0
        ctx.lowest_price = 0.0

    for ts, row in df.iterrows():
        atr_val = atr.loc[ts]
        if np.isnan(atr_val):
            continue

        long_sig = bool(signal_frame.at[ts, "long_signal"])
        short_sig = bool(signal_frame.at[ts, "short_signal"])
        fade_long_sig = bool(signal_frame.at[ts, "fade_long"]) if use_mom_fade else False
        fade_short_sig = bool(signal_frame.at[ts, "fade_short"]) if use_mom_fade else False

        if use_rsi:
            rsi_val = rsi.loc[ts]
            if np.isnan(rsi_val):
                long_sig = False
                short_sig = False
            else:
                long_sig = long_sig and rsi_val > 50
                short_sig = short_sig and rsi_val < 50

        if use_macd:
            macd_val = macd.loc[ts, "macd"]
            macd_signal = macd.loc[ts, "signal"]
            if np.isnan(macd_val) or np.isnan(macd_signal):
                long_sig = False
                short_sig = False
            else:
                long_sig = long_sig and macd_val > macd_signal
                short_sig = short_sig and macd_val < macd_signal

        if use_stoch:
            stoch_val = stoch.loc[ts]
            if np.isnan(stoch_val):
                long_sig = False
                short_sig = False
            else:
                long_sig = long_sig and stoch_val > 50
                short_sig = short_sig and stoch_val < 50

        htf_pass = bool(htf_ok.loc[ts]) if use_htf else True
        event_blocked = bool(event_mask.loc[ts]) if use_event_filter else False

        if ctx.position != 0:
            ctx.bars_held += 1
            if ctx.position > 0:
                favorable = (row["high"] - ctx.entry_price) / ctx.entry_price * leverage
                adverse = (row["low"] - ctx.entry_price) / ctx.entry_price * leverage
                ctx.highest_price = max(ctx.highest_price, row["high"])
                ctx.lowest_price = row["low"] if ctx.lowest_price == 0.0 else min(ctx.lowest_price, row["low"])
            else:
                favorable = (ctx.entry_price - row["low"]) / ctx.entry_price * leverage
                adverse = (ctx.entry_price - row["high"]) / ctx.entry_price * leverage
                ctx.lowest_price = row["low"] if ctx.lowest_price == 0.0 else min(ctx.lowest_price, row["low"])
                ctx.highest_price = max(ctx.highest_price, row["high"])
            ctx.mfe = max(ctx.mfe, favorable)
            ctx.mae = min(ctx.mae, -abs(adverse))

        exit_reason: Optional[str] = None
        exit_price = row["close"]
        allow_signal_exit = ctx.bars_held >= min_hold_bars
        atr_trail_val = atr_trail_series.loc[ts] if (use_atr_trail or use_breakeven) else atr_val
        pivot_low_val = pivot_low_series.loc[ts]
        pivot_high_val = pivot_high_series.loc[ts]
        if ctx.position > 0:
            stop_price = ctx.entry_price - atr_mult * atr_val if use_atr_stop else None
            guard_price = ctx.entry_price * (1 - liq_buffer_pct)
            if liq_buffer_pct > 0 and row["low"] <= guard_price:
                exit_reason = "liq_guard"
                exit_price = guard_price
            else:
                breakeven_ready = (
                    use_breakeven
                    and ctx.highest_price > 0
                    and not np.isnan(atr_trail_val)
                    and (ctx.highest_price - ctx.entry_price) >= breakeven_mult * atr_trail_val
                )
                if breakeven_ready and row["low"] <= ctx.entry_price:
                    exit_reason = "breakeven"
                    exit_price = ctx.entry_price
                elif (
                    exit_reason is None
                    and use_stop_loss
                    and not np.isnan(swing_low_series.loc[ts])
                    and row["low"] <= swing_low_series.loc[ts]
                ):
                    exit_reason = "swing_stop"
                    exit_price = swing_low_series.loc[ts]
                elif (
                    exit_reason is None
                    and use_pivot_stop
                    and not np.isnan(pivot_low_val)
                    and row["low"] <= pivot_low_val
                ):
                    exit_reason = "pivot_stop"
                    exit_price = pivot_low_val
                elif (
                    exit_reason is None
                    and use_atr_trail
                    and not np.isnan(atr_trail_val)
                    and ctx.highest_price > 0
                ):
                    trail_stop = ctx.highest_price - atr_trail_mult * atr_trail_val
                    if row["low"] <= trail_stop:
                        exit_reason = "atr_trail"
                        exit_price = trail_stop
                if exit_reason is None:
                    candidate_price: Optional[float] = None
                    candidate_reason: Optional[str] = None
                    if use_fixed_stop and fixed_stop_pct > 0:
                        fixed_price = ctx.entry_price * (1 - fixed_stop_pct)
                        if row["low"] <= fixed_price:
                            candidate_price = fixed_price
                            candidate_reason = "fixed_stop"
                    if (
                        use_atr_stop
                        and stop_price is not None
                        and row["low"] <= stop_price
                        and (candidate_price is None or stop_price > candidate_price)
                    ):
                        candidate_price = stop_price
                        candidate_reason = "atr_stop"
                    if candidate_reason:
                        exit_reason = candidate_reason
                        exit_price = candidate_price  # type: ignore[assignment]
                if exit_reason is None and max_hold_bars and ctx.bars_held >= max_hold_bars:
                    exit_reason = "time_stop"
                elif exit_reason is None and allow_signal_exit and htf_pass:
                    if exit_opposite and short_sig:
                        exit_reason = "reverse_short"
                    elif use_mom_fade and fade_long_sig:
                        exit_reason = "mom_fade"
        elif ctx.position < 0:
            stop_price = ctx.entry_price + atr_mult * atr_val if use_atr_stop else None
            guard_price = ctx.entry_price * (1 + liq_buffer_pct)
            if liq_buffer_pct > 0 and row["high"] >= guard_price:
                exit_reason = "liq_guard"
                exit_price = guard_price
            else:
                breakeven_ready = (
                    use_breakeven
                    and ctx.lowest_price > 0
                    and not np.isnan(atr_trail_val)
                    and (ctx.entry_price - ctx.lowest_price) >= breakeven_mult * atr_trail_val
                )
                if breakeven_ready and row["high"] >= ctx.entry_price:
                    exit_reason = "breakeven"
                    exit_price = ctx.entry_price
                elif (
                    exit_reason is None
                    and use_stop_loss
                    and not np.isnan(swing_high_series.loc[ts])
                    and row["high"] >= swing_high_series.loc[ts]
                ):
                    exit_reason = "swing_stop"
                    exit_price = swing_high_series.loc[ts]
                elif (
                    exit_reason is None
                    and use_pivot_stop
                    and not np.isnan(pivot_high_val)
                    and row["high"] >= pivot_high_val
                ):
                    exit_reason = "pivot_stop"
                    exit_price = pivot_high_val
                elif (
                    exit_reason is None
                    and use_atr_trail
                    and not np.isnan(atr_trail_val)
                    and ctx.lowest_price > 0
                ):
                    trail_stop = ctx.lowest_price + atr_trail_mult * atr_trail_val
                    if row["high"] >= trail_stop:
                        exit_reason = "atr_trail"
                        exit_price = trail_stop
                if exit_reason is None:
                    candidate_price = None
                    candidate_reason = None
                    if use_fixed_stop and fixed_stop_pct > 0:
                        fixed_price = ctx.entry_price * (1 + fixed_stop_pct)
                        if row["high"] >= fixed_price:
                            candidate_price = fixed_price
                            candidate_reason = "fixed_stop"
                    if (
                        use_atr_stop
                        and stop_price is not None
                        and row["high"] >= stop_price
                        and (candidate_price is None or stop_price < candidate_price)
                    ):
                        candidate_price = stop_price
                        candidate_reason = "atr_stop"
                    if candidate_reason:
                        exit_reason = candidate_reason
                        exit_price = candidate_price  # type: ignore[assignment]
                if exit_reason is None and max_hold_bars and ctx.bars_held >= max_hold_bars:
                    exit_reason = "time_stop"
                elif exit_reason is None and allow_signal_exit and htf_pass:
                    if exit_opposite and long_sig:
                        exit_reason = "reverse_long"
                    elif use_mom_fade and fade_short_sig:
                        exit_reason = "mom_fade"

        if exit_reason:
            close_position(ts, exit_price, exit_reason)

        if ctx.position == 0 and htf_pass and not event_blocked:
            if long_sig and not short_sig:
                ctx.position = 1
                ctx.entry_price = row["close"] + slippage_value
                ctx.entry_time = ts
                ctx.highest_price = row["high"]
                ctx.lowest_price = row["low"]
            elif short_sig and not long_sig:
                ctx.position = -1
                ctx.entry_price = row["close"] - slippage_value
                ctx.entry_time = ts
                ctx.highest_price = row["high"]
                ctx.lowest_price = row["low"]

    if ctx.position != 0 and ctx.entry_time is not None:
        close_position(df.index[-1], df.iloc[-1]["close"], "end_of_data")

    metrics = aggregate_metrics(trades, returns)
    metrics["Returns"] = returns
    metrics["TradesList"] = trades
    metrics["Trades"] = int(metrics.get("Trades", 0))
    metrics["Wins"] = int(metrics.get("Wins", 0))
    metrics["Losses"] = int(metrics.get("Losses", 0))
    metrics["MinTrades"] = float(min_trades)
    metrics["MinHoldBars"] = float(min_hold_bars)
    metrics["MaxConsecutiveLossLimit"] = float(max_consec_losses)
    metrics["TradePenalty"] = trade_penalty
    metrics["HoldPenalty"] = hold_penalty
    metrics["ConsecutiveLossPenalty"] = loss_penalty
    metrics["Valid"] = (
        metrics["Trades"] >= min_trades
        and metrics.get("AvgHoldBars", 0.0) >= min_hold_bars
        and metrics.get("MaxConsecutiveLosses", 0.0) <= max_consec_losses
    )
    return metrics
