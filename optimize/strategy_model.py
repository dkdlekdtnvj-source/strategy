"""Python backtest implementation mirroring the Pine strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import Trade, aggregate_metrics


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
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

    wt_len = int(params.get("wtLen", 14))
    thr = float(params.get("thr", 0.6))
    atr_mult = float(params.get("atrMult", 1.5))
    use_htf = bool(params.get("useHTF", True))
    use_rsi = bool(params.get("useRSI", True))
    use_macd = bool(params.get("useMACD", True))
    use_stoch = bool(params.get("useStoch", True))
    htf_ema_len = int(params.get("htfEmaLen", 10))
    use_event_filter = bool(params.get("useEventFilter", False))
    event_windows_raw = str(params.get("eventWindows", "")) if use_event_filter else ""
    use_time_stop = bool(params.get("useTimeStop", False))
    max_hold_param = int(params.get("maxHoldBars", 0))
    use_breakeven = bool(params.get("useBreakevenStop", False))
    breakeven_mult = float(params.get("breakevenMult", 1.0))
    use_atr_trail = bool(params.get("useAtrTrail", False))
    atr_trail_len = int(params.get("atrTrailLen", max(1, wt_len)))
    atr_trail_mult = float(params.get("atrTrailMult", 2.0))
    use_pivot_stop = bool(params.get("usePivotStop", False))
    pivot_len = int(params.get("pivotLen", 5))

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

    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    ap = _ema(hlc3, wt_len)
    atr = _atr(df, wt_len).bfill()
    atr = atr.replace(0, np.nan)
    wavetrend = (df["close"] - ap) / atr
    osc = _ema(wavetrend.fillna(0), max(2, wt_len // 2))

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
        osc_val = osc.loc[ts]
        atr_val = atr.loc[ts]
        if np.isnan(osc_val) or np.isnan(atr_val):
            continue

        long_sig = osc_val > thr
        short_sig = osc_val < -thr

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
            stop_price = ctx.entry_price - atr_mult * atr_val
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
                elif use_pivot_stop and not np.isnan(pivot_low_val) and row["low"] <= pivot_low_val:
                    exit_reason = "pivot_stop"
                    exit_price = pivot_low_val
                elif use_atr_trail and not np.isnan(atr_trail_val) and ctx.highest_price > 0:
                    trail_stop = ctx.highest_price - atr_trail_mult * atr_trail_val
                    if row["low"] <= trail_stop:
                        exit_reason = "atr_trail"
                        exit_price = trail_stop
                if exit_reason is None and row["low"] <= stop_price:
                    exit_reason = "atr_stop"
                    exit_price = stop_price
                elif exit_reason is None and max_hold_bars and ctx.bars_held >= max_hold_bars:
                    exit_reason = "time_stop"
                elif (
                    exit_reason is None
                    and allow_signal_exit
                    and short_sig
                    and not long_sig
                    and htf_pass
                ):
                    exit_reason = "reverse_short"
        elif ctx.position < 0:
            stop_price = ctx.entry_price + atr_mult * atr_val
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
                elif use_pivot_stop and not np.isnan(pivot_high_val) and row["high"] >= pivot_high_val:
                    exit_reason = "pivot_stop"
                    exit_price = pivot_high_val
                elif use_atr_trail and not np.isnan(atr_trail_val) and ctx.lowest_price > 0:
                    trail_stop = ctx.lowest_price + atr_trail_mult * atr_trail_val
                    if row["high"] >= trail_stop:
                        exit_reason = "atr_trail"
                        exit_price = trail_stop
                if exit_reason is None and row["high"] >= stop_price:
                    exit_reason = "atr_stop"
                    exit_price = stop_price
                elif exit_reason is None and max_hold_bars and ctx.bars_held >= max_hold_bars:
                    exit_reason = "time_stop"
                elif (
                    exit_reason is None
                    and allow_signal_exit
                    and long_sig
                    and not short_sig
                    and htf_pass
                ):
                    exit_reason = "reverse_long"

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
