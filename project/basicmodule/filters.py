"""필터 및 컨텍스트 로직."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import indicators
from .schema import FilterParams


@dataclass
class FilterContext:
    ma100: pd.Series
    dir_long: pd.Series
    dir_short: pd.Series
    micro_long: pd.Series
    micro_short: pd.Series
    context_long: pd.Series
    context_short: pd.Series
    extras: Dict[str, pd.Series]


def compute_filters(
    df: pd.DataFrame,
    params: FilterParams,
    htf_long: Optional[pd.Series] = None,
    htf_short: Optional[pd.Series] = None,
    regime_long: Optional[pd.Series] = None,
    regime_short: Optional[pd.Series] = None,
) -> FilterContext:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ma100 = indicators.ma(close, 100, params.maType)

    def optional(cond: pd.Series, active: bool) -> pd.Series:
        return cond if active else pd.Series(True, index=cond.index)

    dir_long = optional(close > ma100, params.useMA100)
    dir_short = optional(close < ma100, params.useMA100)

    ema_fast = indicators.ema(close, params.emaFastLenBase)
    ema_slow = indicators.ema(close, params.emaSlowLenBase)
    micro_long = optional(ema_fast > ema_slow, params.useMicroTrend)
    micro_short = optional(ema_fast < ema_slow, params.useMicroTrend)

    trend_bias = indicators.ema(close, params.trendLenBase)
    conf_bias = indicators.ema(close, params.confLenBase)
    trend_long_ok = optional(close > trend_bias, params.useTrendBias)
    trend_short_ok = optional(close < trend_bias, params.useTrendBias)
    conf_long_ok = optional(close > conf_bias, params.useConfBias)
    conf_short_ok = optional(close < conf_bias, params.useConfBias)

    slope_prev = trend_bias.shift(params.emaFastLenBase)
    slope_pct = (trend_bias - slope_prev) / trend_bias.replace(0, np.nan) * 100
    slope_long_ok = slope_pct >= 0
    slope_short_ok = slope_pct <= 0

    range_high = indicators.highest(high, params.emaFastLenBase)
    range_low = indicators.lowest(low, params.emaFastLenBase)
    atr_val = indicators.atr(high, low, close, params.emaFastLenBase)
    is_range = (range_high - range_low) < atr_val * params.bbMult
    range_ok = ~is_range

    atr_distance = indicators.atr(high, low, close, params.emaFastLenBase)
    vw_distance = (close - trend_bias).abs() / atr_distance.replace(0, np.nan)
    distance_long_ok = vw_distance <= params.bbMult
    distance_short_ok = distance_long_ok

    bb_upper, bb_lower, kc_upper, kc_lower, momentum = indicators.squeeze_momentum(
        close=close,
        high=high,
        low=low,
        length=params.bbLen,
        mult=params.bbMult,
        kc_len=params.kcLen,
    )
    squeeze_on = (bb_upper - bb_lower) < (kc_upper - kc_lower)
    mom_long_ok = optional(
        (momentum > 0) & ((~squeeze_on) | (momentum.diff() > 0)), params.useMomConfirm
    )
    mom_short_ok = optional(
        (momentum < 0) & ((~squeeze_on) | (momentum.diff() < 0)), params.useMomConfirm
    )

    piv_high = indicators.pivot_high(high, params.pL, params.pR)
    piv_low = indicators.pivot_low(low, params.pL, params.pR)
    last_high = piv_high.ffill()
    last_low = piv_low.ffill()
    choch_long_ok = optional((close > last_high) & (low > last_low), params.useCHoCH)
    choch_short_ok = optional((close < last_low) & (high < last_high), params.useCHoCH)

    vol_sma = volume.rolling(params.volumeLookback).mean()
    volume_ok = optional(volume >= vol_sma * params.volumeMultiplier, params.useVolumeFilter)

    body = (close - df["open"]).abs()
    candle_range = (high - low).replace(0, np.nan)
    candle_ratio = body / candle_range * 100
    candle_ok = optional(candle_ratio >= params.candleBodyRatio, params.useCandleFilter)

    ewo_raw = indicators.ewo(close, params.ewoFast, params.ewoSlow)
    ewo_sig = indicators.ewo_signal(ewo_raw, params.ewoSignal)
    ewo_long_ok = optional((ewo_raw < 0) & (ewo_raw.diff() < 0) & (ewo_raw < ewo_sig), params.useEWO)
    ewo_short_ok = optional((ewo_raw > 0) & (ewo_raw.diff() > 0) & (ewo_raw > ewo_sig), params.useEWO)

    chop = indicators.choppiness(high, low, close, params.chopLen)
    chop_ok = optional(chop <= params.chopMax, params.useChop)

    atr_perc = indicators.atr_percent(high, low, close, params.kcLen)
    vol_ok = optional(
        (atr_perc >= params.minAtrPerc) & (atr_perc <= params.maxAtrPerc), params.useVolatility
    )

    vol_boost = volume.rolling(params.volLen).mean()
    vol_pass = optional(volume >= vol_boost, params.useVolumeBoost)

    highest_struct = indicators.highest(high, params.structureLen)
    lowest_struct = indicators.lowest(low, params.structureLen)
    structure_long_ok = optional(close > highest_struct, params.useStructure)
    structure_short_ok = optional(close < lowest_struct, params.useStructure)

    context_long = (
        dir_long
        & micro_long
        & trend_long_ok
        & conf_long_ok
        & slope_long_ok
        & range_ok
        & distance_long_ok
        & mom_long_ok
        & choch_long_ok
        & volume_ok
        & candle_ok
        & ewo_long_ok
        & chop_ok
        & vol_ok
        & vol_pass
        & structure_long_ok
    )

    context_short = (
        dir_short
        & micro_short
        & trend_short_ok
        & conf_short_ok
        & slope_short_ok
        & range_ok
        & distance_short_ok
        & mom_short_ok
        & choch_short_ok
        & volume_ok
        & candle_ok
        & ewo_short_ok
        & chop_ok
        & vol_ok
        & vol_pass
        & structure_short_ok
    )

    if htf_long is not None:
        context_long &= htf_long
    if htf_short is not None:
        context_short &= htf_short
    if regime_long is not None:
        context_long &= regime_long
    if regime_short is not None:
        context_short &= regime_short

    extras = {
        "trend_bias": trend_bias,
        "conf_bias": conf_bias,
        "ewo_raw": ewo_raw,
        "choppiness": chop,
        "atr_percent": atr_perc,
        "momentum": momentum,
    }

    return FilterContext(
        ma100=ma100,
        dir_long=dir_long,
        dir_short=dir_short,
        micro_long=micro_long,
        micro_short=micro_short,
        context_long=context_long,
        context_short=context_short,
        extras=extras,
    )


__all__ = ["FilterContext", "compute_filters"]
