"""지표 계산 모듈.

Pine Script에서 사용한 주요 지표를 pandas Series 기반으로 재구현한다.
룩어헤드를 방지하기 위해 모든 계산은 확정 봉 데이터만 사용하고,
HTF 리샘플 후에는 항상 `shift(1)` 처리된 값을 반환한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """전형적인 Heikin-Ashi 변환을 수행한다."""

    ha_df = pd.DataFrame(index=df.index, dtype=float)
    ha_df["ha_close"] = df[["open", "high", "low", "close"]].mean(axis=1)
    ha_open = [df["open"].iloc[0]]
    for close in ha_df["ha_close"].iloc[1:]:
        ha_open.append((ha_open[-1] + close) / 2)
    ha_df["ha_open"] = ha_open
    ha_df["ha_high"] = pd.concat([ha_df["ha_open"], ha_df["ha_close"], df["high"]], axis=1).max(axis=1)
    ha_df["ha_low"] = pd.concat([ha_df["ha_open"], ha_df["ha_close"], df["low"]], axis=1).min(axis=1)
    return ha_df


@dataclass
class UTBotResult:
    trail: pd.Series
    buy: pd.Series
    sell: pd.Series


def utbot(
    price: pd.Series,
    atr: pd.Series,
    key: float,
) -> UTBotResult:
    """UTBot 트레일 및 시그널 계산."""

    trailing = price.copy().astype(float)
    trailing.iloc[0] = price.iloc[0]
    buy = pd.Series(False, index=price.index)
    sell = pd.Series(False, index=price.index)

    for i in range(1, len(price)):
        prev_trail = trailing.iat[i - 1]
        src = price.iat[i]
        atr_val = atr.iat[i]
        loss = key * atr_val

        if src > prev_trail and price.iat[i - 1] > prev_trail:
            new_trail = max(prev_trail, src - loss)
        elif src < prev_trail and price.iat[i - 1] < prev_trail:
            new_trail = min(prev_trail, src + loss)
        else:
            new_trail = src - loss if src > prev_trail else src + loss

        trailing.iat[i] = new_trail
        prev_src = price.iat[i - 1]
        if prev_src <= prev_trail and src > new_trail:
            buy.iat[i] = True
        if prev_src >= prev_trail and src < new_trail:
            sell.iat[i] = True
    return UTBotResult(trail=trailing, buy=buy, sell=sell)


def stoch_rsi(
    close: pd.Series,
    rsi_len: int,
    stoch_len: int,
    k_len: int,
    d_len: int,
) -> Tuple[pd.Series, pd.Series]:
    rsi = close.diff().fillna(0)
    gain = rsi.clip(lower=0)
    loss = -rsi.clip(upper=0)
    avg_gain = gain.rolling(rsi_len).mean()
    avg_loss = loss.rolling(rsi_len).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - 100 / (1 + rs)
    rsi_val = rsi_val.fillna(0)

    lowest = rsi_val.rolling(stoch_len).min()
    highest = rsi_val.rolling(stoch_len).max()
    stoch = (rsi_val - lowest) / (highest - lowest).replace(0, np.nan)
    stoch = stoch.fillna(0)
    k = stoch.rolling(k_len).mean()
    d = k.rolling(d_len).mean()
    return k.fillna(0), d.fillna(0)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def ma(series: pd.Series, length: int, mode: Literal["EMA", "SMA", "WMA"] = "EMA") -> pd.Series:
    if mode == "EMA":
        return ema(series, length)
    if mode == "SMA":
        return series.rolling(length).mean()
    if mode == "WMA":
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    raise ValueError(f"지원하지 않는 MA 모드: {mode}")


def ewo(series: pd.Series, fast: int, slow: int) -> pd.Series:
    return ema(series, fast) - ema(series, slow)


def ewo_signal(series: pd.Series, signal: int) -> pd.Series:
    return ema(series, signal)


def choppiness(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1))
    tr_sum = tr.rolling(length).sum()
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    denom = (highest - lowest).replace(0, np.nan)
    chop = 100 * np.log10(tr_sum / denom) / np.log10(length)
    return chop.fillna(100)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1))
    tr = tr.fillna(high - low)
    return tr.rolling(length).mean()


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    return atr(high, low, close, length) / close * 100


def squeeze_momentum(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    length: int,
    mult: float,
    kc_len: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    basis = close.rolling(length).mean()
    dev = close.rolling(length).std(ddof=0)
    upper = basis + mult * dev
    lower = basis - mult * dev
    tr = atr(high, low, close, kc_len)
    kc_upper = basis + mult * tr
    kc_lower = basis - mult * tr
    momentum = close - basis

    def linreg_slope(window: pd.Series) -> float:
        if window.isna().any():
            return 0.0
        x = np.arange(len(window))
        y = window.values
        slope, _ = np.polyfit(x, y, 1)
        return slope

    momentum = momentum.rolling(14).apply(linreg_slope, raw=False)
    return upper, lower, kc_upper, kc_lower, momentum


def highest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).max()


def lowest(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).min()


def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    return high.rolling(window=left + right + 1, center=True).apply(
        lambda x: x[left] if x[left] == x.max() else np.nan, raw=True
    )


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    return low.rolling(window=left + right + 1, center=True).apply(
        lambda x: x[left] if x[left] == x.min() else np.nan, raw=True
    )


__all__ = [
    "UTBotResult",
    "atr",
    "atr_percent",
    "choppiness",
    "ema",
    "ewo",
    "ewo_signal",
    "heikin_ashi",
    "highest",
    "lowest",
    "ma",
    "pivot_high",
    "pivot_low",
    "squeeze_momentum",
    "stoch_rsi",
    "utbot",
]
