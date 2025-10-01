"""청산 로직 계산 모듈."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import indicators
from .schema import ExitParams


@dataclass
class ExitLevels:
    stop: float
    tp1: Optional[float]
    tp2: Optional[float]
    percent_stop: Optional[float]
    percent_take: Optional[float]
    percent_trail: Optional[float]
    atr_trail: Optional[float]
    atr_value: float
    roi_hit: bool
    max_bars_hit: bool


class ExitCalculator:
    def __init__(self, params: ExitParams):
        self.params = params

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        result["atr"] = indicators.atr(df["high"], df["low"], df["close"], self.params.atrLen)
        if self.params.useSwingSL:
            result["swing_low"] = df["low"].rolling(self.params.swN).min()
            result["swing_high"] = df["high"].rolling(self.params.swN).max()
        else:
            result["swing_low"] = pd.NA
            result["swing_high"] = pd.NA
        return result

    def calc_levels(
        self,
        row: pd.Series,
        entry_price: float,
        direction: str,
        bars_in_trade: int,
        minutes_in_trade: float,
    ) -> ExitLevels:
        atr_raw = row.get("atr", 0.0)
        atr_val = float(atr_raw) if pd.notna(atr_raw) else 0.0
        swing_low = row.get("swing_low")
        swing_high = row.get("swing_high")
        price = float(row.get("close", entry_price))

        if direction == "long":
            base_stop = entry_price - self.params.initKEff * atr_val
            if self.params.useSwingSL and pd.notna(swing_low):
                base_stop = min(base_stop, float(swing_low))
            r_multiple = entry_price - base_stop
            tp1 = entry_price + self.params.tp1RR * r_multiple if r_multiple > 0 else None
            tp2 = entry_price + self.params.tp2RR * r_multiple if r_multiple > 0 else None
            if self.params.usePercentStops:
                stop_percent = entry_price * (1 - self.params.stopPctEff / 100)
                take_percent = entry_price * (1 + self.params.takePctEff / 100)
                activate = entry_price * (1 + self.params.trailStartEff / 100)
                percent_trail = price * (1 - self.params.trailGapEff / 100) if price >= activate else None
            else:
                stop_percent = None
                take_percent = None
                percent_trail = None
            atr_trail = price - self.params.trailKEff * atr_val if self.params.trailKEff > 0 else None
            roi_hit = self._roi_check(minutes_in_trade, entry_price, price, True)
        else:
            base_stop = entry_price + self.params.initKEff * atr_val
            if self.params.useSwingSL and pd.notna(swing_high):
                base_stop = max(base_stop, float(swing_high))
            r_multiple = base_stop - entry_price
            tp1 = entry_price - self.params.tp1RR * r_multiple if r_multiple > 0 else None
            tp2 = entry_price - self.params.tp2RR * r_multiple if r_multiple > 0 else None
            if self.params.usePercentStops:
                stop_percent = entry_price * (1 + self.params.stopPctEff / 100)
                take_percent = entry_price * (1 - self.params.takePctEff / 100)
                activate = entry_price * (1 - self.params.trailStartEff / 100)
                percent_trail = price * (1 + self.params.trailGapEff / 100) if price <= activate else None
            else:
                stop_percent = None
                take_percent = None
                percent_trail = None
            atr_trail = price + self.params.trailKEff * atr_val if self.params.trailKEff > 0 else None
            roi_hit = self._roi_check(minutes_in_trade, entry_price, price, False)

        max_bars_hit = self.params.maxBarsHoldEff > 0 and bars_in_trade >= self.params.maxBarsHoldEff
        return ExitLevels(
            stop=base_stop,
            tp1=tp1,
            tp2=tp2,
            percent_stop=stop_percent,
            percent_take=take_percent,
            percent_trail=percent_trail,
            atr_trail=atr_trail,
            atr_value=atr_val,
            roi_hit=roi_hit,
            max_bars_hit=max_bars_hit,
        )

    def _roi_check(self, minutes: float, entry: float, current: float, is_long: bool) -> bool:
        def hit(min_req: int, pct: float) -> bool:
            if min_req <= 0:
                return False
            if minutes < min_req:
                return False
            target = entry * (1 + pct / 100) if is_long else entry * (1 - pct / 100)
            return current >= target if is_long else current <= target

        return (
            hit(self.params.roi1MinEff, self.params.roi1PctEff)
            or hit(self.params.roi2MinEff, self.params.roi2PctEff)
            or hit(self.params.roi3MinEff, self.params.roi3PctEff)
        )


__all__ = ["ExitCalculator", "ExitLevels"]
