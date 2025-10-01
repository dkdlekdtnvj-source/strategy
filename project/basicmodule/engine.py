"""전략 상태 머신 및 백테스트 엔진."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import filters, indicators, metrics
from .exits import ExitCalculator
from .schema import StrategyParams


@dataclass
class Trade:
    timestamp: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fee: float
    reason: str


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    score: float
    metrics: metrics.Metrics
    logs: pd.DataFrame


class StrategyEngine:
    def __init__(
        self,
        params: StrategyParams,
        fee: float,
        slippage: float,
        initial_capital: float,
        objective_weights: Dict[str, float],
        timeframe: str,
        tf_settings: Dict[str, str],
    ) -> None:
        self.params = params
        self.fee = fee
        self.slippage = slippage
        self.initial_capital = initial_capital
        self.objective_weights = objective_weights
        self.timeframe = timeframe
        self.tf_settings = tf_settings
        self.exit_calc = ExitCalculator(params.exits)

    def _align_series(self, base_index: pd.Index, series: pd.Series) -> pd.Series:
        aligned = series.shift(1).reindex(base_index, method="ffill")
        return aligned.ffill()

    def prepare(self, df: pd.DataFrame, htf_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        ut_src = df["ha_close"] if self.params.ut.utHA and "ha_close" in df else df["close"]
        ut_atr = indicators.atr(df["high"], df["low"], ut_src, self.params.ut.utAtrEff)
        ut_result = indicators.utbot(ut_src, ut_atr, self.params.ut.utKeyEff)

        k, d = indicators.stoch_rsi(
            df["close"],
            self.params.stoch.rsiLen,
            self.params.stoch.stLen,
            self.params.stoch.kLen,
            self.params.stoch.dLen,
        )
        stoch_tf = self.tf_settings.get("stoch", self.timeframe)
        if stoch_tf != self.timeframe:
            stoch_df = pd.DataFrame({"k": k, "d": d})
            resampled = stoch_df.resample(stoch_tf).last().shift(1)
            k = self._align_series(df.index, resampled["k"])
            d = self._align_series(df.index, resampled["d"])
        else:
            k = k.shift(1)
            d = d.shift(1)

        stoch_long = (
            (k < self.params.stoch.osEff) & (k > d)
            if self.params.stoch.stMode == "Bounce"
            else ((k > d) & (k < 50))
        ).fillna(False)
        stoch_short = (
            (k > self.params.stoch.obEff) & (k < d)
            if self.params.stoch.stMode == "Bounce"
            else ((k < d) & (k > 50))
        ).fillna(False)

        htf_long = None
        htf_short = None
        trend_tf1 = htf_data.get("htf1")
        trend_tf2 = htf_data.get("htf2")
        if trend_tf1 is not None and trend_tf2 is not None:
            htf1 = trend_tf1
            htf2 = trend_tf2
            ha1 = indicators.heikin_ashi(htf1)["ha_close"]
            ha2 = indicators.heikin_ashi(htf2)["ha_close"]
            ema1 = htf1["close"].rolling(20).mean()
            ema2 = htf2["close"].rolling(20).mean()
            cond_long = (ha1 > ema1) & (ha2 > ema2)
            cond_short = (ha1 < ema1) & (ha2 < ema2)
            htf_long = self._align_series(df.index, cond_long.astype(float)).astype(bool)
            htf_short = self._align_series(df.index, cond_short.astype(float)).astype(bool)

        regime_long = None
        regime_short = None
        if "regime" in htf_data:
            reg_df = htf_data["regime"]
            ema200 = reg_df["close"].ewm(span=200, adjust=False).mean()
            adx = reg_df["close"].diff().abs().rolling(14).mean()  # 근사치
            cond_long = (reg_df["close"] > ema200) & (adx >= self.params.stoch.obEff)
            cond_short = (reg_df["close"] < ema200) & (adx >= self.params.stoch.obEff)
            regime_long = self._align_series(df.index, cond_long.astype(float)).astype(bool)
            regime_short = self._align_series(df.index, cond_short.astype(float)).astype(bool)

        filter_context = filters.compute_filters(
            df,
            self.params.filters,
            htf_long=htf_long,
            htf_short=htf_short,
            regime_long=regime_long,
            regime_short=regime_short,
        )

        return {
            "ut": ut_result,
            "stoch_long": stoch_long,
            "stoch_short": stoch_short,
            "filters": filter_context,
        }

    def run(self, df: pd.DataFrame, indicators_map: Dict[str, any]) -> BacktestResult:
        ut = indicators_map["ut"]
        stoch_long = indicators_map["stoch_long"]
        stoch_short = indicators_map["stoch_short"]
        filter_ctx = indicators_map["filters"]

        signals_long = ut.buy & stoch_long & filter_ctx.context_long
        signals_short = ut.sell & stoch_short & filter_ctx.context_short

        if self.params.ut.noRP:
            signals_long = signals_long.shift(1).fillna(False)
            signals_short = signals_short.shift(1).fillna(False)

        position = 0
        entry_price = 0.0
        trades: List[Trade] = []
        equity = self.initial_capital
        equity_curve = []
        qty = 1.0
        armed_long = False
        armed_short = False
        trade_bars = 0
        entry_time: Optional[pd.Timestamp] = None
        exit_cache = self.exit_calc.prepare(df)
        log_rows = []

        for ts, row in df.iterrows():
            price = row["close"]
            bar_fee = 0.0
            reason = ""
            action = "hold"

            long_ready = signals_long.loc[ts]
            short_ready = signals_short.loc[ts]

            if position == 0:
                if self.params.ut.ibs:
                    if long_ready and not armed_long:
                        position = 1
                        entry_price = price + self.slippage
                        entry_time = ts
                        trade_bars = 0
                        armed_long = True
                        action = "enter_long"
                    elif short_ready and not armed_short:
                        position = -1
                        entry_price = price - self.slippage
                        entry_time = ts
                        trade_bars = 0
                        armed_short = True
                        action = "enter_short"
                else:
                    if long_ready:
                        position = 1
                        entry_price = price + self.slippage
                        entry_time = ts
                        trade_bars = 0
                        action = "enter_long"
                    elif short_ready:
                        position = -1
                        entry_price = price - self.slippage
                        entry_time = ts
                        trade_bars = 0
                        action = "enter_short"

                if position != 0:
                    bar_fee = abs(entry_price) * self.fee * qty
            else:
                trade_bars += 1

            if position != 0:
                minutes = trade_bars * self._bar_minutes()
                exit_row = row.combine_first(exit_cache.loc[ts])
                direction = "long" if position > 0 else "short"
                exit_levels = self.exit_calc.calc_levels(
                    exit_row,
                    entry_price,
                    direction,
                    trade_bars,
                    minutes,
                )

                exit_price = None
                exit_reason = ""

                if self.params.exits.utFlipExit:
                    if position > 0 and ut.sell.loc[ts]:
                        exit_price = price - self.slippage
                        exit_reason = "ut_flip"
                    elif position < 0 and ut.buy.loc[ts]:
                        exit_price = price + self.slippage
                        exit_reason = "ut_flip"

                if exit_levels.roi_hit and not exit_reason:
                    exit_price = price
                    exit_reason = "roi"
                if exit_levels.max_bars_hit and not exit_reason:
                    exit_price = price
                    exit_reason = "time"

                if exit_price is None:
                    if direction == "long":
                        stop_level = exit_levels.stop
                        if exit_levels.stop_percent:
                            stop_level = max(stop_level, exit_levels.stop_percent)
                        if price <= stop_level:
                            exit_price = stop_level - self.slippage
                            exit_reason = "stop"
                        elif exit_levels.take_percent and price >= exit_levels.take_percent:
                            exit_price = exit_levels.take_percent - self.slippage
                            exit_reason = "take_pct"
                    else:
                        stop_level = exit_levels.stop
                        if exit_levels.stop_percent:
                            stop_level = min(stop_level, exit_levels.stop_percent)
                        if price >= stop_level:
                            exit_price = stop_level + self.slippage
                            exit_reason = "stop"
                        elif exit_levels.take_percent and price <= exit_levels.take_percent:
                            exit_price = exit_levels.take_percent + self.slippage
                            exit_reason = "take_pct"

                if exit_price is not None:
                    pnl = (exit_price - entry_price) * position * qty
                    bar_fee += abs(exit_price) * self.fee * qty
                    net = pnl - bar_fee
                    equity += net
                    trades.append(
                        Trade(
                            timestamp=ts,
                            direction=direction,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            qty=qty,
                            pnl=net,
                            fee=bar_fee,
                            reason=exit_reason,
                        )
                    )
                    position = 0
                    entry_price = 0.0
                    trade_bars = 0
                    entry_time = None
                    action = f"exit_{exit_reason}"
                    reason = exit_reason

            if self.params.ut.ibs:
                armed_long = long_ready
                armed_short = short_ready

            equity_curve.append(equity)
            log_rows.append(
                {
                    "timestamp": ts,
                    "price": price,
                    "equity": equity,
                    "position": position,
                    "action": action,
                    "fee": bar_fee,
                    "reason": reason,
                }
            )

        equity_series = pd.Series(equity_curve, index=df.index)
        trade_pnl = pd.Series([t.pnl for t in trades]) if trades else pd.Series(dtype=float)
        exposure = trade_pnl.count() / len(df) if len(df) else 0
        metric = metrics.compute_metrics(trade_pnl, equity_series, exposure)
        score = metrics.objective_score(metric, self.objective_weights)
        logs_df = pd.DataFrame(log_rows)
        return BacktestResult(trades=trades, equity_curve=equity_series, score=score, metrics=metric, logs=logs_df)

    def _bar_minutes(self) -> float:
        mapping = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60}
        return mapping.get(self.timeframe, 1)


__all__ = ["BacktestResult", "StrategyEngine", "Trade"]
