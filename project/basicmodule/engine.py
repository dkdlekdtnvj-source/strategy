"""전략 상태 머신 및 백테스트 엔진."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from . import filters, indicators, metrics
from .exits import ExitCalculator
from .schema import StrategyParams
from .utils import KST


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

    def run(self, df: pd.DataFrame, indicators_map: Dict[str, Any]) -> BacktestResult:
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
        equity_curve: List[float] = []
        position_qty = 0.0
        initial_qty = 0.0
        total_entry_fee = 0.0
        remaining_entry_fee = 0.0
        armed_long = False
        armed_short = False
        trade_bars = 0
        entry_time: Optional[pd.Timestamp] = None
        tp1_hit = False
        breakeven_price: Optional[float] = None
        cooldown = 0
        daily_trades: Dict[str, int] = {}
        exit_cache = self.exit_calc.prepare(df)
        log_rows = []

        for ts, row in df.iterrows():
            price = float(row["close"])
            bar_fee = 0.0
            events: List[str] = []
            action = "hold"

            long_ready = bool(signals_long.loc[ts])
            short_ready = bool(signals_short.loc[ts])

            allow_entry = cooldown == 0
            if cooldown > 0:
                cooldown -= 1

            day_key = self._day_key(ts)
            if self.params.risk.maxTradesPerDay > 0:
                if daily_trades.get(day_key, 0) >= self.params.risk.maxTradesPerDay:
                    allow_entry = False

            if not allow_entry:
                long_ready = False
                short_ready = False

            if position == 0:
                if self.params.ut.ibs:
                    if long_ready and not armed_long:
                        position = 1
                        entry_price = price + self.slippage
                        position_qty = 1.0
                        initial_qty = position_qty
                        total_entry_fee = abs(entry_price) * self.fee * position_qty
                        remaining_entry_fee = total_entry_fee
                        entry_time = ts
                        trade_bars = 0
                        tp1_hit = False
                        breakeven_price = None
                        armed_long = True
                        daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
                        action = "enter_long"
                    elif short_ready and not armed_short:
                        position = -1
                        entry_price = price - self.slippage
                        position_qty = 1.0
                        initial_qty = position_qty
                        total_entry_fee = abs(entry_price) * self.fee * position_qty
                        remaining_entry_fee = total_entry_fee
                        entry_time = ts
                        trade_bars = 0
                        tp1_hit = False
                        breakeven_price = None
                        armed_short = True
                        daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
                        action = "enter_short"
                else:
                    if long_ready:
                        position = 1
                        entry_price = price + self.slippage
                        position_qty = 1.0
                        initial_qty = position_qty
                        total_entry_fee = abs(entry_price) * self.fee * position_qty
                        remaining_entry_fee = total_entry_fee
                        entry_time = ts
                        trade_bars = 0
                        tp1_hit = False
                        breakeven_price = None
                        daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
                        action = "enter_long"
                    elif short_ready:
                        position = -1
                        entry_price = price - self.slippage
                        position_qty = 1.0
                        initial_qty = position_qty
                        total_entry_fee = abs(entry_price) * self.fee * position_qty
                        remaining_entry_fee = total_entry_fee
                        entry_time = ts
                        trade_bars = 0
                        tp1_hit = False
                        breakeven_price = None
                        daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
                        action = "enter_short"
            else:
                trade_bars += 1

            def execute_exit(exit_price: float, close_qty: float, reason_label: str) -> None:
                nonlocal position, position_qty, remaining_entry_fee, equity, entry_price, trade_bars
                nonlocal entry_time, tp1_hit, breakeven_price, total_entry_fee, initial_qty, cooldown, bar_fee
                if close_qty <= 0:
                    return
                entry_fee_share = 0.0
                if initial_qty > 0 and total_entry_fee > 0:
                    entry_fee_share = total_entry_fee * (close_qty / initial_qty)
                    entry_fee_share = min(entry_fee_share, remaining_entry_fee)
                    remaining_entry_fee -= entry_fee_share
                exit_fee = abs(exit_price) * self.fee * close_qty
                gross = (exit_price - entry_price) * position * close_qty
                net = gross - entry_fee_share - exit_fee
                equity += net
                bar_fee += entry_fee_share + exit_fee
                trades.append(
                    Trade(
                        timestamp=ts,
                        direction="long" if position > 0 else "short",
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=close_qty,
                        pnl=net,
                        fee=entry_fee_share + exit_fee,
                        reason=reason_label,
                    )
                )
                if net < 0:
                    cooldown = max(cooldown, self.params.risk.lossCooldownBars)
                position_qty -= close_qty
                if position_qty <= 1e-9:
                    position = 0
                    position_qty = 0.0
                    entry_price = 0.0
                    trade_bars = 0
                    entry_time = None
                    tp1_hit = False
                    breakeven_price = None
                    total_entry_fee = 0.0
                    remaining_entry_fee = 0.0
                    initial_qty = 0.0

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

                if self.params.exits.utFlipExit:
                    if position > 0 and bool(ut.sell.loc[ts]):
                        close_qty = position_qty
                        if close_qty > 0:
                            execute_exit(price - self.slippage, close_qty, "ut_flip")
                            events.append("ut_flip")
                            action = "exit_ut_flip"
                    elif position < 0 and bool(ut.buy.loc[ts]):
                        close_qty = position_qty
                        if close_qty > 0:
                            execute_exit(price + self.slippage, close_qty, "ut_flip")
                            events.append("ut_flip")
                            action = "exit_ut_flip"

                if position != 0 and exit_levels.roi_hit:
                    close_qty = position_qty
                    if close_qty > 0:
                        execute_exit(price, close_qty, "roi")
                        events.append("roi")
                        action = "exit_roi"

                if position != 0 and exit_levels.max_bars_hit:
                    close_qty = position_qty
                    if close_qty > 0:
                        execute_exit(price, close_qty, "time")
                        events.append("time")
                        action = "exit_time"

                if position != 0 and exit_levels.tp1 is not None and not tp1_hit:
                    triggered = price >= exit_levels.tp1 if position > 0 else price <= exit_levels.tp1
                    if triggered:
                        tp1_hit = True
                        pct = max(0.0, min(100.0, self.params.exits.tp1PctEff)) / 100.0
                        if pct > 0 and position_qty > 0:
                            close_qty = min(position_qty, position_qty * pct)
                            fill = exit_levels.tp1 - self.slippage if position > 0 else exit_levels.tp1 + self.slippage
                            execute_exit(fill, close_qty, "tp1_partial")
                            events.append("tp1_partial")
                            action = "partial_tp1"
                        else:
                            events.append("tp1_trigger")
                        if position > 0:
                            candidate_be = entry_price + self.params.exits.beOffsetEff * exit_levels.atr_value
                            breakeven_price = max(breakeven_price or float("-inf"), candidate_be)
                        else:
                            candidate_be = entry_price - self.params.exits.beOffsetEff * exit_levels.atr_value
                            breakeven_price = min(breakeven_price or float("inf"), candidate_be)

                if position != 0 and tp1_hit:
                    if position > 0:
                        candidate_be = entry_price + self.params.exits.beOffsetEff * exit_levels.atr_value
                        breakeven_price = max(breakeven_price or float("-inf"), candidate_be)
                    else:
                        candidate_be = entry_price - self.params.exits.beOffsetEff * exit_levels.atr_value
                        breakeven_price = min(breakeven_price or float("inf"), candidate_be)

                if position != 0 and exit_levels.percent_take is not None:
                    level = exit_levels.percent_take
                    if position > 0 and price >= level:
                        close_qty = position_qty
                        if close_qty > 0:
                            execute_exit(level - self.slippage, close_qty, "take_pct")
                            events.append("take_pct")
                            action = "exit_take_pct"
                    elif position < 0 and price <= level:
                        close_qty = position_qty
                        if close_qty > 0:
                            execute_exit(level + self.slippage, close_qty, "take_pct")
                            events.append("take_pct")
                            action = "exit_take_pct"

                if position != 0 and exit_levels.tp2 is not None:
                    if position > 0 and price >= exit_levels.tp2:
                        close_qty = position_qty
                        if close_qty > 0:
                            execute_exit(exit_levels.tp2 - self.slippage, close_qty, "tp2")
                            events.append("tp2")
                            action = "exit_tp2"
                    elif position < 0 and price <= exit_levels.tp2:
                        close_qty = position_qty
                        if close_qty > 0:
                            execute_exit(exit_levels.tp2 + self.slippage, close_qty, "tp2")
                            events.append("tp2")
                            action = "exit_tp2"

                if position != 0:
                    if position > 0:
                        candidates = [(exit_levels.stop, "stop")]
                        if exit_levels.percent_stop is not None:
                            candidates.append((exit_levels.percent_stop, "percent_stop"))
                        if breakeven_price is not None:
                            candidates.append((breakeven_price, "breakeven"))
                        if tp1_hit and exit_levels.atr_trail is not None:
                            candidates.append((exit_levels.atr_trail, "trail_atr"))
                        if exit_levels.percent_trail is not None:
                            candidates.append((exit_levels.percent_trail, "trail_pct"))
                        effective_stop, stop_reason = max(candidates, key=lambda item: item[0])
                        if price <= effective_stop:
                            close_qty = position_qty
                            if close_qty > 0:
                                execute_exit(effective_stop - self.slippage, close_qty, stop_reason)
                                events.append(stop_reason)
                                action = f"exit_{stop_reason}"
                    else:
                        candidates = [(exit_levels.stop, "stop")]
                        if exit_levels.percent_stop is not None:
                            candidates.append((exit_levels.percent_stop, "percent_stop"))
                        if breakeven_price is not None:
                            candidates.append((breakeven_price, "breakeven"))
                        if tp1_hit and exit_levels.atr_trail is not None:
                            candidates.append((exit_levels.atr_trail, "trail_atr"))
                        if exit_levels.percent_trail is not None:
                            candidates.append((exit_levels.percent_trail, "trail_pct"))
                        effective_stop, stop_reason = min(candidates, key=lambda item: item[0])
                        if price >= effective_stop:
                            close_qty = position_qty
                            if close_qty > 0:
                                execute_exit(effective_stop + self.slippage, close_qty, stop_reason)
                                events.append(stop_reason)
                                action = f"exit_{stop_reason}"

            if self.params.ut.ibs:
                armed_long = long_ready
                armed_short = short_ready

            reason = ";".join(events)
            if not reason and action.startswith("enter_"):
                reason = action

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

    def _day_key(self, ts: pd.Timestamp) -> str:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(KST).strftime("%Y-%m-%d")


__all__ = ["BacktestResult", "StrategyEngine", "Trade"]
