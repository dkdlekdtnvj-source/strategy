"""Python backtest implementation mirroring the Pine strategy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import Trade, aggregate_metrics


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.rolling(length, min_periods=length).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(span=length, adjust=False).mean()


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def _wma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def _hma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    half_len = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))
    wma1 = _wma(series, half_len)
    wma2 = _wma(series, length)
    diff = 2 * wma1 - wma2
    return _wma(diff, sqrt_len)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = _true_range(df)
    return _rma(tr, length)


def _linreg_series(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    idx = np.arange(length, dtype=float)

    def _calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        slope, intercept = np.polyfit(idx, values, 1)
        return slope * (length - 1) + intercept

    return series.rolling(length, min_periods=length).apply(_calc, raw=True)


def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = ha_close.copy()
    if len(df) > 0:
        ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
    for idx in range(1, len(df)):
        ha_open.iloc[idx] = (ha_open.iloc[idx - 1] + ha_close.iloc[idx - 1]) / 2.0
    ha_high = pd.concat([ha_open, ha_close, df["high"]], axis=1).max(axis=1)
    ha_low = pd.concat([ha_open, ha_close, df["low"]], axis=1).min(axis=1)
    ha["open"] = ha_open
    ha["high"] = ha_high
    ha["low"] = ha_low
    ha["close"] = ha_close
    return ha


def _directional_flux(df: pd.DataFrame, length: int) -> pd.Series:
    tr = _true_range(df)
    up_move = df["high"] - df["high"].shift()
    down_move = df["low"].shift() - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = _rma(pd.Series(plus_dm, index=df.index), length)
    minus_dm = _rma(pd.Series(minus_dm, index=df.index), length)
    atr = _rma(tr, length).replace(0, np.nan)
    plus_di = plus_dm / atr
    minus_di = minus_dm / atr
    return _rma(plus_di - minus_di, max(1, length // 2)) * 100


def _adx(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, length: int) -> pd.Series:
    df = pd.DataFrame({"high": series_high, "low": series_low, "close": series_close})
    tr = _true_range(df)
    up_move = series_high.diff()
    down_move = -series_low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (_rma(pd.Series(plus_dm, index=series_high.index), length) / _rma(tr, length))
    minus_di = 100 * (_rma(pd.Series(minus_dm, index=series_high.index), length) / _rma(tr, length))
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return _rma(dx.fillna(0.0), length)


def _bars_since(condition: pd.Series) -> pd.Series:
    last_idx = -1
    values: List[float] = []
    for idx, flag in enumerate(condition.fillna(False)):
        if flag:
            last_idx = 0
        elif last_idx >= 0:
            last_idx += 1
        values.append(float(last_idx) if last_idx >= 0 else np.nan)
    return pd.Series(values, index=condition.index)


def _estimate_tick(prices: pd.Series) -> float:
    diffs = prices.diff().abs()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float(prices.iloc[-1]) * 1e-6 if len(prices) else 0.01
    return float(diffs.min())


def _time_key(ts: pd.Timestamp) -> Tuple[int, int, int]:
    return ts.year, ts.weekofyear, ts.dayofyear


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------


@dataclass
class PositionState:
    side: int = 0
    qty: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    highest: float = 0.0
    lowest: float = 0.0
    bars_held: int = 0
    mfe: float = 0.0
    mae: float = 0.0

    def reset(self) -> None:
        self.side = 0
        self.qty = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.highest = 0.0
        self.lowest = 0.0
        self.bars_held = 0
        self.mfe = 0.0
        self.mae = 0.0


@dataclass
class GuardState:
    initial_capital: float
    leverage: float
    tradable: float
    equity: float
    withdrawable: float = 0.0
    peak_equity: float = 0.0
    daily_start: float = 0.0
    daily_peak: float = 0.0
    week_start: float = 0.0
    week_peak: float = 0.0
    daily_losses: int = 0
    loss_streak: int = 0
    guard_frozen: bool = False
    guard_fires: int = 0
    recent_trades: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.peak_equity = max(self.equity, self.initial_capital)
        self.daily_start = self.tradable
        self.daily_peak = self.tradable
        self.week_start = self.equity
        self.week_peak = self.equity

    def register_profit(self, profit: float, reserve_pct: float, use_wallet: bool) -> None:
        self.equity += profit
        if use_wallet and profit > 0:
            self.withdrawable += profit * reserve_pct
        effective_equity = self.equity - (self.withdrawable if use_wallet else 0.0)
        self.tradable = max(effective_equity, self.initial_capital * 0.01)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_peak = max(self.daily_peak, self.tradable)
        self.week_peak = max(self.week_peak, self.equity)

    def reset_daily(self) -> None:
        self.daily_start = self.tradable
        self.daily_peak = self.tradable
        self.daily_losses = 0
        self.guard_frozen = False

    def reset_week(self) -> None:
        self.week_start = self.equity
        self.week_peak = self.equity


# ---------------------------------------------------------------------------
# Backtest implementation
# ---------------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, float | bool | str],
    fees: Dict[str, float],
    risk: Dict[str, float],
    htf_df: Optional[pd.DataFrame] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, float]:
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError("DataFrame must contain OHLCV columns")

    params = dict(params)

    # --- Core parameters ---------------------------------------------------
    osc_len = int(params.get("len", params.get("oscLen", 12)))
    signal_len = int(params.get("sig", params.get("signalLen", 3)))
    use_same_len = bool(params.get("useSameLen", False))
    bb_len = int(params.get("sqz_bbLen", params.get("bbLen", 20)))
    kc_len = int(params.get("sqz_kcLen", params.get("kcLen", 18)))
    bb_mult = float(params.get("sqz_bbMult", params.get("bbMult", 1.4)))
    kc_mult = float(params.get("sqz_kcMult", params.get("kcMult", 1.0)))

    flux_len = int(params.get("dfl", params.get("fluxLen", 14)))
    flux_smooth_len = max(1, int(params.get("dfSmoothLen", params.get("fluxSmoothLen", 1))))
    use_flux_ha = bool(params.get("dfh", params.get("useFluxHeikin", True)))

    # --- Threshold parameters ---------------------------------------------
    use_dynamic_thresh = bool(params.get("useDynamicThresh", True))
    use_sym_threshold = bool(params.get("useSymThreshold", False))
    stat_threshold = float(params.get("statThreshold", 38.0))
    buy_threshold = float(params.get("buyThreshold", 36.0))
    sell_threshold = float(params.get("sellThreshold", 36.0))
    dyn_len = int(params.get("dynLen", 21))
    dyn_mult = float(params.get("dynMult", 1.1))

    # --- Exit parameters ---------------------------------------------------
    exit_opposite = bool(params.get("exitOpposite", True))
    use_mom_fade = bool(params.get("useMomFade", False))
    mom_fade_bars = int(params.get("momFadeBars", 1))
    mom_fade_reg_len = int(params.get("momFadeRegLen", 20))
    mom_fade_bb_len = int(params.get("momFadeBbLen", 20))
    mom_fade_kc_len = int(params.get("momFadeKcLen", 20))
    mom_fade_bb_mult = float(params.get("momFadeBbMult", 2.0))
    mom_fade_kc_mult = float(params.get("momFadeKcMult", 1.5))
    mom_fade_use_tr = bool(params.get("momFadeUseTrueRange", True))
    mom_fade_zero_delay = int(params.get("momFadeZeroDelay", 0))
    mom_fade_min_abs = float(params.get("momFadeMinAbs", 0.0))
    mom_fade_release_only = bool(params.get("momFadeReleaseOnly", True))
    mom_fade_min_after = int(params.get("momFadeMinBarsAfterRel", 1))
    mom_fade_window = int(params.get("momFadeWindowBars", 6))
    mom_fade_two_bars = bool(params.get("momFadeRequireTwoBars", False))
    min_hold_bars = int(params.get("minHoldBars", params.get("min_hold_bars", 0)))

    use_stop_loss = bool(params.get("useStopLoss", False))
    stop_lookback = int(params.get("stopLookback", 5))
    use_atr_trail = bool(params.get("useAtrTrail", False))
    atr_trail_len = int(params.get("atrTrailLen", 7))
    atr_trail_mult = float(params.get("atrTrailMult", 2.5))
    use_breakeven = bool(params.get("useBreakevenStop", params.get("useBreakeven", False)))
    breakeven_mult = float(params.get("breakevenMult", 1.0))
    use_pivot_stop = bool(params.get("usePivotStop", False))
    pivot_len = int(params.get("pivotLen", 5))
    use_pivot_htf = bool(params.get("usePivotHtf", False))
    pivot_tf = params.get("pivotTf", "5")
    use_atr_profit = bool(params.get("useAtrProfit", False))
    atr_profit_mult = float(params.get("atrProfitMult", 2.0))
    use_dyn_vol = bool(params.get("useDynVol", False))
    use_stop_guard = bool(params.get("useStopDistanceGuard", False))
    max_stop_atr_mult = float(params.get("maxStopAtrMult", 2.8))
    use_time_stop = bool(params.get("useTimeStop", False))
    max_hold_bars = int(params.get("maxHoldBars", 45))
    use_kasa = bool(params.get("useKASA", False))
    kasa_rsi_len = int(params.get("kasa_rsiLen", 14))
    kasa_rsi_ob = float(params.get("kasa_rsiOB", 72.0))
    kasa_rsi_os = float(params.get("kasa_rsiOS", 28.0))
    use_be_tiers = bool(params.get("useBETiers", False))

    use_shock = bool(params.get("useShock", False))
    atr_fast_len = int(params.get("atrFastLen", 5))
    atr_slow_len = int(params.get("atrSlowLen", 20))
    shock_mult = float(params.get("shockMult", 2.5))
    shock_action = params.get("shockAction", "손절 타이트닝")

    # --- Filters -----------------------------------------------------------
    use_adx = bool(params.get("useAdx", False))
    adx_len = int(params.get("adxLen", 10))
    adx_thresh = float(params.get("adxThresh", 15.0))
    use_ema_filter = bool(params.get("useEma", False))
    ema_fast_len = int(params.get("emaFastLen", 8))
    ema_slow_len = int(params.get("emaSlowLen", 20))
    ema_mode = params.get("emaMode", "Trend")
    use_bb_filter = bool(params.get("useBb", False))
    bb_filter_len = int(params.get("bbLenFilter", 20))
    bb_filter_mult = float(params.get("bbMultFilter", 2.0))
    use_stoch_rsi = bool(params.get("useStochRsi", False))
    stoch_len = int(params.get("stochLen", 14))
    stoch_ob = float(params.get("stochOB", 80.0))
    stoch_os = float(params.get("stochOS", 20.0))
    use_obv = bool(params.get("useObv", False))
    obv_smooth_len = int(params.get("obvSmoothLen", 3))
    use_atr_diff = bool(params.get("useAtrDiff", False))
    adx_atr_tf = params.get("adxAtrTf", "5")
    use_htf_trend = bool(params.get("useHtfTrend", False))
    htf_trend_tf = params.get("htfTrendTf", "240")
    htf_ma_len = int(params.get("htfMaLen", 20))
    use_hma_filter = bool(params.get("useHmaFilter", False))
    hma_len = int(params.get("hmaLen", 20))
    use_range_filter = bool(params.get("useRangeFilter", False))
    range_tf = params.get("rangeTf", "5")
    range_bars = int(params.get("rangeBars", 20))
    range_percent = float(params.get("rangePercent", 1.0))

    use_regime_filter = bool(params.get("useRegimeFilter", False))
    ctx_htf_tf = params.get("ctxHtfTf", "240")
    ctx_htf_ema_len = int(params.get("ctxHtfEmaLen", 120))
    ctx_htf_adx_len = int(params.get("ctxHtfAdxLen", 14))
    ctx_htf_adx_th = float(params.get("ctxHtfAdxTh", 22.0))
    use_slope_filter = bool(params.get("useSlopeFilter", False))
    slope_lookback = int(params.get("slopeLookback", 8))
    slope_min_pct = float(params.get("slopeMinPct", 0.06))
    use_distance_guard = bool(params.get("useDistanceGuard", False))
    distance_atr_len = int(params.get("distanceAtrLen", 21))
    distance_trend_len = int(params.get("distanceTrendLen", 55))
    distance_max_atr = float(params.get("distanceMaxAtr", 2.4))
    use_equity_slope = bool(params.get("useEquitySlopeFilter", False))
    eq_slope_len = int(params.get("eqSlopeLen", 120))

    # --- Structure gate ---------------------------------------------------
    use_structure_gate = bool(params.get("useStructureGate", False))
    use_bos = bool(params.get("useBOS", False))
    bos_state_bars = int(params.get("bos_stateBars", 5))
    use_choch = bool(params.get("useCHOCH", False))
    choch_state_bars = int(params.get("choch_stateBars", 5))
    structure_mode = params.get("structureGateMode", "어느 하나 충족")
    bos_tf = params.get("bosTf", "15")
    bos_lookback = int(params.get("bosLookback", 50))
    pivot_left_vn = int(params.get("pivotLeft_vn", 5))
    pivot_right_vn = int(params.get("pivotRight_vn", 5))

    use_sqz_gate = bool(params.get("useSqzGate", False))
    sqz_release_bars = int(params.get("sqzReleaseBars", 5))

    # --- Risk configuration ----------------------------------------------
    leverage = float(risk.get("leverage", params.get("leverage", 10.0)))
    qty_pct = float(risk.get("qty_pct", 30.0)) / 100.0
    commission = float(fees.get("commission_pct", 0.0))
    slippage_ticks = float(fees.get("slippage_ticks", params.get("slipTicks", 1)))
    non_finite_penalty = float(risk.get("non_finite_penalty", -1e6))

    base_qty_percent = float(params.get("baseQtyPercent", 30.0))
    use_sizing_override = bool(params.get("useSizingOverride", False))
    sizing_mode = params.get("sizingMode", "자본 비율")
    advanced_percent = float(params.get("advancedPercent", 25.0))
    fixed_usd_amount = float(params.get("fixedUsdAmount", 100.0))
    fixed_contract_size = float(params.get("fixedContractSize", 1.0))
    risk_sizing_type = params.get("riskSizingType", "손절 기반 %")
    base_risk_pct = float(params.get("baseRiskPct", 0.6))
    risk_contract_size = float(params.get("riskContractSize", 1.0))
    use_wallet = bool(params.get("useWallet", False))
    profit_reserve_pct = float(params.get("profitReservePct", 0.2))
    apply_reserve_to_sizing = bool(params.get("applyReserveToSizing", True))
    min_tradable_capital = float(params.get("minTradableCapital", 250.0))
    use_drawdown_scaling = bool(params.get("useDrawdownScaling", False))
    drawdown_trigger_pct = float(params.get("drawdownTriggerPct", 7.0))
    drawdown_risk_scale = float(params.get("drawdownRiskScale", 0.5))
    use_perf_risk = bool(params.get("usePerfAdaptiveRisk", False))
    par_lookback = int(params.get("parLookback", 6))
    par_min_trades = int(params.get("parMinTrades", 3))
    par_hot_win = float(params.get("parHotWinRate", 65.0))
    par_cold_win = float(params.get("parColdWinRate", 35.0))
    par_hot_mult = float(params.get("parHotRiskMult", 1.25))
    par_cold_mult = float(params.get("parColdRiskMult", 0.35))
    par_pause_on_cold = bool(params.get("parPauseOnCold", True))
    use_vol_guard = bool(params.get("useVolatilityGuard", False))
    vol_lookback = int(params.get("volatilityLookback", 50))
    vol_lower_pct = float(params.get("volatilityLowerPct", 0.15))
    vol_upper_pct = float(params.get("volatilityUpperPct", 2.5))

    use_session_filter = bool(params.get("useSessionFilter", False))
    use_day_filter = bool(params.get("useDayFilter", False))
    allow_long = bool(params.get("allowLongEntry", True))
    allow_short = bool(params.get("allowShortEntry", True))
    reentry_bars = int(params.get("reentryBars", 0))
    start_timestamp = pd.to_datetime(params.get("startDate", params.get("start", None)), utc=True, errors="ignore")
    maintenance_margin_pct = float(params.get("maintenanceMarginPct", 0.5))
    preempt_ticks = int(params.get("preemptTicks", 8))

    use_daily_loss_guard = bool(params.get("useDailyLossGuard", False))
    daily_loss_limit = float(params.get("dailyLossLimit", 80.0))
    use_daily_profit_lock = bool(params.get("useDailyProfitLock", False))
    daily_profit_target = float(params.get("dailyProfitTarget", 120.0))
    use_weekly_profit_lock = bool(params.get("useWeeklyProfitLock", False))
    weekly_profit_target = float(params.get("weeklyProfitTarget", 250.0))
    use_loss_streak_guard = bool(params.get("useLossStreakGuard", False))
    max_consecutive_losses = int(params.get("maxConsecutiveLosses", 3))
    use_capital_guard = bool(params.get("useCapitalGuard", False))
    capital_guard_pct = float(params.get("capitalGuardPct", 20.0))
    max_daily_losses = int(params.get("maxDailyLosses", 0))
    max_weekly_dd = float(params.get("maxWeeklyDD", 0.0))
    max_guard_fires = int(params.get("maxGuardFires", 0))
    use_guard_exit = bool(params.get("useGuardExit", False))

    # --- Derived configuration -------------------------------------------
    initial_capital = float(params.get("initial_capital", 500.0))
    if np.isnan(initial_capital) or initial_capital <= 0:
        initial_capital = 500.0

    index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    tick_size = _estimate_tick(df["close"])
    slippage_value = tick_size * slippage_ticks

    ha_df = _heikin_ashi(df) if use_flux_ha else df

    hl2 = (df["high"] + df["low"]) / 2.0
    hl2_sma = _sma(hl2, osc_len)
    highest = df["high"].rolling(osc_len, min_periods=osc_len).max()
    lowest = df["low"].rolling(osc_len, min_periods=osc_len).min()
    channel_mid = (highest + lowest) / 2.0
    atr = _atr(df, osc_len).replace(0, np.nan)
    avg_line = (channel_mid + hl2_sma) / 2.0
    pressure = (_sma(df["close"], bb_len if use_same_len else bb_len) - _sma(df["close"], kc_len if use_same_len else kc_len)).fillna(0.0)
    momentum_input = ((df["close"] - avg_line) / atr * 100).fillna(0.0) + pressure.fillna(0.0)
    momentum = _linreg_series(momentum_input, osc_len)
    momentum_signal = momentum.rolling(signal_len, min_periods=signal_len).mean()

    flux_raw = _directional_flux(ha_df, flux_len)
    if flux_smooth_len > 1:
        flux_smoothed = flux_raw.rolling(flux_smooth_len, min_periods=flux_smooth_len).mean()
    else:
        flux_smoothed = flux_raw

    mom_fade_source = (df["high"] + df["low"] + df["close"]) / 3.0
    mom_fade_basis = _sma(mom_fade_source, mom_fade_bb_len)
    mom_fade_dev = mom_fade_source.rolling(mom_fade_bb_len, min_periods=mom_fade_bb_len).std(ddof=0) * mom_fade_bb_mult
    mom_fade_range = _true_range(df) if mom_fade_use_tr else (df["high"] - df["low"])
    mom_fade_range_ma = (_rma(mom_fade_range, mom_fade_kc_len) if mom_fade_use_tr else _sma(mom_fade_range, mom_fade_kc_len)) * mom_fade_kc_mult
    mom_fade_mid = mom_fade_basis + (mom_fade_range_ma / 2.0)
    mom_fade_hist = _linreg_series(mom_fade_source - mom_fade_mid, mom_fade_reg_len)

    gate_sq_on = mom_fade_dev < mom_fade_range_ma
    gate_sq_rel = gate_sq_on.shift(1).fillna(False) & (~gate_sq_on) | (
        (mom_fade_dev > mom_fade_range_ma) & (mom_fade_dev.shift(1) <= mom_fade_range_ma.shift(1))
    )
    gate_bars_since = _bars_since(gate_sq_rel)
    gate_release_seen = gate_bars_since.notna()
    gate_sq_valid = gate_release_seen & (gate_bars_since <= sqz_release_bars) & (~gate_sq_on)

    if use_dynamic_thresh:
        dyn_std = momentum.rolling(max(1, dyn_len), min_periods=max(1, dyn_len)).std() * dyn_mult
        fallback = abs(stat_threshold) if stat_threshold else dyn_std.dropna().median()
        dyn_std = dyn_std.fillna(fallback if fallback else abs(stat_threshold) or 1.0)
        buy_thresh_series = -dyn_std.abs()
        sell_thresh_series = dyn_std.abs()
    else:
        if use_sym_threshold:
            buy_val = -abs(stat_threshold)
            sell_val = abs(stat_threshold)
        else:
            buy_val = -abs(buy_threshold)
            sell_val = abs(sell_threshold)
        buy_thresh_series = pd.Series(buy_val, index=index)
        sell_thresh_series = pd.Series(sell_val, index=index)

    # Higher timeframe helpers (best effort fallback to same frame)
    def _security(series: pd.Series, timeframe: str, agg: str = "last") -> pd.Series:
        if timeframe in {"", df.index.inferred_freq}:
            return series
        try:
            rule = pd.Timedelta(timeframe)
        except Exception:
            try:
                rule = pd.Timedelta(int(timeframe), unit="m")
            except Exception:
                return series
        resampled = series.resample(rule, label="right", closed="right").agg(agg)
        return resampled.reindex(series.index, method="ffill")

    if use_atr_diff:
        atr_htf = _security(df["high"], adx_atr_tf, agg="max")  # placeholder
        atr_htf_series = _security(df["close"], adx_atr_tf, agg="last")
        atr_diff_htf = atr_htf_series.rolling(adx_len).mean().diff().fillna(0.0)
    else:
        atr_diff_htf = pd.Series(0.0, index=index)

    if use_htf_trend:
        htf_ma_series = _security(df["close"], htf_trend_tf, agg="last").rolling(htf_ma_len, min_periods=htf_ma_len).mean()
    else:
        htf_ma_series = pd.Series(np.nan, index=index)

    if use_range_filter:
        range_high = _security(df["high"], range_tf, agg="max").rolling(range_bars, min_periods=range_bars).max()
        range_low = _security(df["low"], range_tf, agg="min").rolling(range_bars, min_periods=range_bars).min()
        range_perc = np.where(range_low != 0, (range_high - range_low) / range_low * 100.0, 0.0)
        in_range_box = pd.Series(range_perc, index=index) <= range_percent
    else:
        in_range_box = pd.Series(False, index=index)

    if use_regime_filter:
        regime_close = _security(df["close"], ctx_htf_tf, agg="last")
        regime_ema = _security(df["close"], ctx_htf_tf, agg="last").ewm(span=ctx_htf_ema_len, adjust=False).mean()
        regime_adx = _adx(
            _security(df["high"], ctx_htf_tf, agg="max"),
            _security(df["low"], ctx_htf_tf, agg="min"),
            _security(df["close"], ctx_htf_tf, agg="last"),
            ctx_htf_adx_len,
        )
    else:
        regime_close = df["close"]
        regime_ema = _ema(df["close"], ctx_htf_ema_len)
        regime_adx = _adx(df["high"], df["low"], df["close"], ctx_htf_adx_len)

    # Structure gate proxies
    if use_structure_gate:
        bos_high = _security(df["high"], bos_tf, agg="max").rolling(bos_lookback, min_periods=bos_lookback).max()
        bos_low = _security(df["low"], bos_tf, agg="min").rolling(bos_lookback, min_periods=bos_lookback).min()
        bos_long_event = df["close"].gt(bos_high.shift()).fillna(False)
        bos_short_event = df["close"].lt(bos_low.shift()).fillna(False)
        bos_long_state = _bars_since(bos_long_event) <= bos_state_bars
        bos_short_state = _bars_since(bos_short_event) <= bos_state_bars
        pivot_high = _security(df["high"], bos_tf, agg="max").rolling(pivot_left_vn + pivot_right_vn + 1, min_periods=pivot_left_vn + pivot_right_vn + 1).max()
        pivot_low = _security(df["low"], bos_tf, agg="min").rolling(pivot_left_vn + pivot_right_vn + 1, min_periods=pivot_left_vn + pivot_right_vn + 1).min()
        choch_long_event = df["close"].gt(pivot_high.shift()).fillna(False)
        choch_short_event = df["close"].lt(pivot_low.shift()).fillna(False)
        choch_long_state = _bars_since(choch_long_event) <= choch_state_bars
        choch_short_state = _bars_since(choch_short_event) <= choch_state_bars
    else:
        bos_long_state = bos_short_state = choch_long_state = choch_short_state = pd.Series(True, index=index)

    # Account + state initialisation
    guard_state = GuardState(
        initial_capital=initial_capital,
        leverage=leverage,
        tradable=initial_capital,
        equity=initial_capital,
    )

    position = PositionState()
    returns = pd.Series(0.0, index=index)
    trades: List[Trade] = []

    reentry_countdown = 0
    reversal_countdown = 0
    last_pos_dir = 0

    daily_marker = index[0].date() if not index.empty else None
    week_marker = index[0].isocalendar()[1] if not index.empty else None

    def _calc_order_size(close_price: float, stop_distance: float, risk_mult: float) -> float:
        mult = max(risk_mult, 0.0)
        risk_scale = (guard_state.tradable > 0 and base_risk_pct > 0)
        scale = (guard_state.tradable > 0) and base_risk_pct > 0
        if scale:
            risk_scale = guard_state.tradable
        risk_ratio = base_risk_pct if base_risk_pct > 0 else 1.0
        scaled_risk_pct = base_risk_pct
        current_dd = 0.0 if guard_state.peak_equity == 0 else (guard_state.peak_equity - guard_state.equity) / guard_state.peak_equity * 100.0
        if use_drawdown_scaling and current_dd > drawdown_trigger_pct:
            scaled_risk_pct = base_risk_pct * drawdown_risk_scale
        perf_mult = 1.0
        if use_perf_risk and guard_state.recent_trades:
            wins = sum(1 for val in guard_state.recent_trades if val > 0)
            rate = wins / len(guard_state.recent_trades) * 100.0
            if len(guard_state.recent_trades) >= par_min_trades:
                if rate >= par_hot_win:
                    perf_mult = par_hot_mult
                elif rate <= par_cold_win:
                    perf_mult = par_cold_mult
        final_risk_pct = scaled_risk_pct * perf_mult

        effective_capital = guard_state.tradable
        if use_wallet and apply_reserve_to_sizing:
            effective_capital = max(guard_state.equity - guard_state.withdrawable, guard_state.initial_capital * 0.01)

        if not use_sizing_override:
            pct_to_use = max(base_qty_percent * mult * (final_risk_pct / base_risk_pct if base_risk_pct else 1.0), 0.0)
            capital_portion = effective_capital * pct_to_use / 100.0
            return (capital_portion * leverage) / close_price if close_price > 0 else 0.0

        mode = sizing_mode
        if mode == "자본 비율":
            pct_to_use = max(advanced_percent * mult * (final_risk_pct / base_risk_pct if base_risk_pct else 1.0), 0.0)
            capital_portion = effective_capital * pct_to_use / 100.0
            return (capital_portion * leverage) / close_price if close_price > 0 else 0.0
        if mode == "고정 금액 (USD)":
            usd_to_use = max(fixed_usd_amount, 0.0)
            return (usd_to_use * leverage) / close_price if close_price > 0 else 0.0
        if mode == "고정 계약":
            return max(fixed_contract_size * mult, 0.0)
        if mode == "리스크 기반":
            if risk_sizing_type == "고정 계약":
                return max(risk_contract_size * mult, 0.0)
            if stop_distance <= 0:
                return 0.0
            risk_pct = max(final_risk_pct * mult, 0.0)
            risk_capital = effective_capital * risk_pct / 100.0
            return risk_capital / (stop_distance + slippage_value) if risk_capital > 0 else 0.0
        return 0.0

    def _close_position(ts: pd.Timestamp, price: float, reason: str) -> None:
        nonlocal position, returns, trades
        if position.side == 0 or position.entry_time is None:
            return
        exit_price = price
        if position.side > 0:
            exit_price -= slippage_value
        else:
            exit_price += slippage_value
        gross = (exit_price - position.entry_price) * position.qty * position.side
        profit = gross * leverage
        notional = position.qty * position.entry_price * leverage
        profit -= notional * commission * 2
        guard_state.register_profit(profit, profit_reserve_pct, use_wallet)
        equity_before = guard_state.equity - profit
        equity_return = profit / equity_before if equity_before else 0.0
        returns.loc[ts] += equity_return
        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                direction="long" if position.side > 0 else "short",
                size=notional,
                entry_price=position.entry_price,
                exit_price=exit_price,
                profit=profit,
                return_pct=equity_return,
                mfe=position.mfe,
                mae=position.mae,
                bars_held=position.bars_held,
                reason=reason,
            )
        )
        position.reset()

    for ts, row in df.iterrows():
        if pd.isna(row["close"]):
            continue
        idx = ts
        if daily_marker is None or idx.date() != daily_marker:
            guard_state.reset_daily()
            daily_marker = idx.date()
        if week_marker is None or idx.isocalendar()[1] != week_marker:
            guard_state.reset_week()
            week_marker = idx.isocalendar()[1]

        if use_vol_guard:
            atr_pct = (_atr(df.loc[:idx], vol_lookback).iloc[-1] / row["close"]) * 100.0 if row["close"] else 0.0
        else:
            atr_pct = 0.0

        if start_timestamp is not None and idx < start_timestamp:
            continue

        mom_val = momentum.loc[idx]
        mom_sig = momentum_signal.loc[idx]
        if np.isnan(mom_val) or np.isnan(mom_sig):
            continue
        flux_val = flux_smoothed.loc[idx] if not np.isnan(flux_smoothed.loc[idx]) else 0.0
        buy_threshold_val = buy_thresh_series.loc[idx]
        sell_threshold_val = sell_thresh_series.loc[idx]

        cross_up = momentum.shift(1).loc[idx] < momentum_signal.shift(1).loc[idx] and mom_val > mom_sig
        cross_down = momentum.shift(1).loc[idx] > momentum_signal.shift(1).loc[idx] and mom_val < mom_sig
        long_signal = cross_up and mom_val <= buy_threshold_val and flux_val > 0
        short_signal = cross_down and mom_val >= sell_threshold_val and flux_val < 0

        if use_adx:
            adx_series = _security(df["close"], adx_atr_tf, agg="last")
            adx_value = _adx(df["high"], df["low"], df["close"], adx_len).loc[idx]
        else:
            adx_value = np.nan

        long_filters = True
        short_filters = True
        if use_adx and not np.isnan(adx_value):
            long_filters &= adx_value > adx_thresh
            short_filters &= adx_value > adx_thresh
        if use_ema_filter:
            ema_fast = _ema(df["close"], ema_fast_len).loc[idx]
            ema_slow = _ema(df["close"], ema_slow_len).loc[idx]
            if ema_mode == "Crossover":
                long_filters &= ema_fast > ema_slow
                short_filters &= ema_fast < ema_slow
            else:
                long_filters &= row["close"] > ema_slow
                short_filters &= row["close"] < ema_slow
        if use_bb_filter:
            bb_basis = _sma(df["close"], bb_filter_len).loc[idx]
            bb_dev = df["close"].rolling(bb_filter_len).std(ddof=0).loc[idx] * bb_filter_mult
            bb_upper = bb_basis + bb_dev
            bb_lower = bb_basis - bb_dev
            long_filters &= row["close"] <= bb_upper
            short_filters &= row["close"] >= bb_lower
        if use_stoch_rsi:
            long_filters &= stochRsiVal.loc[idx] <= stoch_os
            short_filters &= stochRsiVal.loc[idx] >= stoch_ob
        if use_obv:
            long_filters &= obvSlope.loc[idx] > 0
            short_filters &= obvSlope.loc[idx] < 0
        if use_atr_diff:
            long_filters &= atr_diff_htf.loc[idx] > 0
            short_filters &= atr_diff_htf.loc[idx] > 0
        if use_htf_trend:
            htf_val = htf_ma_series.loc[idx]
            long_filters &= row["close"] > htf_val
            short_filters &= row["close"] < htf_val
        if use_hma_filter:
            hma_val = _hma(df["close"], hma_len).loc[idx]
            long_filters &= row["close"] > hma_val
            short_filters &= row["close"] < hma_val
        if use_range_filter:
            long_filters &= not in_range_box.loc[idx]
            short_filters &= not in_range_box.loc[idx]

        slope_basis_val = _ema(df["close"], slope_lookback).loc[idx]
        prev_trend = slope_basis_val if slope_basis_val == slope_basis_val else slope_basis_val
        slope_pct = (slope_basis_val - prev_trend) / slope_basis_val * 100.0 if slope_basis_val else 0.0
        if use_slope_filter:
            long_filters &= slope_pct >= slope_min_pct
            short_filters &= slope_pct <= -slope_min_pct

        if use_distance_guard:
            dist_atr = _atr(df, distance_atr_len).loc[idx]
            vw_dist = abs(row["close"] - df["close"].rolling(distance_trend_len).mean().loc[idx]) / dist_atr if dist_atr else 0.0
            long_filters &= vw_dist <= distance_max_atr
            short_filters &= vw_dist <= distance_max_atr

        if use_equity_slope:
            eq_series = returns.cumsum().add(1.0).rolling(eq_slope_len).apply(lambda x: x[-1] - x[0], raw=True)
            eq_slope = eq_series.loc[idx] if eq_series.notna().any() else 0.0
            long_filters &= eq_slope >= 0
            short_filters &= eq_slope <= 0

        regime_long_ok = (not use_regime_filter) or (regime_close.loc[idx] > regime_ema.loc[idx] and regime_adx.loc[idx] > ctx_htf_adx_th)
        regime_short_ok = (not use_regime_filter) or (regime_close.loc[idx] < regime_ema.loc[idx] and regime_adx.loc[idx] > ctx_htf_adx_th)

        long_filters &= regime_long_ok
        short_filters &= regime_short_ok

        structure_all = structure_mode == "모두 충족"
        long_struct_ok = True
        short_struct_ok = True
        if use_structure_gate:
            if structure_all:
                long_struct_ok = (not use_bos or bos_long_state.loc[idx]) and (not use_choch or choch_long_state.loc[idx])
                short_struct_ok = (not use_bos or bos_short_state.loc[idx]) and (not use_choch or choch_short_state.loc[idx])
            else:
                long_struct_ok = (use_bos and bos_long_state.loc[idx]) or (use_choch and choch_long_state.loc[idx]) or (not use_bos and not use_choch)
                short_struct_ok = (use_bos and bos_short_state.loc[idx]) or (use_choch and choch_short_state.loc[idx]) or (not use_bos and not use_choch)

        sqz_gate_ok = True
        if use_sqz_gate:
            sqz_gate_ok = bool((not gate_sq_on.loc[idx]) and gate_sq_valid.loc[idx])

        session_ok = True
        day_ok = True
        if use_session_filter or use_day_filter:
            session_ok = True
            day_ok = True
        can_trade = (
            session_ok
            and day_ok
            and not guard_state.guard_frozen
            and (not use_vol_guard or (atr_pct >= vol_lower_pct and atr_pct <= vol_upper_pct))
            and guard_state.tradable >= min_tradable_capital
        )

        long_ok = (
            long_signal
            and long_filters
            and long_struct_ok
            and sqz_gate_ok
            and can_trade
            and allow_long
        )
        short_ok = (
            short_signal
            and short_filters
            and short_struct_ok
            and sqz_gate_ok
            and can_trade
            and allow_short
        )

        if reentry_countdown > 0 and position.side == 0:
            reentry_countdown -= 1
        if position.side == 0 and reentry_countdown > 0:
            long_ok = short_ok = False

        if last_pos_dir != 0 and reversal_countdown > 0:
            reversal_countdown -= 1
        if use_reversal := bool(params.get("useReversal", False)):
            reversal_delay_sec = float(params.get("reversalDelaySec", 0.0))
            if position.side == 0 and last_pos_dir != 0 and reversal_countdown == 0 and can_trade:
                if last_pos_dir == 1:
                    short_ok = True
                else:
                    long_ok = True
                last_pos_dir = 0

        if position.side != 0:
            position.bars_held += 1
            if position.side > 0:
                favorable = (row["high"] - position.entry_price) / position.entry_price * leverage
                adverse = (row["low"] - position.entry_price) / position.entry_price * leverage
            else:
                favorable = (position.entry_price - row["low"]) / position.entry_price * leverage
                adverse = (position.entry_price - row["high"]) / position.entry_price * leverage
            position.mfe = max(position.mfe, favorable)
            position.mae = min(position.mae, -abs(adverse))

        if guard_state.guard_frozen and position.side != 0 and use_guard_exit:
            qty = abs(position.qty)
            entry_price = position.entry_price
            direction = position.side
            initial_margin = (qty * entry_price) / leverage
            maint_margin = (qty * entry_price) * (maintenance_margin_pct / 100.0)
            offset = (initial_margin - maint_margin) / qty if qty else 0.0
            liq_price = entry_price - offset if direction > 0 else entry_price + offset
            preempt_price = liq_price + preempt_ticks * tick_size if direction > 0 else liq_price - preempt_ticks * tick_size
            if (direction > 0 and row["low"] <= preempt_price) or (direction < 0 and row["high"] >= preempt_price):
                _close_position(idx, preempt_price, "guard_exit")
                guard_state.guard_fires += 1
                continue

        # Exits
        exit_reason = None
        exit_price = row["close"]

        if position.side > 0:
            if exit_opposite and short_signal and position.bars_held >= min_hold_bars:
                exit_reason = "reverse_short"
            elif use_mom_fade and position.bars_held >= min_hold_bars and mom_fade_hist.loc[idx] > 0:
                if mom_fade_hist.loc[idx] > 0 and mom_fade_hist.loc[idx] < mom_fade_hist.shift(1).loc[idx]:
                    exit_reason = "fade_long"
            if use_time_stop and max_hold_bars > 0 and position.bars_held >= max_hold_bars:
                exit_reason = "time_stop"
            if use_kasa and pd.notna(kasa_rsi := _ema(df["close"], kasa_rsi_len).loc[idx]) and kasa_rsi < kasa_rsi_ob:
                exit_reason = "kasa"
        elif position.side < 0:
            if exit_opposite and long_signal and position.bars_held >= min_hold_bars:
                exit_reason = "reverse_long"
            elif use_mom_fade and position.bars_held >= min_hold_bars and mom_fade_hist.loc[idx] < 0:
                if mom_fade_hist.loc[idx] < 0 and mom_fade_hist.loc[idx] > mom_fade_hist.shift(1).loc[idx]:
                    exit_reason = "fade_short"
            if use_time_stop and max_hold_bars > 0 and position.bars_held >= max_hold_bars:
                exit_reason = "time_stop"
            if use_kasa and pd.notna(kasa_rsi := _ema(df["close"], kasa_rsi_len).loc[idx]) and kasa_rsi > kasa_rsi_os:
                exit_reason = "kasa"

        if use_breakeven and position.side != 0:
            atr_trail_val = _atr(df, atr_trail_len).loc[idx]
            if position.side > 0 and position.highest - position.entry_price >= atr_trail_val * breakeven_mult:
                exit_price = max(exit_price, position.entry_price)
            if position.side < 0 and position.entry_price - position.lowest >= atr_trail_val * breakeven_mult:
                exit_price = min(exit_price, position.entry_price)

        if exit_reason and position.side != 0:
            _close_position(idx, exit_price, exit_reason)
            reentry_countdown = reentry_bars
            last_pos_dir = 1 if position.side > 0 else -1
            if use_reversal:
                bar_seconds = (idx - index.shift(1).loc[idx]).total_seconds() if idx in index[1:] else 0
                if bar_seconds > 0:
                    reversal_countdown = int(np.round(float(params.get("reversalDelaySec", 0.0)) / bar_seconds))
            continue

        # Entries
        if position.side == 0:
            if long_ok:
                stop_hint = tick_size
                if use_stop_loss:
                    swing_low = df["low"].rolling(stop_lookback).min().loc[idx]
                    if not np.isnan(swing_low):
                        stop_hint = max(stop_hint, row["close"] - swing_low)
                if use_atr_trail:
                    stop_hint = max(stop_hint, _atr(df, atr_trail_len).loc[idx] * atr_trail_mult)
                stop_dist = max(stop_hint, tick_size)
                qty = _calc_order_size(row["close"], stop_dist, 1.0)
                if qty > 0 and (not use_stop_guard or _atr(df, osc_len).loc[idx] == 0 or stop_dist <= _atr(df, osc_len).loc[idx] * max_stop_atr_mult):
                    position.side = 1
                    position.qty = qty
                    position.entry_price = row["close"] + slippage_value
                    position.entry_time = idx
                    position.highest = row["high"]
                    position.lowest = row["low"]
                    position.bars_held = 0
            elif short_ok:
                stop_hint = tick_size
                if use_stop_loss:
                    swing_high = df["high"].rolling(stop_lookback).max().loc[idx]
                    if not np.isnan(swing_high):
                        stop_hint = max(stop_hint, swing_high - row["close"])
                if use_atr_trail:
                    stop_hint = max(stop_hint, _atr(df, atr_trail_len).loc[idx] * atr_trail_mult)
                stop_dist = max(stop_hint, tick_size)
                qty = _calc_order_size(row["close"], stop_dist, 1.0)
                if qty > 0 and (not use_stop_guard or _atr(df, osc_len).loc[idx] == 0 or stop_dist <= _atr(df, osc_len).loc[idx] * max_stop_atr_mult):
                    position.side = -1
                    position.qty = qty
                    position.entry_price = row["close"] - slippage_value
                    position.entry_time = idx
                    position.highest = row["high"]
                    position.lowest = row["low"]
                    position.bars_held = 0

    if position.side != 0 and position.entry_time is not None:
        _close_position(index[-1], df.iloc[-1]["close"], "end_of_data")

    metrics = aggregate_metrics(trades, returns)
    metrics["Returns"] = returns
    metrics["TradesList"] = trades
    metrics["Trades"] = float(len(trades))
    metrics["Wins"] = float(sum(1 for trade in trades if trade.profit > 0))
    metrics["Losses"] = float(sum(1 for trade in trades if trade.profit < 0))
    metrics["MinTrades"] = float(min_trades if min_trades is not None else risk.get("min_trades", 0))
    metrics["Valid"] = metrics["Trades"] >= metrics["MinTrades"]
    return metrics

