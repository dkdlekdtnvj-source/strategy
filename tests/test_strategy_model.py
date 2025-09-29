import pandas as pd

from optimize.strategy_model import run_backtest


def _make_ohlcv(prices):
    index = pd.date_range("2025-07-01", periods=len(prices), freq="1min", tz="UTC")
    close = pd.Series(prices, index=index)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )
    return df


def _base_params(**overrides):
    params = {
        "oscLen": 3,
        "signalLen": 1,
        "fluxLen": 3,
        "useFluxHeikin": False,
        "useDynamicThresh": False,
        "useSymThreshold": True,
        "statThreshold": 0.0,
        "startDate": "2025-07-01T00:00:00",
        "allowLongEntry": True,
        "allowShortEntry": False,
        "debugForceLong": True,
    }
    params.update(overrides)
    return params


FEES = {"commission_pct": 0.0, "slippage_ticks": 0.0}
RISK = {"initial_capital": 1000.0, "min_trades": 0, "min_hold_bars": 0, "max_consecutive_losses": 10}


def test_debug_force_long_creates_trade():
    df = _make_ohlcv([100, 101, 102, 103, 104, 105])
    params = _base_params(useTimeStop=True, maxHoldBars=1)

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] >= 1


def test_daily_loss_guard_freezes_after_loss():
    prices = [100, 99, 98, 97, 96, 95, 94, 93]
    df = _make_ohlcv(prices)
    params = _base_params(
        useTimeStop=True,
        maxHoldBars=1,
        useDailyLossGuard=True,
        dailyLossLimit=0.5,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 1
    assert metrics["GuardFrozen"] == 1.0


def test_squeeze_gate_blocks_without_release():
    df = _make_ohlcv([100] * 20)
    params = _base_params(
        useSqzGate=True,
        sqzReleaseBars=0,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 0


def test_stop_distance_guard_prevents_entry():
    prices = [100, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0]
    df = _make_ohlcv(prices)
    params = _base_params(
        useStopDistanceGuard=True,
        maxStopAtrMult=0.5,
    )

    metrics = run_backtest(df, params, FEES, RISK)

    assert metrics["Trades"] == 0
