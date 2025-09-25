import pandas as pd
import pytest

from optimize.metrics import Trade, aggregate_metrics, equity_curve_from_returns, max_drawdown, score_metrics
from optimize.strategy_model import run_backtest


def test_equity_curve_and_drawdown():
    returns = pd.Series([0.01, -0.02, 0.015, -0.01])
    equity = equity_curve_from_returns(returns, initial=1.0)
    dd = max_drawdown(equity)
    assert equity.iloc[-1] == pytest.approx(0.9946, rel=1e-3)
    assert dd <= 0


def test_aggregate_metrics_basic():
    trades = [
        Trade(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), "long", 1.0, 100, 101, 0.01, 0.01, 0.02, -0.01, 5),
        Trade(pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"), "short", 1.0, 105, 104, 0.01, 0.01, 0.03, -0.02, 4),
    ]
    returns = pd.Series([0.01, 0.0, -0.005])
    metrics = aggregate_metrics(trades, returns)
    assert metrics["NetProfit"] != 0
    assert metrics["Trades"] == pytest.approx(2.0)
    assert metrics["WinRate"] == 1.0
    assert metrics["ProfitFactor"] > 0
    assert "WeeklyNetProfit" in metrics
    assert "Expectancy" in metrics


def test_run_backtest_deterministic():
    data = pd.read_csv("tests/tests_data/sample_ohlcv.csv", parse_dates=["timestamp"], index_col="timestamp")
    params = {"wtLen": 6, "thr": 0.3, "atrMult": 1.0, "useHTF": False, "useRSI": False, "useMACD": False, "useStoch": False}
    fees = {"commission_pct": 0.0005, "slippage_ticks": 1}
    risk = {
        "leverage": 2,
        "qty_pct": 10,
        "min_trades": 1,
        "min_hold_bars": 0,
        "max_consecutive_losses": 10,
        "penalty_trade": 0.0,
        "penalty_hold": 0.0,
        "penalty_consecutive_loss": 0.0,
    }

    first = run_backtest(data, params, fees, risk)
    second = run_backtest(data, params, fees, risk)
    assert first["Trades"] == second["Trades"]
    assert first["NetProfit"] == second["NetProfit"]
    assert first["Valid"] == second["Valid"]


def test_score_metrics_handles_objectives():
    metrics = {
        "NetProfit": 0.5,
        "MaxDD": -0.1,
        "Sortino": 1.2,
        "WinRate": 0.6,
        "Trades": 50,
        "MinTrades": 20,
        "AvgHoldBars": 5,
        "MinHoldBars": 1,
        "MaxConsecutiveLosses": 2,
        "MaxConsecutiveLossLimit": 5,
    }
    score = score_metrics(metrics, ["NetProfit", "MaxDD", "Sortino", "WinRate"])
    assert score > 0


def test_score_metrics_applies_penalties():
    metrics = {
        "NetProfit": 0.5,
        "Trades": 5,
        "MinTrades": 10,
        "TradePenalty": 1.0,
        "AvgHoldBars": 0.5,
        "MinHoldBars": 1.0,
        "HoldPenalty": 0.5,
        "MaxConsecutiveLosses": 4,
        "MaxConsecutiveLossLimit": 2,
        "ConsecutiveLossPenalty": 1.0,
    }
    score = score_metrics(metrics, ["NetProfit"])
    assert score < 0.5
