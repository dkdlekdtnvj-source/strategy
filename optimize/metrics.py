"""Performance metric calculations for optimisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Container describing the outcome of a single trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    size: float
    entry_price: float
    exit_price: float
    profit: float
    return_pct: float
    mfe: float
    mae: float
    bars_held: int
    reason: str = ""


def equity_curve_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Create an equity curve from percentage returns."""

    equity = (1 + returns.fillna(0)).cumprod() * initial
    return equity


def max_drawdown(equity: pd.Series) -> float:
    """Return the maximum drawdown as a negative percentage."""

    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    downside = returns[returns < risk_free]
    if downside.empty:
        return float("inf")
    expected = returns.mean() - risk_free
    downside_std = downside.std(ddof=0)
    return float(expected / downside_std) if downside_std != 0 else float("inf")


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    std = returns.std(ddof=0)
    if std == 0:
        return float("inf")
    return float((returns.mean() - risk_free) / std)


def profit_factor(trades: Iterable[Trade]) -> float:
    gross_profit = sum(max(trade.profit, 0.0) for trade in trades)
    gross_loss = sum(min(trade.profit, 0.0) for trade in trades)
    return float(gross_profit / abs(gross_loss)) if gross_loss else float("inf")


def win_rate(trades: Sequence[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for trade in trades if trade.profit > 0)
    return wins / len(trades)


def average_rr(trades: Sequence[Trade]) -> float:
    rs = [trade.mfe / abs(trade.mae) for trade in trades if trade.mae < 0]
    return float(np.mean(rs)) if rs else 0.0


def average_hold_time(trades: Sequence[Trade]) -> float:
    holds = [trade.bars_held for trade in trades]
    return float(np.mean(holds)) if holds else 0.0


def _consecutive_losses(trades: Sequence[Trade]) -> int:
    streak = 0
    worst = 0
    for trade in trades:
        if trade.profit < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def _weekly_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    weekly = returns.resample("W").sum()
    return weekly.dropna()


def aggregate_metrics(trades: List[Trade], returns: pd.Series) -> Dict[str, float]:
    """Aggregate trade-level information into rich performance metrics."""

    returns = returns.fillna(0.0)
    equity = equity_curve_from_returns(returns, initial=1.0)
    net_profit = float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]) if len(equity) > 1 else 0.0

    weekly = _weekly_returns(returns)
    weekly_mean = float(weekly.mean()) if not weekly.empty else 0.0
    weekly_std = float(weekly.std(ddof=0)) if len(weekly) > 1 else 0.0

    gross_profit = float(sum(max(trade.profit, 0.0) for trade in trades))
    gross_loss = float(sum(min(trade.profit, 0.0) for trade in trades))
    wins = sum(1 for trade in trades if trade.profit > 0)
    losses = sum(1 for trade in trades if trade.profit < 0)

    metrics: Dict[str, float] = {
        "NetProfit": net_profit,
        "TotalReturn": net_profit,
        "MaxDD": float(max_drawdown(equity)),
        "WinRate": float(win_rate(trades)),
        "ProfitFactor": float(profit_factor(trades)),
        "Sortino": float(sortino_ratio(returns)),
        "Sharpe": float(sharpe_ratio(returns)),
        "AvgRR": float(average_rr(trades)),
        "AvgHoldBars": float(average_hold_time(trades)),
        "Trades": float(len(trades)),
        "Wins": float(wins),
        "Losses": float(losses),
        "GrossProfit": gross_profit,
        "GrossLoss": gross_loss,
        "Expectancy": float((gross_profit + gross_loss) / len(trades)) if trades else 0.0,
        "WeeklyNetProfit": weekly_mean,
        "WeeklyReturnStd": weekly_std,
        "MaxConsecutiveLosses": float(_consecutive_losses(trades)),
    }

    mfe = [trade.mfe for trade in trades]
    mae = [trade.mae for trade in trades]
    metrics["AvgMFE"] = float(np.mean(mfe)) if mfe else 0.0
    metrics["AvgMAE"] = float(np.mean(mae)) if mae else 0.0
    return metrics


def _normalise_direction(name: str, direction: Optional[str]) -> str:
    if direction:
        text = str(direction).lower()
        if text in {"maximize", "max", "maximise"}:
            return "maximize"
        if text in {"minimize", "min", "minimise"}:
            return "minimize"
    if name.lower() in {"maxdd", "maxdrawdown"}:
        return "minimize"
    return "maximize"


def _objective_iterator(
    objectives: Iterable[object],
) -> Iterable[Tuple[str, float, str]]:
    for obj in objectives:
        if isinstance(obj, str):
            name = obj
            weight = 1.0
            direction = _normalise_direction(name, None)
            yield name, weight, direction
        elif isinstance(obj, dict):
            name = obj.get("name") or obj.get("metric")
            if not name:
                continue
            weight = float(obj.get("weight", 1.0))
            direction = _normalise_direction(name, obj.get("direction") or obj.get("goal"))
            yield str(name), weight, direction


def score_metrics(metrics: Dict[str, float], objectives: Iterable[object]) -> float:
    """Score a metric dictionary according to weighted objectives and penalties."""

    score = 0.0
    for name, weight, direction in _objective_iterator(objectives):
        value = metrics.get(name)
        if value is None:
            continue
        contribution = float(value)
        if direction == "minimize":
            contribution = -abs(contribution)
        score += weight * contribution

    trades = float(metrics.get("Trades", 0))
    min_trades = metrics.get("MinTrades")
    if min_trades is not None and trades < float(min_trades):
        penalty = float(metrics.get("TradePenalty", 1.0))
        score -= (float(min_trades) - trades) * penalty

    avg_hold = float(metrics.get("AvgHoldBars", 0.0))
    min_hold = metrics.get("MinHoldBars")
    if min_hold is not None and avg_hold < float(min_hold):
        penalty = float(metrics.get("HoldPenalty", 1.0))
        score -= (float(min_hold) - avg_hold) * penalty

    losses = float(metrics.get("MaxConsecutiveLosses", 0.0))
    loss_cap = metrics.get("MaxConsecutiveLossLimit")
    if loss_cap is not None and losses > float(loss_cap):
        penalty = float(metrics.get("ConsecutiveLossPenalty", 1.0))
        score -= (losses - float(loss_cap)) * penalty

    return float(score)


__all__ = [
    "Trade",
    "aggregate_metrics",
    "equity_curve_from_returns",
    "max_drawdown",
    "score_metrics",
]
