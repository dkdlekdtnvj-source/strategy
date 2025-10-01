"""백테스트 메트릭 계산."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    profit_factor: float
    sortino: float
    net_profit: float
    max_dd: float
    win_rate: float
    trades: int
    avg_trade: float
    exposure: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "profit_factor": self.profit_factor,
            "sortino": self.sortino,
            "net_profit": self.net_profit,
            "max_dd": self.max_dd,
            "win_rate": self.win_rate,
            "trades": self.trades,
            "avg_trade": self.avg_trade,
            "exposure": self.exposure,
        }


def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / abs(losses)


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return excess.mean() / downside_std


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return dd.min() * 100


def compute_metrics(trade_pnl: pd.Series, equity_curve: pd.Series, exposure: float) -> Metrics:
    pf = profit_factor(trade_pnl)
    sortino = sortino_ratio(trade_pnl)
    net_profit = trade_pnl.sum()
    max_dd_pct = abs(max_drawdown(equity_curve))
    trades = trade_pnl.count()
    wins = (trade_pnl > 0).sum()
    win_rate = wins / trades * 100 if trades > 0 else 0.0
    avg_trade = trade_pnl.mean() if trades > 0 else 0.0
    return Metrics(
        profit_factor=float(pf),
        sortino=float(sortino),
        net_profit=float(net_profit),
        max_dd=float(max_dd_pct),
        win_rate=float(win_rate),
        trades=int(trades),
        avg_trade=float(avg_trade),
        exposure=float(exposure),
    )


def objective_score(metrics: Metrics, weights: Dict[str, float]) -> float:
    def normalize(value: float, clip: float = 5.0) -> float:
        value = min(value, clip)
        return value / clip

    pf_norm = normalize(metrics.profit_factor)
    sortino_norm = normalize(metrics.sortino)
    net_norm = np.tanh(metrics.net_profit / 1000)

    return (
        weights.get("pf", 0.6) * pf_norm
        + weights.get("sortino", 0.3) * sortino_norm
        + weights.get("netprofit", 0.1) * net_norm
    )


__all__ = [
    "Metrics",
    "compute_metrics",
    "max_drawdown",
    "objective_score",
    "profit_factor",
    "sortino_ratio",
]
