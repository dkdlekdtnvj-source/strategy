"""Walk-forward analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .strategy_model import run_backtest


def _clean_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool)):
            clean[key] = float(value)
    return clean


@dataclass
class SegmentResult:
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def run_walk_forward(
    df: pd.DataFrame,
    params: Dict[str, float | bool],
    fees: Dict[str, float],
    risk: Dict[str, float],
    train_bars: int,
    test_bars: int,
    step: int,
    htf_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    segments: List[SegmentResult] = []
    total = len(df)
    start = 0

    while start + train_bars + test_bars <= total:
        train_slice = slice(start, start + train_bars)
        test_slice = slice(start + train_bars, start + train_bars + test_bars)
        train_df = df.iloc[train_slice]
        test_df = df.iloc[test_slice]

        if train_df.empty or test_df.empty:
            break

        train_htf = htf_df.loc[: train_df.index[-1]] if htf_df is not None else None
        test_htf = htf_df.loc[: test_df.index[-1]] if htf_df is not None else None

        train_metrics = run_backtest(train_df, params, fees, risk, htf_df=train_htf)
        test_metrics = run_backtest(test_df, params, fees, risk, htf_df=test_htf)

        segments.append(
            SegmentResult(
                train_metrics=_clean_metrics(train_metrics),
                test_metrics=_clean_metrics(test_metrics),
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
            )
        )
        start += step

    oos_returns = [seg.test_metrics.get("NetProfit", 0.0) for seg in segments if seg.test_metrics.get("Valid", True)]
    oos_series = pd.Series(oos_returns) if oos_returns else pd.Series(dtype=float)
    summary = {
        "segments": segments,
        "oos_mean": float(oos_series.mean()) if not oos_series.empty else 0.0,
        "oos_median": float(oos_series.median()) if not oos_series.empty else 0.0,
        "count": len(segments),
    }
    return summary
