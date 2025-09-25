"""Command line interface for running parameter optimisation."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import optuna
import pandas as pd
import yaml

from datafeed.cache import DataCache
from optimize.metrics import score_metrics
from optimize.report import generate_reports
from optimize.search_spaces import build_space, grid_choices, sample_parameters
from optimize.strategy_model import run_backtest
from optimize.wf import run_walk_forward

LOGGER = logging.getLogger("optimize")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class DatasetSpec:
    symbol: str
    timeframe: str
    start: str
    end: str
    df: pd.DataFrame
    htf: Optional[pd.DataFrame]

    @property
    def name(self) -> str:
        return f"{self.symbol}_{self.timeframe}_{self.start}_{self.end}"

    @property
    def meta(self) -> Dict[str, str]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "from": self.start,
            "to": self.end,
        }


def prepare_datasets(
    params_cfg: Dict[str, object],
    backtest_cfg: Dict[str, object],
    data_dir: Path,
) -> List[DatasetSpec]:
    cache = DataCache(data_dir, futures=bool(backtest_cfg.get("futures", False)))

    base_symbol = str(params_cfg.get("symbol"))
    base_timeframe = str(params_cfg.get("timeframe"))
    base_period = params_cfg.get("backtest", {}) or {}

    symbols = backtest_cfg.get("symbols") or ([base_symbol] if base_symbol else [])
    timeframes = backtest_cfg.get("timeframes") or ([base_timeframe] if base_timeframe else [])
    periods = backtest_cfg.get("periods") or ([base_period] if base_period else [])

    if not symbols or not timeframes or not periods:
        raise ValueError("Backtest configuration must specify symbol, timeframe, and period ranges")

    htf_timeframe = params_cfg.get("htf_timeframe") or backtest_cfg.get("htf_timeframe")

    datasets: List[DatasetSpec] = []
    for symbol, timeframe, period in product(symbols, timeframes, periods):
        start = str(period["from"])
        end = str(period["to"])
        LOGGER.info("Preparing dataset %s %s %s→%s", symbol, timeframe, start, end)
        df = cache.get(symbol, timeframe, start, end)
        htf = (
            cache.get(symbol, str(htf_timeframe), start, end, allow_partial=True)
            if htf_timeframe
            else None
        )
        datasets.append(DatasetSpec(symbol=symbol, timeframe=timeframe, start=start, end=end, df=df, htf=htf))
    return datasets


def _nanmean(values: Iterable[float]) -> float:
    arr = [float(v) for v in values if v is not None]
    return float(np.nanmean(arr)) if arr else 0.0


def combine_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}

    total_trades = float(sum(m.get("Trades", 0) for m in metric_list))
    total_wins = float(sum(m.get("Wins", 0) for m in metric_list))
    total_losses = float(sum(m.get("Losses", 0) for m in metric_list))
    gross_profit = float(sum(m.get("GrossProfit", 0.0) for m in metric_list))
    gross_loss = float(sum(m.get("GrossLoss", 0.0) for m in metric_list))

    combined: Dict[str, float] = {
        "Trades": total_trades,
        "Wins": total_wins,
        "Losses": total_losses,
        "GrossProfit": gross_profit,
        "GrossLoss": gross_loss,
        "NetProfit": _nanmean([m.get("NetProfit", 0.0) for m in metric_list]),
        "TotalReturn": float(sum(m.get("NetProfit", 0.0) for m in metric_list)),
        "MaxDD": min(m.get("MaxDD", 0.0) for m in metric_list),
        "Sortino": _nanmean([m.get("Sortino", 0.0) for m in metric_list]),
        "Sharpe": _nanmean([m.get("Sharpe", 0.0) for m in metric_list]),
        "AvgRR": _nanmean([m.get("AvgRR", 0.0) for m in metric_list]),
        "AvgHoldBars": _nanmean([m.get("AvgHoldBars", 0.0) for m in metric_list]),
        "AvgMFE": _nanmean([m.get("AvgMFE", 0.0) for m in metric_list]),
        "AvgMAE": _nanmean([m.get("AvgMAE", 0.0) for m in metric_list]),
        "WeeklyNetProfit": _nanmean([m.get("WeeklyNetProfit", 0.0) for m in metric_list]),
        "WeeklyReturnStd": _nanmean([m.get("WeeklyReturnStd", 0.0) for m in metric_list]),
        "ProfitFactor": (gross_profit / abs(gross_loss)) if gross_loss else float("inf"),
        "WinRate": (total_wins / total_trades) if total_trades else 0.0,
        "MaxConsecutiveLosses": max(m.get("MaxConsecutiveLosses", 0.0) for m in metric_list),
        "Expectancy": _nanmean([m.get("Expectancy", 0.0) for m in metric_list]),
    }

    base = metric_list[0]
    for key in [
        "MinTrades",
        "MinHoldBars",
        "MaxConsecutiveLossLimit",
        "TradePenalty",
        "HoldPenalty",
        "ConsecutiveLossPenalty",
    ]:
        if key in base:
            combined[key] = float(base[key])

    combined["Valid"] = all(m.get("Valid", True) for m in metric_list)
    return combined


def _clean_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    clean: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool, str)):
            clean[key] = value
    return clean


def optimisation_loop(
    datasets: List[DatasetSpec],
    params_cfg: Dict[str, object],
    objectives: Iterable[object],
    fees: Dict[str, float],
    risk: Dict[str, float],
) -> Dict[str, object]:
    search_cfg = params_cfg.get("search", {})
    space = build_space(params_cfg.get("space", {}))

    algo = search_cfg.get("algo", "bayes")
    seed = search_cfg.get("seed")
    n_trials = int(search_cfg.get("n_trials", 50))

    if algo == "grid":
        sampler = optuna.samplers.GridSampler(grid_choices(space))
    elif algo == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    results: List[Dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        params = sample_parameters(trial, space)
        dataset_metrics: List[Dict[str, object]] = []
        numeric_metrics: List[Dict[str, float]] = []

        for dataset in datasets:
            metrics = run_backtest(dataset.df, params, fees, risk, htf_df=dataset.htf)
            numeric_metrics.append(metrics)
            dataset_metrics.append(
                {
                    "name": dataset.name,
                    "meta": dataset.meta,
                    "metrics": _clean_metrics(metrics),
                }
            )

        aggregated = combine_metrics(numeric_metrics)
        score = score_metrics(aggregated, objectives)

        record = {
            "trial": trial.number,
            "params": params,
            "metrics": _clean_metrics(aggregated),
            "datasets": dataset_metrics,
            "score": score,
            "valid": aggregated.get("Valid", True),
        }
        results.append(record)
        return score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial.number
    best_record = next(res for res in results if res["trial"] == best_trial)
    return {"study": study, "results": results, "best": best_record}


def merge_dicts(primary: Dict[str, float], secondary: Dict[str, float]) -> Dict[str, float]:
    merged = dict(primary)
    merged.update({k: v for k, v in secondary.items() if v is not None})
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pine strategy optimisation")
    parser.add_argument("--params", type=Path, default=Path("config/params.yaml"))
    parser.add_argument("--backtest", type=Path, default=Path("config/backtest.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    params_cfg = load_yaml(args.params)
    backtest_cfg = load_yaml(args.backtest)

    datasets = prepare_datasets(params_cfg, backtest_cfg, args.data)
    if not datasets:
        raise RuntimeError("No datasets prepared for optimisation")

    fees = merge_dicts(params_cfg.get("fees", {}), backtest_cfg.get("fees", {}))
    risk = merge_dicts(params_cfg.get("risk", {}), backtest_cfg.get("risk", {}))

    objectives = params_cfg.get("objectives", [])
    optimisation = optimisation_loop(datasets, params_cfg, objectives, fees, risk)

    walk_cfg = params_cfg.get("walk_forward", {"train_bars": 5000, "test_bars": 2000, "step": 2000})
    primary_dataset = datasets[0]
    wf_summary = run_walk_forward(
        primary_dataset.df,
        optimisation["best"]["params"],
        fees,
        risk,
        train_bars=int(walk_cfg.get("train_bars", 5000)),
        test_bars=int(walk_cfg.get("test_bars", 2000)),
        step=int(walk_cfg.get("step", 2000)),
        htf_df=primary_dataset.htf,
    )

    generate_reports(
        optimisation["results"],
        optimisation["best"],
        wf_summary,
        objectives,
        args.output,
    )


if __name__ == "__main__":
    main()
