"""Command line interface for running parameter optimisation."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
import yaml

from datafeed.cache import DataCache
from optimize.context import (
    RegimeBandit,
    compute_regime_features,
    label_regime,
    summarise_regimes,
)
from optimize.metrics import score_metrics
from optimize.pbt import generate_pbt_mutations
from optimize.report import generate_reports
from optimize.search_spaces import (
    build_space,
    grid_choices,
    mutate_parameters,
    sample_parameters,
)
from optimize.strategy_model import run_backtest
from optimize.wf import run_purged_kfold, run_walk_forward

LOGGER = logging.getLogger("optimize")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _prompt_choice(label: str, choices: List[str], default: Optional[str] = None) -> Optional[str]:
    if not choices:
        return default
    while True:
        print(f"\n{label}:")
        for idx, value in enumerate(choices, start=1):
            marker = " (default)" if default == value else ""
            print(f"  {idx}. {value}{marker}")
        raw = input("Select option (press Enter for default): ").strip()
        if not raw:
            return default or (choices[0] if choices else None)
        if raw.isdigit():
            sel = int(raw)
            if 1 <= sel <= len(choices):
                return choices[sel - 1]
        print("Invalid selection. Please try again.")


def _prompt_bool(label: str, default: Optional[bool] = None) -> Optional[bool]:
    suffix = " [y/n]" if default is None else (" [Y/n]" if default else " [y/N]")
    while True:
        raw = input(f"{label}{suffix}: ").strip().lower()
        if not raw and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        if not raw:
            return default
        print("Please answer 'y' or 'n'.")


def _collect_tokens(items: Iterable[str]) -> List[str]:
    tokens: List[str] = []
    for item in items:
        if not item:
            continue
        for token in item.split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def _ensure_dict(root: Dict[str, object], key: str) -> Dict[str, object]:
    value = root.get(key)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


def _sanitize_slug(value: Optional[str], fallback: str) -> str:
    text = str(value or fallback)
    slug = "".join(ch for ch in text if ch.isalnum() or ch in {"-", "_"})
    return slug or fallback


def _ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True)
        return path
    if not any(path.iterdir()):
        return path
    base = path
    suffix = 1
    while True:
        candidate = base.parent / f"{base.name}_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def _resolve_output_dir(base: Optional[Path], run_tag: str) -> Path:
    root = base or Path("reports")
    root.mkdir(parents=True, exist_ok=True)
    target = root / run_tag
    return _ensure_unique_dir(target)


def _configure_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)


def _space_hash(space: Dict[str, object]) -> str:
    payload = json.dumps(space, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonify(data: object) -> object:
    try:
        json.dumps(data)
        return data
    except TypeError:
        if isinstance(data, dict):
            return {key: _jsonify(value) for key, value in data.items()}
        if isinstance(data, list):
            return [_jsonify(item) for item in data]
        if isinstance(data, Path):
            return str(data)
        return str(data)


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def _load_bank(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except json.JSONDecodeError:
        LOGGER.warning("Failed to decode bank file at %s", path)
        return {}


def _find_latest_bank(base: Path, symbol_slug: str, timeframe_slug: str, htf_slug: str) -> Optional[Path]:
    if not base.exists():
        return None
    pattern = f"*{symbol_slug}*{timeframe_slug}*{htf_slug}*"
    candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        bank_path = candidate / "bank.json"
        if bank_path.exists():
            return bank_path
    return None


def _unique_params(param_sets: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: set[str] = set()
    unique: List[Dict[str, object]] = []
    for params in param_sets:
        key = json.dumps(params, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(params)
    return unique


def _build_pruner(name: str, args: Dict[str, object]) -> optuna.pruners.BasePruner:
    key = (name or "asha").lower()
    if key in {"none", "nop", "disabled"}:
        return optuna.pruners.NopPruner()
    if key in {"none", "off", "disable"}:
        return optuna.pruners.NopPruner()
    if key in {"median", "medianpruner"}:
        return optuna.pruners.MedianPruner(**{k: v for k, v in args.items() if k in {"n_startup_trials", "interval_steps"}})
    if key in {"hyperband", "hb"}:
        return optuna.pruners.HyperbandPruner(**{k: v for k, v in args.items() if k in {"min_resource", "max_resource", "reduction_factor"}})
    if key in {"threshold", "th"}:
        return optuna.pruners.ThresholdPruner(**{k: v for k, v in args.items() if k in {"lower", "upper", "warmup_steps"}})
    if key in {"patient", "patience"}:
        return optuna.pruners.PatientPruner(optuna.pruners.NopPruner(), **{k: v for k, v in args.items() if k in {"patience", "min_delta"}})
    if key in {"wilcoxon", "wilcoxonpruner"}:
        return optuna.pruners.WilcoxonPruner(**{k: v for k, v in args.items() if k in {"alpha", "interval_steps"}})
    # Default to ASHA / Successive Halving.
    return optuna.pruners.SuccessiveHalvingPruner(
        **{k: v for k, v in args.items() if k in {"min_resource", "reduction_factor", "min_early_stopping_rate"}}
    )


def _current_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


@dataclass
class DatasetSpec:
    symbol: str
    timeframe: str
    start: str
    end: str
    df: pd.DataFrame
    htf: Optional[pd.DataFrame]
    htf_timeframe: Optional[str] = None
    source_symbol: Optional[str] = None

    @property
    def name(self) -> str:
        parts = [self.symbol, self.timeframe]
        if self.htf_timeframe:
            parts.append(f"htf{self.htf_timeframe}")
        parts.extend([self.start, self.end])
        return "_".join(parts)

    @property
    def meta(self) -> Dict[str, str]:
        return {
            "symbol": self.symbol,
            "source_symbol": self.source_symbol or self.symbol,
            "timeframe": self.timeframe,
            "from": self.start,
            "to": self.end,
            "htf_timeframe": self.htf_timeframe or "",
        }


def _normalise_timeframe_value(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_htf_value(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "na", "off", "0"}:
        return None
    return text


def _group_datasets(
    datasets: Sequence[DatasetSpec],
) -> Tuple[Dict[Tuple[str, Optional[str]], List[DatasetSpec]], Dict[str, List[DatasetSpec]], Tuple[str, Optional[str]]]:
    groups: Dict[Tuple[str, Optional[str]], List[DatasetSpec]] = {}
    timeframe_groups: Dict[str, List[DatasetSpec]] = {}
    for dataset in datasets:
        key = (dataset.timeframe, dataset.htf_timeframe or None)
        groups.setdefault(key, []).append(dataset)
        timeframe_groups.setdefault(dataset.timeframe, []).append(dataset)

    if not groups:
        raise RuntimeError("No datasets available for optimisation")

    default_key = next(iter(groups))
    return groups, timeframe_groups, default_key


def _select_datasets_for_params(
    params_cfg: Dict[str, object],
    dataset_groups: Dict[Tuple[str, Optional[str]], List[DatasetSpec]],
    timeframe_groups: Dict[str, List[DatasetSpec]],
    default_key: Tuple[str, Optional[str]],
    params: Dict[str, object],
) -> Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]:
    def _match(tf: str, htf: Optional[str]) -> Optional[Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]]:
        tf_lower = tf.lower()
        htf_lower = (htf or "").lower()
        for key, group in dataset_groups.items():
            key_tf, key_htf = key
            if key_tf.lower() != tf_lower:
                continue
            key_htf_lower = (key_htf or "").lower()
            if key_htf_lower == htf_lower:
                return key, group
        return None

    timeframe_value = (
        _normalise_timeframe_value(params.get("timeframe"))
        or _normalise_timeframe_value(params.get("ltf"))
        or _normalise_timeframe_value(params_cfg.get("timeframe"))
    )

    htf_value = (
        _normalise_htf_value(params.get("htf"))
        or _normalise_htf_value(params.get("htf_timeframe"))
    )

    if htf_value is None:
        cfg_htf = params_cfg.get("htf_timeframe")
        if cfg_htf:
            htf_value = _normalise_htf_value(cfg_htf)
        elif isinstance(params_cfg.get("htf_timeframes"), list) and len(params_cfg["htf_timeframes"]) == 1:
            htf_value = _normalise_htf_value(params_cfg["htf_timeframes"][0])

    if timeframe_value is None:
        timeframe_value = default_key[0]

    selected = None
    if timeframe_value:
        selected = _match(timeframe_value, htf_value)
        if selected is None and htf_value is not None:
            selected = _match(timeframe_value, None)
        if selected is None:
            for key, group in dataset_groups.items():
                if key[0].lower() == timeframe_value.lower():
                    selected = (key, group)
                    break

        if selected is None:
            for tf, group in timeframe_groups.items():
                if tf.lower() == timeframe_value.lower():
                    key = (group[0].timeframe, group[0].htf_timeframe or None)
                    selected = (key, group)
                    break

    if selected is None:
        selected = (default_key, dataset_groups[default_key])

    return selected


def _pick_primary_dataset(datasets: Sequence[DatasetSpec]) -> DatasetSpec:
    return max(datasets, key=lambda item: len(item.df))


def _resolve_symbol_entry(entry: object, alias_map: Dict[str, str]) -> Tuple[str, str]:
    """Normalise a symbol entry to a display name and a Binance fetch symbol."""

    if isinstance(entry, dict):
        alias = entry.get("alias") or entry.get("name") or entry.get("symbol") or entry.get("id") or ""
        resolved = entry.get("symbol") or entry.get("id") or alias
        alias = str(alias) if alias else str(resolved)
        resolved = str(resolved) if resolved else alias
    else:
        alias = str(entry)
        resolved = alias

    resolved = alias_map.get(alias, alias_map.get(resolved, resolved))
    if not alias:
        alias = resolved
    if not resolved:
        resolved = alias
    return alias, resolved


def _normalise_periods(
    periods_cfg: Optional[Iterable[Dict[str, object]]],
    base_period: Dict[str, object],
) -> List[Dict[str, str]]:
    periods: List[Dict[str, str]] = []
    if periods_cfg:
        for idx, raw in enumerate(periods_cfg):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Period entry #{idx + 1} must be a mapping with 'from'/'to' keys, got {type(raw).__name__}."
                )
            start = raw.get("from")
            end = raw.get("to")
            if not start or not end:
                raise ValueError(
                    f"Period entry #{idx + 1} is missing required 'from'/'to' values: {raw}."
                )
            periods.append({"from": str(start), "to": str(end)})

    if not periods:
        start = base_period.get("from") if isinstance(base_period, dict) else None
        end = base_period.get("to") if isinstance(base_period, dict) else None
        if start and end:
            periods.append({"from": str(start), "to": str(end)})

    return periods


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
    periods = _normalise_periods(backtest_cfg.get("periods"), base_period)

    if not symbols or not timeframes or not periods:
        raise ValueError(
            "Backtest configuration must specify symbol(s), timeframe(s), and at least one period with 'from'/'to' dates."
        )

    alias_map: Dict[str, str] = {}
    for source in (backtest_cfg.get("symbol_aliases"), params_cfg.get("symbol_aliases")):
        if isinstance(source, dict):
            for key, value in source.items():
                if key and value:
                    alias_map[str(key)] = str(value)

    # Allow either a single HTF timeframe or a list so that users can compare 15m vs 1h, etc.
    def _to_list(value: Optional[object]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if v]
        return [str(value)] if str(value) else []

    htf_candidates = _to_list(params_cfg.get("htf_timeframes"))
    if not htf_candidates:
        htf_candidates = _to_list(params_cfg.get("htf_timeframe"))
    if not htf_candidates:
        htf_candidates = _to_list(backtest_cfg.get("htf_timeframes"))
    if not htf_candidates:
        htf_candidates = _to_list(backtest_cfg.get("htf_timeframe"))
    if not htf_candidates:
        htf_candidates = [None]

    symbol_pairs = [_resolve_symbol_entry(symbol, alias_map) for symbol in symbols]

    datasets: List[DatasetSpec] = []
    for (display_symbol, source_symbol), timeframe, period, htf_tf in product(
        symbol_pairs, timeframes, periods, htf_candidates
    ):
        start = str(period["from"])
        end = str(period["to"])
        symbol_log = (
            display_symbol if display_symbol == source_symbol else f"{display_symbol}→{source_symbol}"
        )
        LOGGER.info(
            "Preparing dataset %s %s (HTF %s) %s→%s",
            symbol_log,
            timeframe,
            htf_tf or "-",
            start,
            end,
        )
        df = cache.get(source_symbol, timeframe, start, end)
        htf = (
            cache.get(source_symbol, str(htf_tf), start, end, allow_partial=True)
            if htf_tf
            else None
        )
        datasets.append(
            DatasetSpec(
                symbol=display_symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                df=df,
                htf=htf,
                htf_timeframe=str(htf_tf) if htf_tf else None,
                source_symbol=source_symbol,
            )
        )
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
    forced_params: Optional[Dict[str, object]] = None,
    space: Optional[Dict[str, object]] = None,
    study_storage: Optional[str] = None,
    study_name: Optional[str] = None,
    initial_params: Optional[Iterable[Dict[str, object]]] = None,
    pruner_name: Optional[str] = None,
    pruner_args: Optional[Dict[str, object]] = None,
    bandit: Optional[RegimeBandit] = None,
) -> Dict[str, object]:
    search_cfg = params_cfg.get("search", {})
    space_spec = space or build_space(params_cfg.get("space", {}))

    dataset_groups, timeframe_groups, default_key = _group_datasets(datasets)

    algo = str(search_cfg.get("algo", "bayes")).lower()
    seed = search_cfg.get("seed")
    n_trials = int(search_cfg.get("n_trials", 50))
    forced_params = forced_params or {}

    if algo == "grid":
        sampler = optuna.samplers.GridSampler(grid_choices(space_spec))
    elif algo == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)

    pruner_cfg = pruner_name or search_cfg.get("pruner", "asha")
    pruner_params: Dict[str, object] = dict(search_cfg.get("pruner_args", {}))
    if pruner_args:
        pruner_params.update(pruner_args)
    pruner = _build_pruner(str(pruner_cfg), pruner_params)

    study_kwargs: Dict[str, Any] = {"direction": "maximize", "sampler": sampler, "pruner": pruner}
    if study_storage:
        study_kwargs["storage"] = study_storage
        study_kwargs["load_if_exists"] = True
        if study_name:
            study_kwargs["study_name"] = study_name

    study = optuna.create_study(**study_kwargs)

    if initial_params:
        space_keys = set(space_spec.keys())
        for suggestion in _unique_params(initial_params):
            filtered = {key: value for key, value in suggestion.items() if key in space_keys}
            if filtered:
                study.enqueue_trial(filtered)

    def objective(trial: optuna.Trial) -> float:
        params = sample_parameters(trial, space_spec)
        params.update(forced_params)
        key, selected_datasets = _select_datasets_for_params(
            params_cfg, dataset_groups, timeframe_groups, default_key, params
        )
        if not selected_datasets:
            raise optuna.TrialPruned("No datasets matched trial parameters")

        dataset_metrics: List[Dict[str, object]] = []
        numeric_metrics: List[Dict[str, float]] = []

        prune_step = 0
        for dataset in selected_datasets:
            metrics = run_backtest(dataset.df, params, fees, risk, htf_df=dataset.htf)
            numeric_metrics.append(metrics)
            dataset_metrics.append(
                {
                    "name": dataset.name,
                    "meta": dataset.meta,
                    "metrics": _clean_metrics(metrics),
                }
            )
            prune_step += 1
            trial.report(float(metrics.get("NetProfit", 0.0)), step=prune_step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        aggregated = combine_metrics(numeric_metrics)
        score = score_metrics(aggregated, objectives)

        primary_dataset = _pick_primary_dataset(selected_datasets)
        features = compute_regime_features(primary_dataset.df)
        regime_label = label_regime(features)
        arm_id = hashlib.sha1(
            json.dumps(params, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]

        record_metrics = _clean_metrics(aggregated)
        record = {
            "trial": trial.number,
            "params": params,
            "metrics": record_metrics,
            "datasets": dataset_metrics,
            "score": score,
            "valid": aggregated.get("Valid", True),
            "dataset_key": {"timeframe": key[0], "htf_timeframe": key[1]},
            "regime": regime_label,
            "regime_features": features,
            "arm_id": arm_id,
        }

        if bandit is not None and regime_label:
            bandit.update(regime_label, arm_id, score)

        trial.set_user_attr("metrics", record["metrics"])
        trial.set_user_attr("datasets", record["datasets"])
        trial.set_user_attr("score", record["score"])
        trial.set_user_attr("valid", record["valid"])
        trial.set_user_attr("dataset_key", record["dataset_key"])
        trial.set_user_attr("regime", record["regime"])
        trial.set_user_attr("regime_features", record["regime_features"])
        trial.set_user_attr("arm_id", record["arm_id"])
        trial.set_user_attr("params", record["params"])

        trial.report(score, step=prune_step + 1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    results: List[Dict[str, object]] = []
    for trial in study.trials:
        attrs = trial.user_attrs or {}
        metrics = attrs.get("metrics")
        if not metrics:
            continue
        results.append(
            {
                "trial": trial.number,
                "params": dict(attrs.get("params", trial.params)),
                "metrics": attrs.get("metrics", {}),
                "datasets": attrs.get("datasets", []),
                "score": attrs.get("score", trial.value),
                "valid": attrs.get("valid", True),
                "dataset_key": attrs.get("dataset_key"),
                "regime": attrs.get("regime"),
                "regime_features": attrs.get("regime_features"),
                "arm_id": attrs.get("arm_id"),
                "pruned": trial.state == optuna.trial.TrialState.PRUNED,
            }
        )

    if not results:
        raise RuntimeError("No optimisation results available")

    best_trial = study.best_trial.number
    best_record = next((res for res in results if res["trial"] == best_trial), None)
    if best_record is None:
        trial = study.best_trial
        attrs = trial.user_attrs or {}
        best_record = {
            "trial": trial.number,
            "params": dict(attrs.get("params", trial.params)),
            "metrics": attrs.get("metrics", {}),
            "datasets": attrs.get("datasets", []),
            "score": attrs.get("score", trial.value),
            "valid": attrs.get("valid", True),
            "dataset_key": attrs.get("dataset_key"),
            "regime": attrs.get("regime"),
            "regime_features": attrs.get("regime_features"),
            "arm_id": attrs.get("arm_id"),
            "pruned": trial.state == optuna.trial.TrialState.PRUNED,
        }
        results.append(best_record)

    return {
        "study": study,
        "results": results,
        "best": best_record,
        "space": space_spec,
        "bandit": bandit,
    }


def merge_dicts(primary: Dict[str, float], secondary: Dict[str, float]) -> Dict[str, float]:
    merged = dict(primary)
    merged.update({k: v for k, v in secondary.items() if v is not None})
    return merged


def build_parser() -> argparse.ArgumentParser:
    """Build an :class:`argparse.ArgumentParser` for optimisation commands."""

    parser = argparse.ArgumentParser(description="Run Pine strategy optimisation")
    parser.add_argument("--params", type=Path, default=Path("config/params.yaml"))
    parser.add_argument("--backtest", type=Path, default=Path("config/backtest.yaml"))
    parser.add_argument("--output", type=Path, default=None, help="Base directory for reports (timestamped run folder will be created)")
    parser.add_argument("--run-tag", type=str, help="Custom run tag for output folder naming")
    parser.add_argument("--resume-from", type=Path, help="Path to a previous bank.json for warm-starting suggestions")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--symbol", type=str, help="Override symbol to optimise")
    parser.add_argument("--timeframe", type=str, help="Override lower timeframe")
    parser.add_argument("--htf", type=str, help="Override higher timeframe for confirmations")
    parser.add_argument("--start", type=str, help="Override backtest start date (ISO8601)")
    parser.add_argument("--end", type=str, help="Override backtest end date (ISO8601)")
    parser.add_argument("--leverage", type=float, help="Override leverage setting")
    parser.add_argument("--qty-pct", type=float, help="Override quantity percent")
    parser.add_argument("--interactive", action="store_true", help="Prompt for dataset and toggle selections")
    parser.add_argument("--enable", action="append", default=[], help="Force-enable boolean parameters (comma separated)")
    parser.add_argument("--disable", action="append", default=[], help="Force-disable boolean parameters (comma separated)")
    parser.add_argument("--top-k", type=int, default=0, help="Re-rank top-K trials by walk-forward OOS mean")
    parser.add_argument("--n-trials", type=int, help="Override Optuna trial count")
    parser.add_argument("--pruner", type=str, help="Override pruner strategy (asha, hyperband, median, threshold, patient, wilcoxon, none)")
    parser.add_argument("--cv", type=str, help="Additional validation scheme (e.g. purged-kfold)")
    parser.add_argument("--cv-k", type=int, help="Number of folds for purged-kfold")
    parser.add_argument("--cv-embargo", type=float, help="Embargo ratio for purged-kfold")
    parser.add_argument("--pbt-epsilon", type=float, default=0.1, help="Exploration rate for PBT-style mutations")
    parser.add_argument("--study-dir", type=Path, default=Path("studies"), help="Directory to store Optuna study SQLite databases")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments and return a populated :class:`Namespace`."""

    return build_parser().parse_args(argv)



def execute(args: argparse.Namespace) -> None:
    """Execute an optimisation run using an :class:`argparse.Namespace`."""

    params_cfg = load_yaml(args.params)
    backtest_cfg = load_yaml(args.backtest)

    if args.n_trials is not None:
        search_cfg = _ensure_dict(params_cfg, "search")
        search_cfg["n_trials"] = int(args.n_trials)

    forced_params: Dict[str, object] = dict(params_cfg.get("overrides", {}))
    for name in _collect_tokens(args.enable):
        forced_params[name] = True
    for name in _collect_tokens(args.disable):
        forced_params[name] = False

    symbol_choices = list(
        dict.fromkeys(
            backtest_cfg.get("symbols")
            or ([params_cfg.get("symbol")] if params_cfg.get("symbol") else [])
        )
    )

    def _collect_htfs(cfg: Dict[str, object]) -> List[str]:
        values: List[str] = []
        raw = cfg.get("htf_timeframes")
        if isinstance(raw, (list, tuple)):
            values.extend(str(item) for item in raw if item)
        single = cfg.get("htf_timeframe")
        if single:
            values.append(str(single))
        return values

    htf_choices = list(dict.fromkeys(_collect_htfs(backtest_cfg) or _collect_htfs(params_cfg)))

    selected_symbol = (
        args.symbol or params_cfg.get("symbol") or (symbol_choices[0] if symbol_choices else None)
    )
    selected_timeframe = args.timeframe
    selected_htf: Optional[str] = args.htf if args.htf else None
    timeframe_overridden = args.timeframe is not None
    htf_overridden = args.htf is not None

    if args.interactive and symbol_choices:
        selected_symbol = _prompt_choice("Select symbol", symbol_choices, selected_symbol)

    if selected_symbol:
        params_cfg["symbol"] = selected_symbol
        backtest_cfg["symbols"] = [selected_symbol]
    if timeframe_overridden and selected_timeframe:
        params_cfg["timeframe"] = selected_timeframe
        backtest_cfg["timeframes"] = [selected_timeframe]
    if htf_overridden and selected_htf is not None:
        params_cfg["htf_timeframes"] = [selected_htf]
        backtest_cfg["htf_timeframes"] = [selected_htf]
    elif htf_choices:
        params_cfg["htf_timeframes"] = htf_choices
        backtest_cfg["htf_timeframes"] = htf_choices
    params_cfg.pop("htf_timeframe", None)
    backtest_cfg.pop("htf_timeframe", None)

    backtest_periods = backtest_cfg.get("periods") or []
    params_backtest = _ensure_dict(params_cfg, "backtest")
    if args.start or args.end:
        start = (
            args.start
            or params_backtest.get("from")
            or (backtest_periods[0]["from"] if backtest_periods else None)
        )
        end = (
            args.end
            or params_backtest.get("to")
            or (backtest_periods[0]["to"] if backtest_periods else None)
        )
        if start and end:
            params_backtest["from"] = start
            params_backtest["to"] = end
            backtest_cfg["periods"] = [{"from": start, "to": end}]
    elif args.interactive and backtest_periods:
        labels = [f"{p['from']} → {p['to']}" for p in backtest_periods]
        default_label = (
            f"{params_backtest.get('from')} → {params_backtest.get('to')}"
            if params_backtest.get("from") and params_backtest.get("to")
            else labels[0]
        )
        choice = _prompt_choice("Select backtest period", labels, default_label)
        if choice and choice in labels:
            selected = dict(backtest_periods[labels.index(choice)])
            params_backtest["from"] = selected["from"]
            params_backtest["to"] = selected["to"]
            backtest_cfg["periods"] = [selected]
    elif params_backtest.get("from") and params_backtest.get("to"):
        backtest_cfg["periods"] = [
            {"from": params_backtest["from"], "to": params_backtest["to"]}
        ]

    risk_cfg = _ensure_dict(params_cfg, "risk")
    backtest_risk = _ensure_dict(backtest_cfg, "risk")
    if args.leverage is not None:
        risk_cfg["leverage"] = args.leverage
        backtest_risk["leverage"] = args.leverage
    if args.qty_pct is not None:
        risk_cfg["qty_pct"] = args.qty_pct
        backtest_risk["qty_pct"] = args.qty_pct

    if args.interactive:
        leverage_default = risk_cfg.get("leverage")
        qty_default = risk_cfg.get("qty_pct")
        lev_input = input(f"Leverage [{leverage_default}]: ").strip()
        if lev_input:
            try:
                leverage_val = float(lev_input)
                risk_cfg["leverage"] = leverage_val
                backtest_risk["leverage"] = leverage_val
            except ValueError:
                print("Invalid leverage value, keeping previous setting.")
        qty_input = input(f"Position size % [{qty_default}]: ").strip()
        if qty_input:
            try:
                qty_val = float(qty_input)
                risk_cfg["qty_pct"] = qty_val
                backtest_risk["qty_pct"] = qty_val
            except ValueError:
                print("Invalid quantity percentage, keeping previous setting.")

        bool_candidates = [
            name for name, spec in params_cfg.get("space", {}).items() if spec.get("type") == "bool"
        ]
        for name in bool_candidates:
            default_val = forced_params.get(name)
            decision = _prompt_bool(f"Enable {name}", default_val)
            if decision is not None:
                forced_params[name] = decision

    datasets = prepare_datasets(params_cfg, backtest_cfg, args.data)
    if not datasets:
        raise RuntimeError("No datasets prepared for optimisation")

    fees = merge_dicts(params_cfg.get("fees", {}), backtest_cfg.get("fees", {}))
    risk = merge_dicts(params_cfg.get("risk", {}), backtest_cfg.get("risk", {}))
    objectives = params_cfg.get("objectives", [])

    space_cfg = params_cfg.get("space", {})
    space_spec = build_space(space_cfg)
    space_hash = _space_hash(space_cfg)

    now = datetime.utcnow()
    symbol_slug = _sanitize_slug(params_cfg.get("symbol"), "symbol")
    tf_candidate = params_cfg.get("timeframe") or (
        datasets[0].timeframe if datasets else "ltf"
    )
    timeframe_slug = _sanitize_slug(tf_candidate, "ltf")
    htf_values = sorted({dataset.htf_timeframe or "" for dataset in datasets})
    if len(htf_values) == 1:
        htf_slug_raw = htf_values[0] or "nohtf"
    else:
        htf_slug_raw = "multihtf"
    htf_slug = _sanitize_slug(htf_slug_raw, "nohtf")

    custom_tag = _sanitize_slug(args.run_tag, "run") if args.run_tag else None
    run_tag = f"{now:%Y%m%d-%H%M}_{symbol_slug}_{timeframe_slug}_{htf_slug}"
    if custom_tag and custom_tag != "run":
        run_tag = f"{run_tag}_{custom_tag}"

    output_dir = _resolve_output_dir(args.output, run_tag)
    _configure_logging(output_dir)
    LOGGER.info("Output directory: %s", output_dir)

    reports_root = output_dir.parent
    study_dir = args.study_dir
    study_dir.mkdir(parents=True, exist_ok=True)
    study_slug = f"{symbol_slug}_{timeframe_slug}_{htf_slug}"
    study_path = study_dir / f"{study_slug}.db"
    study_storage = f"sqlite:///{study_path.resolve()}"
    study_name = study_slug

    resume_path = args.resume_from or _find_latest_bank(
        reports_root, symbol_slug, timeframe_slug, htf_slug
    )
    resume_source = str(resume_path) if resume_path else None
    if resume_path:
        LOGGER.info("Warm-starting from bank file %s", resume_path)

    bank_data = _load_bank(resume_path)
    if bank_data and bank_data.get("space_hash") not in {None, space_hash}:
        LOGGER.info("Discarding warm-start bank due to space mismatch")
        bank_data = {}

    bandit = RegimeBandit.from_dict(bank_data.get("bandit")) if bank_data else RegimeBandit()

    primary_dataset = _pick_primary_dataset(datasets)
    regime_features = compute_regime_features(primary_dataset.df)
    current_regime = label_regime(regime_features)
    LOGGER.info("Current regime: %s", current_regime)

    search_seed = params_cfg.get("search", {}).get("seed")
    rng = np.random.default_rng(search_seed)

    initial_seeds: List[Dict[str, object]] = []
    bank_entries = bank_data.get("entries", []) if bank_data else []
    sorted_entries = sorted(
        (
            entry
            for entry in bank_entries
            if isinstance(entry.get("params"), dict)
        ),
        key=lambda entry: (
            entry.get("oos_mean") if entry.get("oos_mean") is not None else float("-inf"),
            entry.get("score") if entry.get("score") is not None else float("-inf"),
        ),
        reverse=True,
    )

    selected_arm = bandit.select(current_regime)
    if selected_arm:
        for entry in sorted_entries:
            if entry.get("arm_id") == selected_arm and isinstance(entry.get("params"), dict):
                initial_seeds.append(dict(entry["params"]))
                break

    for entry in sorted_entries[:10]:
        params = entry.get("params")
        if isinstance(params, dict):
            initial_seeds.append(dict(params))
            initial_seeds.append(
                mutate_parameters(params, space_spec, scale=0.15, rng=rng)
            )

    for params in bank_data.get("mutations", []) if bank_data else []:
        if isinstance(params, dict):
            initial_seeds.append(dict(params))

    initial_params = _unique_params(initial_seeds)
    if len(initial_params) > 32:
        initial_params = initial_params[:32]

    optimisation = optimisation_loop(
        datasets,
        params_cfg,
        objectives,
        fees,
        risk,
        forced_params,
        space=space_spec,
        study_storage=study_storage,
        study_name=study_name,
        initial_params=initial_params,
        pruner_name=args.pruner,
        pruner_args={},
        bandit=bandit,
    )

    study = optimisation["study"]
    results = optimisation["results"]
    best_record = optimisation["best"]
    bandit = optimisation.get("bandit", bandit)

    walk_cfg = params_cfg.get(
        "walk_forward", {"train_bars": 5000, "test_bars": 2000, "step": 2000}
    )
    dataset_groups, timeframe_groups, default_key = _group_datasets(datasets)

    def _resolve_record_dataset(record: Dict[str, object]) -> Tuple[
        Tuple[str, Optional[str]], List[DatasetSpec]
    ]:
        key_info = record.get("dataset_key") if isinstance(record, dict) else None
        if isinstance(key_info, dict):
            candidate_key = (key_info.get("timeframe"), key_info.get("htf_timeframe"))
            if candidate_key in dataset_groups:
                return candidate_key, dataset_groups[candidate_key]
        return _select_datasets_for_params(
            params_cfg,
            dataset_groups,
            timeframe_groups,
            default_key,
            record.get("params", {}),
        )

    best_key, best_group = _resolve_record_dataset(best_record)
    primary_dataset = _pick_primary_dataset(best_group)

    wf_summary = run_walk_forward(
        primary_dataset.df,
        best_record["params"],
        fees,
        risk,
        train_bars=int(walk_cfg.get("train_bars", 5000)),
        test_bars=int(walk_cfg.get("test_bars", 2000)),
        step=int(walk_cfg.get("step", 2000)),
        htf_df=primary_dataset.htf,
    )

    candidate_summaries = [
        {
            "trial": best_record["trial"],
            "score": best_record.get("score"),
            "oos_mean": wf_summary.get("oos_mean"),
            "params": best_record.get("params"),
            "timeframe": best_key[0],
            "htf_timeframe": best_key[1],
        }
    ]

    top_k = args.top_k or int(params_cfg.get("search", {}).get("top_k", 0))
    if top_k > 0:
        sorted_results = sorted(
            results,
            key=lambda r: r.get("score", 0.0),
            reverse=True,
        )
        best_oos = wf_summary.get("oos_mean", float("-inf"))
        for record in sorted_results[:top_k]:
            if record["trial"] == best_record["trial"]:
                continue
            candidate_key, candidate_group = _resolve_record_dataset(record)
            candidate_dataset = _pick_primary_dataset(candidate_group)
            candidate_wf = run_walk_forward(
                candidate_dataset.df,
                record["params"],
                fees,
                risk,
                train_bars=int(walk_cfg.get("train_bars", 5000)),
                test_bars=int(walk_cfg.get("test_bars", 2000)),
                step=int(walk_cfg.get("step", 2000)),
                htf_df=candidate_dataset.htf,
            )
            candidate_summaries.append(
                {
                    "trial": record["trial"],
                    "score": record.get("score"),
                    "oos_mean": candidate_wf.get("oos_mean"),
                    "params": record.get("params"),
                    "timeframe": candidate_key[0],
                    "htf_timeframe": candidate_key[1],
                }
            )
            candidate_oos = candidate_wf.get("oos_mean", float("-inf"))
            if candidate_oos > best_oos:
                best_oos = candidate_oos
                best_record = record
                best_key = candidate_key
                best_group = candidate_group
                primary_dataset = candidate_dataset
                wf_summary = candidate_wf
        optimisation["best"] = best_record

    candidate_summaries[0] = {
        "trial": best_record["trial"],
        "score": best_record.get("score"),
        "oos_mean": wf_summary.get("oos_mean"),
        "params": best_record.get("params"),
        "timeframe": best_key[0],
        "htf_timeframe": best_key[1],
    }

    wf_summary["candidates"] = candidate_summaries

    cv_summary: Optional[Dict[str, object]] = None
    if args.cv and args.cv.lower() in {"purged-kfold", "purged_kfold", "purgedkfold"}:
        cv_k = args.cv_k or 5
        cv_embargo = args.cv_embargo if args.cv_embargo is not None else 0.01
        cv_summary = run_purged_kfold(
            primary_dataset.df,
            best_record["params"],
            fees,
            risk,
            k=int(cv_k),
            embargo=float(cv_embargo),
            htf_df=primary_dataset.htf,
        )

    generate_reports(
        results,
        best_record,
        wf_summary,
        objectives,
        output_dir,
        cv_summary=cv_summary,
    )

    try:
        trials_df = study.trials_dataframe()
        if not trials_df.empty:
            trials_df.to_csv(output_dir / "trials.csv", index=False)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to export trials dataframe: %s", exc)

    record_map = {record["trial"]: record for record in results}
    bank_entries_payload: List[Dict[str, object]] = []
    for summary in candidate_summaries:
        entry = {
            "trial": summary.get("trial"),
            "score": summary.get("score"),
            "oos_mean": summary.get("oos_mean"),
            "params": summary.get("params"),
            "timeframe": summary.get("timeframe"),
            "htf_timeframe": summary.get("htf_timeframe"),
        }
        record = record_map.get(summary.get("trial"))
        if record:
            entry["metrics"] = record.get("metrics")
            entry["regime"] = record.get("regime")
            entry["regime_features"] = record.get("regime_features")
            entry["arm_id"] = record.get("arm_id")
        bank_entries_payload.append(entry)

    seed_params = [
        entry.get("params")
        for entry in bank_entries_payload
        if isinstance(entry.get("params"), dict)
    ]
    pbt_mutations = generate_pbt_mutations(
        seed_params[:5],
        optimisation.get("space", space_spec),
        scale=0.1,
        epsilon=float(args.pbt_epsilon),
        rng=rng,
    )

    bank_payload = {
        "timestamp": now.isoformat(),
        "run_tag": run_tag,
        "symbol": params_cfg.get("symbol"),
        "timeframe": tf_candidate,
        "htf": htf_slug_raw,
        "space_hash": space_hash,
        "entries": bank_entries_payload,
        "bandit": bandit.to_dict(),
        "resume_source": resume_source,
        "mutations": pbt_mutations,
        "current_regime": current_regime,
    }
    (output_dir / "bank.json").write_text(json.dumps(_jsonify(bank_payload), indent=2))

    manifest = {
        "timestamp": now.isoformat(),
        "run_tag": run_tag,
        "git_commit": _current_commit(),
        "output_dir": str(output_dir),
        "params_file": str(args.params),
        "backtest_file": str(args.backtest),
        "symbol": params_cfg.get("symbol"),
        "timeframe": tf_candidate,
        "htf": htf_slug_raw,
        "objectives": objectives,
        "space_hash": space_hash,
        "fees": fees,
        "risk": risk,
        "resume_from": resume_source,
        "study_storage": study_storage,
        "study_name": study_name,
        "initial_seed_count": len(initial_params),
        "results": {
            "trial_count": len(results),
            "best_trial": best_record.get("trial"),
            "best_score": best_record.get("score"),
        },
        "walk_forward": {
            "oos_mean": wf_summary.get("oos_mean"),
            "oos_median": wf_summary.get("oos_median"),
            "segments": wf_summary.get("count"),
        },
        "cv": cv_summary,
        "current_regime": current_regime,
        "regime_features": regime_features,
        "bandit_summary": summarise_regimes(bandit),
        "datasets": [dataset.meta for dataset in datasets],
        "forced_params": forced_params,
    }
    _write_manifest(output_dir / "manifest.json", _jsonify(manifest))

    LOGGER.info("Optimisation complete. Best trial %s (score %.4f)", best_record.get("trial"), best_record.get("score"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for ``python -m optimize.run``."""

    args = parse_args(argv)
    execute(args)


if __name__ == "__main__":
    main()
