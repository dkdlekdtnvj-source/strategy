"""Command line interface for running parameter optimisation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import logging
import os
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
import optuna.storages
import pandas as pd
import yaml

from datafeed.cache import DataCache
from optimize.metrics import Trade, aggregate_metrics, score_metrics
from optimize.report import generate_reports, write_bank_file, write_trials_dataframe
from optimize.search_spaces import build_space, grid_choices, mutate_around, sample_parameters
from optimize.strategy_model import DefaultStrategy
from optimize.strategies.base import StrategyModel
from optimize.wf import run_purged_kfold, run_walk_forward
from optimize.regime import detect_regime_label, summarise_regime_performance
from optimize.llm import generate_llm_candidates

LOGGER = logging.getLogger("optimize")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = Path("reports")
STUDY_ROOT = Path("studies")
NON_FINITE_PENALTY = -1e12
HIGHLIGHT_METRICS = ["NetProfit", "ProfitFactor", "Sortino", "MaxDD", "Trades"]


def _slugify_symbol(symbol: str) -> str:
    text = symbol.split(":")[-1]
    return text.replace("/", "").replace(" ", "")


def _space_hash(space: Dict[str, object]) -> str:
    payload = json.dumps(space or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.parent / f"{path.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def _configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(handler)


def _load_strategy(params_cfg: Dict[str, object]) -> StrategyModel:
    strategy_cfg = params_cfg.get("strategy") if isinstance(params_cfg.get("strategy"), dict) else {}
    module_name = str(strategy_cfg.get("module") or "optimize.strategy_model")
    class_name = str(strategy_cfg.get("class") or "DefaultStrategy")
    factory_name = strategy_cfg.get("factory")

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"전략 모듈을 불러오지 못했습니다: {module_name}") from exc

    if factory_name:
        factory = getattr(module, factory_name, None)
        if factory is None or not callable(factory):
            raise RuntimeError(f"전략 팩토리 {factory_name} 를 찾을 수 없습니다")
        strategy = factory(params_cfg)
    else:
        strategy_obj = getattr(module, class_name, None)
        if strategy_obj is None:
            raise RuntimeError(f"전략 클래스 {class_name} 를 찾을 수 없습니다")
        if inspect.isclass(strategy_obj) and issubclass(strategy_obj, StrategyModel):
            if callable(getattr(strategy_obj, "from_config", None)):
                strategy = strategy_obj.from_config(params_cfg)  # type: ignore[misc]
            else:
                strategy = strategy_obj()
        else:
            if callable(strategy_obj):
                strategy = strategy_obj(params_cfg)
            else:
                raise RuntimeError(f"{class_name} 는 StrategyModel을 구현하지 않습니다")

    if not isinstance(strategy, StrategyModel):
        raise TypeError("전략 객체가 StrategyModel을 상속하지 않습니다")
    return strategy


def _build_run_tag(
    datasets: Sequence["DatasetSpec"],
    params_cfg: Dict[str, object],
    run_tag: Optional[str],
) -> Tuple[str, str, str, str]:
    symbol = params_cfg.get("symbol") or (datasets[0].symbol if datasets else "unknown")
    timeframe = (
        params_cfg.get("timeframe")
        or (datasets[0].timeframe if datasets else "multi")
    )
    htf = (
        params_cfg.get("htf_timeframe")
        or params_cfg.get("htf")
        or (datasets[0].htf_timeframe if datasets and datasets[0].htf_timeframe else "nohtf")
    )
    if not htf:
        htf = "nohtf"
    symbol_slug = _slugify_symbol(str(symbol))
    timeframe_slug = str(timeframe).replace("/", "_")
    htf_slug = str(htf).replace("/", "_")
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
    parts = [timestamp, symbol_slug, timeframe_slug, htf_slug]
    if run_tag:
        parts.append(run_tag)
    return timestamp, symbol_slug, timeframe_slug, "_".join(filter(None, parts))


def _resolve_output_directory(
    base: Optional[Path],
    datasets: Sequence["DatasetSpec"],
    params_cfg: Dict[str, object],
    run_tag: Optional[str],
) -> Tuple[Path, Dict[str, str]]:
    ts, symbol_slug, timeframe_slug, tag = _build_run_tag(datasets, params_cfg, run_tag)
    if base is None:
        root = DEFAULT_REPORT_ROOT
        output = root / tag
    else:
        output = base
    output = _next_available_dir(output)
    output.mkdir(parents=True, exist_ok=False)
    return output, {
        "timestamp": ts,
        "symbol": symbol_slug,
        "timeframe": timeframe_slug,
        "tag": tag,
    }


def _write_manifest(
    output_dir: Path,
    *,
    manifest: Dict[str, object],
) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _load_json(path: Path) -> Dict[str, object]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _resolve_study_storage(
    params_cfg: Dict[str, object],
    datasets: Sequence["DatasetSpec"],
) -> Optional[Path]:
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    _, symbol_slug, timeframe_slug, _ = _build_run_tag(datasets, params_cfg, None)
    htf = params_cfg.get("htf_timeframe") or (
        datasets[0].htf_timeframe if datasets and datasets[0].htf_timeframe else "nohtf"
    )
    htf_slug = str(htf or "nohtf").replace("/", "_")
    return STUDY_ROOT / f"{symbol_slug}_{timeframe_slug}_{htf_slug}.db"


def _discover_bank_path(
    current_output: Path,
    tag_info: Dict[str, str],
    space_hash: str,
) -> Optional[Path]:
    root = current_output.parent
    if not root.exists():
        return None
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p != current_output],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        bank_path = candidate / "bank.json"
        if not bank_path.exists():
            continue
        payload = _load_json(bank_path)
        metadata = payload.get("metadata", {})
        if payload.get("space_hash") != space_hash:
            continue
        if metadata.get("symbol") != tag_info.get("symbol"):
            continue
        if metadata.get("timeframe") != tag_info.get("timeframe"):
            continue
        return bank_path
    return None


def _load_seed_trials(
    bank_path: Optional[Path],
    space: Dict[str, object],
    space_hash: str,
    regime_label: Optional[str] = None,
    max_seeds: int = 20,
) -> List[Dict[str, object]]:
    if bank_path is None:
        return []
    payload = _load_json(bank_path)
    if not payload or payload.get("space_hash") != space_hash:
        return []

    entries = payload.get("entries", [])
    if regime_label:
        filtered = [entry for entry in entries if entry.get("regime", {}).get("label") == regime_label]
        if filtered:
            entries = filtered

    seeds: List[Dict[str, object]] = []
    rng = np.random.default_rng()
    for entry in entries[:max_seeds]:
        params = entry.get("params")
        if not isinstance(params, dict):
            continue
        seeds.append(dict(params))
        mutated = mutate_around(params, space, scale=float(payload.get("mutation_scale", 0.1)), rng=rng)
        seeds.append(mutated)
    return seeds


def _build_bank_payload(
    *,
    tag_info: Dict[str, str],
    space_hash: str,
    entries: List[Dict[str, object]],
    regime_summary,
    mutation_scale: float = 0.1,
) -> Dict[str, object]:
    payload_entries: List[Dict[str, object]] = []
    for entry in entries:
        regime_info = summarise_regime_performance(entry, regime_summary)
        payload_entries.append({**entry, "regime": regime_info})

    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "metadata": {
            "symbol": tag_info.get("symbol"),
            "timeframe": tag_info.get("timeframe"),
            "tag": tag_info.get("tag"),
        },
        "space_hash": space_hash,
        "mutation_scale": mutation_scale,
        "entries": payload_entries,
    }


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


def _parse_objectives(objectives: Iterable[object]) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for obj in objectives:
        if isinstance(obj, str):
            name = obj
            weight = 1.0
            direction = "minimize" if name.lower() in {"maxdd", "maxdrawdown"} else "maximize"
        elif isinstance(obj, dict):
            name = obj.get("name") or obj.get("metric")
            if not name:
                continue
            weight = float(obj.get("weight", 1.0))
            raw_direction = obj.get("direction") or obj.get("goal")
            direction = str(raw_direction).lower() if raw_direction else None
            if direction in {"min", "minimise", "minimize"}:
                direction = "minimize"
            elif direction in {"max", "maximise", "maximize"}:
                direction = "maximize"
            else:
                direction = "minimize" if name.lower() in {"maxdd", "maxdrawdown"} else "maximize"
        else:
            continue
        specs.append({"name": str(name), "weight": float(weight), "direction": direction})
    return specs


def combine_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {}

    weight_map: Dict[int, float] = {}
    weighted_returns: Optional[pd.Series] = None
    return_sources = [
        (idx, metrics.get("Returns"))
        for idx, metrics in enumerate(metric_list)
        if isinstance(metrics.get("Returns"), pd.Series)
    ]

    if return_sources:
        weight = 1.0 / len(return_sources)
        for idx, series in return_sources:
            assert isinstance(series, pd.Series)
            weight_map[idx] = weight
            weighted_component = series.fillna(0.0) * weight
            weighted_returns = (
                weighted_component
                if weighted_returns is None
                else weighted_returns.add(weighted_component, fill_value=0.0)
            )
    else:
        fallback_weight = 1.0 / len(metric_list)
        for idx in range(len(metric_list)):
            weight_map[idx] = fallback_weight

    if weighted_returns is None:
        weighted_returns = pd.Series(dtype=float)

    trades_pool: List[Trade] = []
    for idx, metrics in enumerate(metric_list):
        trades = metrics.get("TradesList")
        weight = weight_map.get(idx, 0.0)
        if trades:
            for trade in trades:
                scaled = replace(
                    trade,
                    size=trade.size * weight,
                    profit=trade.profit * weight,
                    return_pct=trade.return_pct * weight,
                    mfe=trade.mfe * weight,
                    mae=trade.mae * weight,
                )
                trades_pool.append(scaled)

    aggregated = aggregate_metrics(trades_pool, weighted_returns)
    aggregated["Returns"] = weighted_returns
    aggregated["TradesList"] = trades_pool
    aggregated["Valid"] = all(m.get("Valid", True) for m in metric_list)

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
            aggregated[key] = float(base[key])

    aggregated["Trades"] = int(aggregated.get("Trades", 0))
    aggregated["Wins"] = int(aggregated.get("Wins", 0))
    aggregated["Losses"] = int(aggregated.get("Losses", 0))
    return aggregated


def _clean_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    clean: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool, str)):
            clean[key] = value
    return clean


def _create_pruner(name: str, params: Dict[str, object]) -> optuna.pruners.BasePruner:
    name = (name or "asha").lower()
    params = params or {}
    if name in {"none", "nop", "off"}:
        return optuna.pruners.NopPruner()
    if name in {"median", "medianpruner"}:
        return optuna.pruners.MedianPruner(**params)
    if name in {"hyperband"}:
        return optuna.pruners.HyperbandPruner(**params)
    if name in {"threshold", "thresholdpruner"}:
        return optuna.pruners.ThresholdPruner(**params)
    if name in {"patient", "patientpruner"}:
        patience = int(params.get("patience", 10))
        wrapped = _create_pruner(params.get("wrapped", "nop"), params.get("wrapped_params", {}))
        return optuna.pruners.PatientPruner(wrapped, patience=patience)
    if name in {"wilcoxon", "wilcoxonpruner"}:
        return optuna.pruners.WilcoxonPruner(**params)
    # Default to ASHA / successive halving
    return optuna.pruners.SuccessiveHalvingPruner(**params)


def optimisation_loop(
    datasets: List[DatasetSpec],
    params_cfg: Dict[str, object],
    objectives: Iterable[object],
    fees: Dict[str, float],
    risk: Dict[str, float],
    strategy: StrategyModel,
    forced_params: Optional[Dict[str, object]] = None,
    *,
    study_storage: Optional[Path] = None,
    space_hash: Optional[str] = None,
    seed_trials: Optional[List[Dict[str, object]]] = None,
    log_dir: Optional[Path] = None,
) -> Dict[str, object]:
    search_cfg = params_cfg.get("search", {})
    objectives = list(objectives)
    objective_specs = _parse_objectives(objectives)
    multi_objective_flag = search_cfg.get("multi_objective")
    if isinstance(multi_objective_flag, bool):
        multi_objective = multi_objective_flag
    else:
        mode = str(search_cfg.get("objective_mode", "single")).lower()
        multi_objective = mode in {"multi", "pareto"}
    multi_objective = bool(multi_objective and objective_specs)

    validation_cfg = params_cfg.get("validation", {}) if isinstance(params_cfg.get("validation"), dict) else {}
    inline_cfg = validation_cfg.get("in_objective", {}) if isinstance(validation_cfg.get("in_objective"), dict) else {}
    inline_enabled = bool(inline_cfg.get("enabled"))
    inline_frequency = max(int(inline_cfg.get("frequency", 1)), 1)
    inline_weight = float(inline_cfg.get("weight", 0.0))
    inline_train = int(inline_cfg.get("train_bars", 0))
    inline_test = int(inline_cfg.get("test_bars", 0))
    inline_step = int(inline_cfg.get("step", inline_test or 1))

    space = build_space(params_cfg.get("space", {}))

    dataset_groups, timeframe_groups, default_key = _group_datasets(datasets)

    algo = search_cfg.get("algo", "bayes")
    seed = search_cfg.get("seed")
    n_trials = int(search_cfg.get("n_trials", 50))
    forced_params = forced_params or {}
    log_dir_path: Optional[Path] = Path(log_dir) if log_dir else None
    trial_log_path: Optional[Path] = None
    best_yaml_path: Optional[Path] = None
    live_csv_path: Optional[Path] = None
    summary_csv_path: Optional[Path] = None
    if log_dir_path:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        trial_log_path = log_dir_path / "trials.jsonl"
        best_yaml_path = log_dir_path / "best.yaml"
        live_csv_path = log_dir_path / "trials_live.csv"
        summary_csv_path = log_dir_path / "trials_final.csv"
        for candidate in (trial_log_path, best_yaml_path, live_csv_path, summary_csv_path):
            if candidate.exists():
                candidate.unlink()
    non_finite_penalty = float(search_cfg.get("non_finite_penalty", NON_FINITE_PENALTY))
    cpu_count = os.cpu_count() or 1
    raw_n_jobs = search_cfg.get("n_jobs", "auto")
    if isinstance(raw_n_jobs, str) and raw_n_jobs.lower() == "auto":
        n_jobs = cpu_count
    else:
        try:
            n_jobs = int(raw_n_jobs)
        except Exception:
            n_jobs = cpu_count
    if n_jobs <= 0:
        n_jobs = cpu_count

    llm_cfg = params_cfg.get("llm", {}) if isinstance(params_cfg.get("llm"), dict) else {}

    def _normalise_metric_direction(name: str, direction: Optional[str]) -> str:
        if direction:
            lowered = str(direction).lower()
            if lowered in {"maximize", "max", "maximise"}:
                return "maximize"
            if lowered in {"minimize", "min", "minimise"}:
                return "minimize"
        lowered_name = str(name).lower()
        if lowered_name in {"maxdd", "maxdrawdown", "max_drawdown", "drawdown"}:
            return "minimize"
        return "maximize"

    best_metric_name = str(search_cfg.get("best_metric") or "ProfitFactor")
    best_metric_direction = _normalise_metric_direction(
        best_metric_name, search_cfg.get("best_metric_direction")
    )

    refine_cfg = search_cfg.get("refine", {}) if isinstance(search_cfg.get("refine"), dict) else {}
    refine_enabled = bool(refine_cfg.get("enabled"))
    refine_interval = max(int(refine_cfg.get("interval", 10)), 1)
    refine_warmup = max(int(refine_cfg.get("warmup", refine_interval)), 1)
    refine_top = max(int(refine_cfg.get("top", 3)), 1)
    refine_mutations = max(int(refine_cfg.get("mutations", 2)), 1)
    refine_scale = float(refine_cfg.get("scale", 0.15))
    refine_rng = np.random.default_rng(seed)

    if algo == "grid":
        sampler = optuna.samplers.GridSampler(grid_choices(space))
    elif algo == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif algo in {"cmaes", "cma-es", "cma"}:
        sampler = optuna.samplers.CmaEsSampler(seed=seed, consider_pruned_trials=True)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)

    if n_jobs > 1 and isinstance(sampler, optuna.samplers.CmaEsSampler):
        LOGGER.warning("CMA-ES 샘플러는 병렬 실행과 궁합이 좋지 않을 수 있습니다 (n_jobs=%s)", n_jobs)

    pruner_cfg = str(search_cfg.get("pruner", "asha"))
    pruner_params = search_cfg.get("pruner_params", {})
    pruner = _create_pruner(pruner_cfg, pruner_params or {})

    storage_url = None
    if study_storage is not None:
        study_storage.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{study_storage}"

    study_name = search_cfg.get("study_name") or (space_hash[:12] if space_hash else None)

    storage: Optional[optuna.storages.RDBStorage]
    storage = None
    if storage_url:
        heartbeat_interval = max(int(search_cfg.get("heartbeat_interval", 60)), 0)
        heartbeat_grace = max(int(search_cfg.get("heartbeat_grace_period", 120)), 0)
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"check_same_thread": False}},
            heartbeat_interval=heartbeat_interval or None,
            grace_period=heartbeat_grace or None,
        )
    storage_arg = storage if storage is not None else storage_url

    if multi_objective:
        directions = [spec["direction"] for spec in objective_specs] or ["maximize"]
        study = optuna.create_study(
            directions=directions,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage_arg,
            load_if_exists=bool(storage_arg),
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage_arg,
            load_if_exists=bool(storage_arg),
        )
    if space_hash:
        study.set_user_attr("space_hash", space_hash)

    results: List[Dict[str, object]] = []
    results_lock = Lock()
    log_lock = Lock()
    enqueue_lock = Lock()
    refine_rng_lock = Lock()
    best_state = {
        "rank": float("-inf"),
        "metric_value": None,
        "score": float("-inf"),
        "trial": None,
        "params": None,
    }
    best_state_lock = Lock()

    for params in seed_trials or []:
        trial_params = dict(params)
        trial_params.update(forced_params)
        try:
            with enqueue_lock:
                study.enqueue_trial(trial_params, skip_if_exists=True)
        except Exception:
            continue

    def _to_native(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial_log_path is None:
            return

        values: List[float] = []
        if trial.values is not None:
            for val in trial.values:
                try:
                    values.append(float(val))
                except Exception:
                    continue

        record = {
            "number": trial.number,
            "values": values if len(values) > 1 else (values[0] if values else None),
            "state": str(trial.state),
            "params": {key: _to_native(val) for key, val in trial.params.items()},
            "datetime_complete": str(trial.datetime_complete) if trial.datetime_complete else None,
        }

        highlights = {
            key: trial.user_attrs[key]
            for key in HIGHLIGHT_METRICS
            if key in trial.user_attrs
        }
        if highlights:
            record["metrics"] = highlights

        with log_lock:
            with trial_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_lock = Lock()
    csv_state = {"header_written": False}

    def _append_csv_row(record: Dict[str, object]) -> None:
        if live_csv_path is None:
            return
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        row = {
            "trial": record.get("trial"),
            "score": record.get("score"),
            "primary_metric": metrics.get(best_metric_name),
            "primary_metric_name": best_metric_name,
            "profit_factor": metrics.get("ProfitFactor"),
            "sortino": metrics.get("Sortino"),
            "max_dd": metrics.get("MaxDD"),
            "net_profit": metrics.get("NetProfit"),
            "trades": metrics.get("Trades"),
            "wf_oos": (record.get("inline_validation") or {}).get("oos_mean")
            if isinstance(record.get("inline_validation"), dict)
            else None,
            "params_json": json.dumps(record.get("params", {}), ensure_ascii=False),
        }
        fieldnames = list(row.keys())
        with csv_lock:
            need_header = not csv_state["header_written"] or not live_csv_path.exists()
            with live_csv_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if need_header:
                    writer.writeheader()
                    csv_state["header_written"] = True
                writer.writerow(row)

    callbacks: List = []
    if trial_log_path is not None:
        callbacks.append(_log_trial)

    def _rank_metric_value(value: Optional[float]) -> float:
        if value is None:
            return float("-inf")
        try:
            numeric = float(value)
        except Exception:
            return float("-inf")
        if np.isnan(numeric):
            return float("-inf")
        if np.isinf(numeric):
            if best_metric_direction == "maximize" and numeric > 0:
                return float("inf")
            return float("-inf")
        if not np.isfinite(numeric):
            return float("-inf")
        if best_metric_direction == "minimize":
            return -abs(numeric)
        return numeric

    def _update_best_snapshot(
        record: Dict[str, object], score_value: float, metric_value: Optional[float]
    ) -> None:
        if not np.isfinite(score_value):
            return
        rank_value = _rank_metric_value(metric_value)
        if rank_value == float("-inf") and metric_value is None:
            return
        with best_state_lock:
            if rank_value < best_state["rank"]:
                return
            if (
                np.isfinite(best_state["rank"])
                and np.isclose(rank_value, best_state["rank"])
                and score_value <= best_state["score"]
            ):
                return
            best_state["rank"] = rank_value
            best_state["metric_value"] = metric_value if metric_value is not None else None
            best_state["score"] = score_value
            best_state["trial"] = record.get("trial")
            best_state["params"] = record.get("params")
            if best_yaml_path is None:
                return
            params_payload = {
                key: _to_native(val)
                for key, val in (record.get("params") or {}).items()
            }
            snapshot = {
                "best_metric": best_metric_name,
                "best_metric_direction": best_metric_direction,
                "best_metric_value": (
                    float(metric_value)
                    if metric_value is not None and np.isfinite(float(metric_value))
                    else None
                ),
                "best_score": float(score_value),
                "best_params": params_payload,
                "trial": record.get("trial"),
            }
            with best_yaml_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)

    def objective(trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        params = sample_parameters(trial, space)
        params.update(forced_params)
        key, selected_datasets = _select_datasets_for_params(
            params_cfg, dataset_groups, timeframe_groups, default_key, params
        )
        dataset_metrics: List[Dict[str, object]] = []
        numeric_metrics: List[Dict[str, float]] = []

        def _sanitise(value: float, stage: str) -> float:
            try:
                numeric = float(value)
            except Exception:
                numeric = non_finite_penalty
            if not np.isfinite(numeric):
                LOGGER.warning(
                    "Non-finite %s score detected for trial %s; applying penalty %.0e",
                    stage,
                    trial.number,
                    non_finite_penalty,
                )
                return non_finite_penalty
            return numeric

        primary_prepared: Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]] = None

        for idx, dataset in enumerate(selected_datasets, start=1):
            prepared_df, prepared_htf = strategy.prepare_data(dataset.df, dataset.htf, params)
            if primary_prepared is None:
                primary_prepared = (prepared_df, prepared_htf)
            metrics = strategy.run_backtest(
                prepared_df,
                params,
                fees,
                risk,
                htf_df=prepared_htf,
                min_trades=risk.get("min_trades"),
            )
            numeric_metrics.append(metrics)
            dataset_metrics.append(
                {
                    "name": dataset.name,
                    "meta": dataset.meta,
                    "metrics": _clean_metrics(metrics),
                }
            )

            partial_metrics = combine_metrics(numeric_metrics)
            partial_score = score_metrics(partial_metrics, objectives)
            partial_score = _sanitise(partial_score, f"partial@{idx}")
            trial.report(partial_score, step=idx)
            if trial.should_prune():
                with results_lock:
                    results.append(
                        {
                            "trial": trial.number,
                            "params": params,
                            "metrics": _clean_metrics(partial_metrics),
                            "datasets": dataset_metrics,
                            "score": partial_score,
                            "valid": partial_metrics.get("Valid", True),
                            "dataset_key": {"timeframe": key[0], "htf_timeframe": key[1]},
                            "pruned": True,
                        }
                    )
                raise optuna.TrialPruned()

        aggregated = combine_metrics(numeric_metrics)
        score = score_metrics(aggregated, objectives)

        inline_summary: Optional[Dict[str, object]] = None
        if (
            inline_enabled
            and selected_datasets
            and inline_train > 0
            and inline_test > 0
            and trial.number % inline_frequency == 0
        ):
            base_dataset = selected_datasets[0]
            if primary_prepared is None:
                primary_prepared = strategy.prepare_data(base_dataset.df, base_dataset.htf, params)
            prepared_df, prepared_htf = primary_prepared
            inline_summary = run_walk_forward(
                prepared_df,
                params,
                fees,
                risk,
                train_bars=inline_train,
                test_bars=inline_test,
                step=inline_step,
                htf_df=prepared_htf,
                strategy=strategy,
            )
            oos_mean = float(inline_summary.get("oos_mean", 0.0))
            aggregated["WF_OOS"] = oos_mean
            if inline_weight:
                score = (1 - inline_weight) * score + inline_weight * oos_mean

        score = _sanitise(score, "final")

        for metric_name in HIGHLIGHT_METRICS:
            value = aggregated.get(metric_name)
            if value is None:
                continue
            try:
                trial.set_user_attr(metric_name, float(value))
            except Exception:
                continue

        raw_metric_value = aggregated.get(best_metric_name)
        try:
            metric_value = float(raw_metric_value) if raw_metric_value is not None else None
        except Exception:
            metric_value = None
        record = {
            "trial": trial.number,
            "params": params,
            "metrics": _clean_metrics(aggregated),
            "datasets": dataset_metrics,
            "score": score,
            "valid": aggregated.get("Valid", True),
            "dataset_key": {"timeframe": key[0], "htf_timeframe": key[1]},
            "pruned": False,
            "primary_metric": metric_value,
            "primary_metric_name": best_metric_name,
        }
        if inline_summary is not None:
            record["inline_validation"] = inline_summary

        if multi_objective:
            objective_values: List[float] = []
            for spec in objective_specs:
                direction = spec.get("direction", "maximize")
                value = aggregated.get(spec["name"])
                if value is None or not np.isfinite(float(value)):
                    penalty = (
                        abs(non_finite_penalty) if direction == "minimize" else non_finite_penalty
                    )
                    if penalty == 0:
                        penalty = 1e12 if direction == "minimize" else -1e12
                    objective_values.append(float(penalty))
                    continue
                numeric = float(value)
                if direction == "minimize":
                    numeric = abs(numeric)
                objective_values.append(numeric)
            record["optuna_values"] = objective_values
            with results_lock:
                results.append(record)
            _update_best_snapshot(record, score, metric_value)
            LOGGER.info(
                "Trial %s 완료 score=%.4f %s=%.3f PF=%.3f Sortino=%.3f DD=%.3f Trades=%s",
                trial.number,
                score,
                best_metric_name,
                float(metric_value) if metric_value is not None else float("nan"),
                aggregated.get("ProfitFactor"),
                aggregated.get("Sortino"),
                aggregated.get("MaxDD"),
                aggregated.get("Trades"),
            )
            _append_csv_row(record)
            return tuple(objective_values)

        with results_lock:
            results.append(record)

        _update_best_snapshot(record, score, metric_value)

        LOGGER.info(
            "Trial %s 완료 score=%.4f %s=%.3f PF=%.3f Sortino=%.3f DD=%.3f Trades=%s",
            trial.number,
            score,
            best_metric_name,
            float(metric_value) if metric_value is not None else float("nan"),
            aggregated.get("ProfitFactor"),
            aggregated.get("Sortino"),
            aggregated.get("MaxDD"),
            aggregated.get("Trades"),
        )

        _append_csv_row(record)

        if refine_enabled and (trial.number + 1) >= refine_warmup and (
            (trial.number + 1) % refine_interval == 0
        ):
            rng = refine_rng
            with results_lock:
                ranked = sorted(
                    results,
                    key=lambda r: _rank_metric_value(r.get("primary_metric")),
                    reverse=True,
                )
                top_candidates = [
                    r for r in ranked if r.get("params") and r.get("primary_metric") is not None
                ][:refine_top]
            if top_candidates:
                for candidate in top_candidates:
                    for _ in range(refine_mutations):
                        with refine_rng_lock:
                            mutated = mutate_around(
                                candidate["params"], space, scale=refine_scale, rng=rng
                            )
                        mutated.update(forced_params)
                        try:
                            with enqueue_lock:
                                trial.study.enqueue_trial(mutated, skip_if_exists=True)
                        except Exception as exc:
                            LOGGER.debug(
                                "Refine enqueue 실패 trial=%s params=%s 오류=%s",
                                candidate.get("trial"),
                                mutated,
                                exc,
                            )

        return score

    def _run_optuna(batch: int) -> None:
        if batch <= 0:
            return
        study.optimize(
            objective,
            n_trials=batch,
            show_progress_bar=False,
            callbacks=callbacks,
            gc_after_trial=True,
            n_jobs=n_jobs,
        )

    use_llm = bool(llm_cfg.get("enabled"))
    llm_count = int(llm_cfg.get("count", 0)) if use_llm else 0
    llm_initial = int(llm_cfg.get("initial_trials", max(10, n_trials // 2))) if use_llm else 0
    llm_initial = max(0, min(llm_initial, n_trials))

    try:
        if use_llm and llm_count > 0 and 0 < llm_initial < n_trials:
            _run_optuna(llm_initial)
            candidates = generate_llm_candidates(
                space, study.trials, llm_cfg, objective_specs
            )
            for candidate in candidates[:llm_count]:
                trial_params = dict(candidate)
                trial_params.update(forced_params)
                try:
                    with enqueue_lock:
                        study.enqueue_trial(trial_params, skip_if_exists=True)
                except Exception as exc:
                    LOGGER.debug("Failed to enqueue LLM candidate %s: %s", candidate, exc)
            remaining = n_trials - llm_initial
            _run_optuna(remaining)
        else:
            _run_optuna(n_trials)
    finally:
        if summary_csv_path is not None:
            try:
                df = study.trials_dataframe()
            except Exception:
                df = None
            if df is not None:
                df.to_csv(summary_csv_path, index=False)

    if not results:
        raise RuntimeError("No completed trials were produced during optimisation.")

    best_record: Optional[Dict[str, object]] = None
    if best_state["trial"] is not None:
        best_trial_num = int(best_state["trial"])
        best_record = next((res for res in results if res["trial"] == best_trial_num), None)
    if best_record is None:
        ranked_results = sorted(
            results,
            key=lambda r: _rank_metric_value(r.get("primary_metric")),
            reverse=True,
        )
        if ranked_results:
            top_rank = _rank_metric_value(ranked_results[0].get("primary_metric"))
            if top_rank != float("-inf"):
                best_record = ranked_results[0]
    if best_record is None:
        try:
            best_trial_num = study.best_trial.number
            best_record = next((res for res in results if res["trial"] == best_trial_num), None)
        except ValueError:
            best_record = None
    if best_record is None:
        best_record = max(results, key=lambda r: r.get("score", float("-inf")))
    if best_record is not None:
        best_record.setdefault("primary_metric_name", best_metric_name)
    return {
        "study": study,
        "results": results,
        "best": best_record,
        "best_metric": {"name": best_metric_name, "direction": best_metric_direction},
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
    parser.add_argument("--output", type=Path, help="Custom output directory (defaults to timestamped folder)")
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
    parser.add_argument("--run-tag", type=str, help="Additional suffix for the output directory name")
    parser.add_argument("--resume-from", type=Path, help="Path to a bank.json file for warm-start seeding")
    parser.add_argument("--pruner", type=str, help="Override pruner selection (asha, hyperband, median, threshold, patient, wilcoxon, none)")
    parser.add_argument("--cv", type=str, choices=["purged-kfold", "none"], help="Enable auxiliary cross-validation scoring")
    parser.add_argument("--cv-k", type=int, help="Number of folds for Purged K-Fold validation")
    parser.add_argument("--cv-embargo", type=float, help="Embargo fraction for Purged K-Fold validation")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments and return a populated :class:`Namespace`."""

    return build_parser().parse_args(argv)


def execute(args: argparse.Namespace, argv: Optional[Sequence[str]] = None) -> None:
    """Execute an optimisation run using an :class:`argparse.Namespace`."""

    params_cfg = load_yaml(args.params)
    backtest_cfg = load_yaml(args.backtest)

    if args.n_trials is not None:
        search_cfg = _ensure_dict(params_cfg, "search")
        search_cfg["n_trials"] = int(args.n_trials)

    if args.pruner:
        search_cfg = _ensure_dict(params_cfg, "search")
        search_cfg["pruner"] = args.pruner

    forced_params: Dict[str, object] = dict(params_cfg.get("overrides", {}))
    for name in _collect_tokens(args.enable):
        forced_params[name] = True
    for name in _collect_tokens(args.disable):
        forced_params[name] = False

    symbol_choices = list(dict.fromkeys(backtest_cfg.get("symbols") or ([params_cfg.get("symbol")] if params_cfg.get("symbol") else [])))

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

    selected_symbol = args.symbol or params_cfg.get("symbol") or (symbol_choices[0] if symbol_choices else None)
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
        start = args.start or params_backtest.get("from") or (backtest_periods[0]["from"] if backtest_periods else None)
        end = args.end or params_backtest.get("to") or (backtest_periods[0]["to"] if backtest_periods else None)
        if start and end:
            params_backtest["from"] = start
            params_backtest["to"] = end
            backtest_cfg["periods"] = [{"from": start, "to": end}]
    elif args.interactive and backtest_periods:
        labels = [f"{p['from']} → {p['to']}" for p in backtest_periods]
        default_label = f"{params_backtest.get('from')} → {params_backtest.get('to')}" if params_backtest.get("from") and params_backtest.get("to") else labels[0]
        choice = _prompt_choice("Select backtest period", labels, default_label)
        if choice and choice in labels:
            selected = dict(backtest_periods[labels.index(choice)])
            params_backtest["from"] = selected["from"]
            params_backtest["to"] = selected["to"]
            backtest_cfg["periods"] = [selected]
    elif params_backtest.get("from") and params_backtest.get("to"):
        backtest_cfg["periods"] = [{"from": params_backtest["from"], "to": params_backtest["to"]}]

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
        qty_input = input(f"Position size %% [{qty_default}]: ").strip()
        if qty_input:
            try:
                qty_val = float(qty_input)
                risk_cfg["qty_pct"] = qty_val
                backtest_risk["qty_pct"] = qty_val
            except ValueError:
                print("Invalid quantity percentage, keeping previous setting.")

        bool_candidates = [name for name, spec in params_cfg.get("space", {}).items() if spec.get("type") == "bool"]
        for name in bool_candidates:
            default_val = forced_params.get(name)
            decision = _prompt_bool(f"Enable {name}", default_val)
            if decision is not None:
                forced_params[name] = decision

    datasets = prepare_datasets(params_cfg, backtest_cfg, args.data)
    if not datasets:
        raise RuntimeError("No datasets prepared for optimisation")

    output_dir, tag_info = _resolve_output_directory(args.output, datasets, params_cfg, args.run_tag)
    _configure_logging(output_dir / "logs")
    LOGGER.info("Writing outputs to %s", output_dir)

    fees = merge_dicts(params_cfg.get("fees", {}), backtest_cfg.get("fees", {}))
    risk = merge_dicts(params_cfg.get("risk", {}), backtest_cfg.get("risk", {}))

    try:
        strategy_instance = _load_strategy(params_cfg)
    except Exception as exc:
        LOGGER.error("전략 로딩에 실패하여 기본 전략으로 대체합니다: %s", exc)
        strategy_instance = DefaultStrategy()
    else:
        LOGGER.info(
            "전략 모듈 %s.%s 를 사용합니다",
            strategy_instance.__class__.__module__,
            strategy_instance.__class__.__name__,
        )

    objectives_cfg = params_cfg.get("objectives", [])
    if isinstance(objectives_cfg, list):
        objectives: List[str] = list(objectives_cfg)
    elif isinstance(objectives_cfg, tuple):
        objectives = list(objectives_cfg)
    elif objectives_cfg:
        objectives = list(objectives_cfg)
    else:
        objectives = []
    space_hash = _space_hash(params_cfg.get("space", {}))
    primary_for_regime = _pick_primary_dataset(datasets)
    regime_summary = detect_regime_label(primary_for_regime.df)

    resume_bank_path = args.resume_from
    if resume_bank_path is None:
        resume_bank_path = _discover_bank_path(output_dir, tag_info, space_hash)

    seed_trials = _load_seed_trials(
        resume_bank_path,
        params_cfg.get("space", {}),
        space_hash,
        regime_label=regime_summary.label,
    )

    study_storage = _resolve_study_storage(params_cfg, datasets)

    trials_log_dir = output_dir / "trials"

    optimisation = optimisation_loop(
        datasets,
        params_cfg,
        objectives,
        fees,
        risk,
        strategy_instance,
        forced_params,
        study_storage=study_storage,
        space_hash=space_hash,
        seed_trials=seed_trials,
        log_dir=trials_log_dir,
    )

    study = optimisation.get("study")
    if study is not None:
        write_trials_dataframe(study, output_dir)
    else:
        LOGGER.warning("No Optuna study returned; skipping trials export")

    walk_cfg = params_cfg.get("walk_forward", {"train_bars": 5000, "test_bars": 2000, "step": 2000})
    dataset_groups, timeframe_groups, default_key = _group_datasets(datasets)

    def _resolve_record_dataset(record: Dict[str, object]) -> Tuple[Tuple[str, Optional[str]], List[DatasetSpec]]:
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

    best_record = optimisation["best"]
    if best_record is None:
        raise RuntimeError("최적화 결과에서 베스트 트라이얼을 찾을 수 없습니다.")
    best_metric_info = optimisation.get("best_metric") or {}
    best_metric_name = (
        best_metric_info.get("name")
        or best_record.get("primary_metric_name")
        or "ProfitFactor"
    )
    best_metric_direction = str(best_metric_info.get("direction") or "maximize").lower()
    if best_metric_direction not in {"maximize", "minimize"}:
        lowered = best_metric_name.lower()
        best_metric_direction = "minimize" if lowered in {"maxdd", "maxdrawdown", "drawdown"} else "maximize"

    def _rank_candidate(value: Optional[object]) -> float:
        if value is None:
            return float("-inf")
        try:
            numeric = float(value)
        except Exception:
            return float("-inf")
        if np.isnan(numeric):
            return float("-inf")
        if np.isinf(numeric):
            if best_metric_direction == "maximize" and numeric > 0:
                return float("inf")
            return float("-inf")
        if best_metric_direction == "minimize":
            return -abs(numeric)
        return numeric

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
        strategy=strategy_instance,
    )

    cv_summary = None
    cv_manifest: Dict[str, object] = {}
    cv_choice = (args.cv or str(params_cfg.get("validation", {}).get("type", ""))).lower()
    if cv_choice == "purged-kfold":
        cv_k = args.cv_k or int(params_cfg.get("validation", {}).get("k", 5))
        cv_embargo = args.cv_embargo or float(params_cfg.get("validation", {}).get("embargo", 0.01))
        cv_summary = run_purged_kfold(
            primary_dataset.df,
            best_record["params"],
            fees,
            risk,
            k=cv_k,
            embargo=cv_embargo,
            htf_df=primary_dataset.htf,
            strategy=strategy_instance,
        )
        wf_summary["purged_kfold"] = cv_summary
        cv_manifest = {"type": "purged-kfold", "k": cv_k, "embargo": cv_embargo}
    elif cv_choice and cv_choice != "none":
        cv_manifest = {"type": cv_choice}

    candidate_summaries = [
        {
            "trial": best_record["trial"],
            "score": best_record.get("score"),
            "oos_mean": wf_summary.get("oos_mean"),
            "params": best_record.get("params"),
            "timeframe": best_key[0],
            "htf_timeframe": best_key[1],
            "primary_metric": best_record.get("primary_metric"),
            "primary_metric_name": best_metric_name,
        }
    ]

    top_k = args.top_k or int(params_cfg.get("search", {}).get("top_k", 0))
    if top_k > 0:
        sorted_results = sorted(
            optimisation["results"],
            key=lambda r: (
                _rank_candidate(r.get("primary_metric")),
                r.get("score", float("-inf")),
            ),
            reverse=True,
        )
        best_oos = wf_summary.get("oos_mean", float("-inf"))
        wf_cache = {best_record["trial"]: wf_summary}
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
                strategy=strategy_instance,
            )
            wf_cache[record["trial"]] = candidate_wf
            candidate_summaries.append(
                {
                    "trial": record["trial"],
                    "score": record.get("score"),
                    "oos_mean": candidate_wf.get("oos_mean"),
                    "params": record.get("params"),
                    "timeframe": candidate_key[0],
                    "htf_timeframe": candidate_key[1],
                    "primary_metric": record.get("primary_metric"),
                    "primary_metric_name": best_metric_name,
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
        "primary_metric": best_record.get("primary_metric"),
        "primary_metric_name": best_metric_name,
    }

    wf_summary["candidates"] = candidate_summaries

    trial_index = {record["trial"]: record for record in optimisation["results"]}
    bank_entries: List[Dict[str, object]] = []
    for item in candidate_summaries:
        trial_record = trial_index.get(item["trial"], {})
        entry = {
            "trial": item["trial"],
            "score": item.get("score"),
            "oos_mean": item.get("oos_mean"),
            "params": item.get("params"),
            "metrics": trial_record.get("metrics"),
            "timeframe": item.get("timeframe"),
            "htf_timeframe": item.get("htf_timeframe"),
            "primary_metric": item.get("primary_metric"),
            "primary_metric_name": item.get("primary_metric_name", best_metric_name),
        }
        if cv_summary:
            entry["cv_mean"] = cv_summary.get("mean")
        bank_entries.append(entry)

    bank_payload = _build_bank_payload(
        tag_info=tag_info,
        space_hash=space_hash,
        entries=bank_entries,
        regime_summary=regime_summary,
    )

    validation_manifest = dict(cv_manifest)
    if cv_summary:
        validation_manifest["summary"] = cv_summary

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "run": tag_info,
        "space_hash": space_hash,
        "symbol": params_cfg.get("symbol"),
        "fees": fees,
        "risk": risk,
        "objectives": list(objectives),
        "search": params_cfg.get("search", {}),
        "strategy": {
            "config": params_cfg.get("strategy", {}),
            "resolved": {
                "module": strategy_instance.__class__.__module__,
                "class": strategy_instance.__class__.__name__,
            },
        },
        "resume_bank": str(resume_bank_path) if resume_bank_path else None,
        "study_storage": str(study_storage) if study_storage else None,
        "regime": regime_summary.__dict__,
        "cli": list(argv or []),
        "best_metric": {
            "name": best_metric_name,
            "direction": best_metric_direction,
            "value": best_record.get("primary_metric"),
        },
    }
    if validation_manifest:
        manifest["validation"] = validation_manifest

    _write_manifest(output_dir, manifest=manifest)
    write_bank_file(output_dir, bank_payload)
    (output_dir / "seed.yaml").write_text(
        yaml.safe_dump(
            {
                "params": params_cfg,
                "backtest": backtest_cfg,
                "forced_params": forced_params,
            },
            sort_keys=False,
        )
    )

    generate_reports(
        optimisation["results"],
        optimisation["best"],
        wf_summary,
        objectives,
        output_dir,
        study=optimisation.get("study"),
    )

    best_metric_value = best_record.get("primary_metric")
    LOGGER.info(
        "최적 trial=%s %s=%.4f (score=%.4f)",
        best_record.get("trial"),
        best_metric_name,
        float(best_metric_value) if best_metric_value is not None else float("nan"),
        best_record.get("score"),
    )
    LOGGER.info("Run complete. Outputs saved to %s", output_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for ``python -m optimize.run``."""

    args = parse_args(argv)
    execute(args, argv)


if __name__ == "__main__":
    main()
