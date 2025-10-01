"""Optuna 기반 최적화 엔진 및 적응형 탐색 로직."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import requests

from ..basicmodule import data as data_mod
from ..basicmodule.engine import StrategyEngine
from ..basicmodule.schema import StrategyParams
from ..basicmodule.utils import set_global_seed, timestamp_to_kst_label
from .reporting import ReportManager, TrialRecord
from .resume import load_resume_state


@dataclass
class OptimizationConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    trials_per_combo: int
    n_jobs: int
    pruner: str
    objective_weights: Dict[str, float]
    resume: bool
    use_gemini: bool
    seed: int
    adapt_interval: int = 100
    adapt_quantile: Tuple[float, float] = (0.1, 0.9)
    top_ratio: float = 0.1

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "OptimizationConfig":
        opt = cfg.get("optuna", {})
        return cls(
            start=pd.Timestamp(cfg["start"]),
            end=pd.Timestamp(cfg["end"]),
            trials_per_combo=int(opt.get("trials_per_combo", 100)),
            n_jobs=int(opt.get("n_jobs", 1)),
            pruner=opt.get("pruner", "median"),
            objective_weights=cfg.get("objective_weights", {"pf": 0.6, "sortino": 0.3, "netprofit": 0.1}),
            resume=cfg.get("resume", True),
            use_gemini=cfg.get("use_gemini", False),
            seed=int(cfg.get("seed", 42)),
            adapt_interval=int(opt.get("adapt_interval", 100)),
            adapt_quantile=(
                float(opt.get("adapt_low_quantile", 0.1)),
                float(opt.get("adapt_high_quantile", 0.9)),
            ),
            top_ratio=float(opt.get("adapt_top_ratio", 0.1)),
        )


class ParamSampler:
    """파라미터 탐색 공간을 관리하는 래퍼."""

    def __init__(self, space: Dict[str, Any]):
        self._base_space: Dict[str, Any] = json.loads(json.dumps(space))
        self._current_space: Dict[str, Any] = json.loads(json.dumps(space))

    def sample(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            key: self._sample_group(trial, key, group)
            for key, group in self._current_space.items()
        }

    def _sample_group(self, trial: optuna.Trial, group_name: str, group: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, spec in group.items():
            full_name = f"{group_name}.{name}"
            ptype = spec.get("type", "float")
            if ptype == "float":
                params[name] = trial.suggest_float(full_name, spec["low"], spec["high"], step=spec.get("step"))
            elif ptype == "int":
                params[name] = trial.suggest_int(full_name, spec["low"], spec["high"], step=spec.get("step", 1))
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(full_name, spec["choices"])
            else:
                raise ValueError(f"알 수 없는 파라미터 타입: {ptype}")
        return params

    def update_space(self, new_space: Dict[str, Any]) -> None:
        self._current_space = json.loads(json.dumps(new_space))

    def get_space(self) -> Dict[str, Any]:
        return self._current_space

    def get_base_space(self) -> Dict[str, Any]:
        return self._base_space


class GeminiAdvisor:
    """Gemini API 기반 범위 추천 도우미."""

    def __init__(self, enabled: bool, model: str = "gemini-1.5-flash-latest") -> None:
        self.enabled = enabled
        self.model = model
        self._api_key = os.getenv("GEMINI_API_KEY") if enabled else None

    def available(self) -> bool:
        return bool(self.enabled and self._api_key)

    def propose(self, records: Iterable[TrialRecord], base_space: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.available():
            return None

        payload = self._build_payload(records, base_space)
        if payload is None:
            return None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self._api_key}"
        )
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            return None
        return self._parse_response(response.json())

    def _build_payload(self, records: Iterable[TrialRecord], base_space: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for record in records:
            summary.append(
                {
                    "trial_id": record.trial_id,
                    "score": record.score,
                    "metrics": record.metrics,
                    "params": record.params,
                }
            )
        if not summary:
            return None

        prompt = (
            "당신은 알고리즘 트레이딩 파라미터 최적화를 돕는 전문가입니다. "
            "제공된 상위 실험 결과와 기본 탐색 공간을 바탕으로 다음 탐색을 위한 범위를 추천하세요. "
            "각 파라미터는 JSON 객체로 low/high 또는 choices만 포함하도록 하며, 다른 텍스트는 포함하지 마세요."
        )

        content = json.dumps({"records": summary, "base_space": base_space}, ensure_ascii=False)
        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"text": content},
                    ]
                }
            ]
        }

    def _parse_response(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for candidate in payload.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if not text:
                    continue
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
        return None


class OptimizationManager:
    def __init__(
        self,
        config: OptimizationConfig,
        param_space: Dict[str, Any],
        data_dir: Path,
        reports_dir: Path,
        tf_settings: Dict[str, str],
    ) -> None:
        self.config = config
        self.param_sampler = ParamSampler(param_space)
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.tf_settings = tf_settings
        self.gemini = GeminiAdvisor(enabled=config.use_gemini)
        set_global_seed(config.seed)

    def _prepare_data(self, symbol: str, timeframe: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        file_name = f"{symbol}_{timeframe}_{timestamp_to_kst_label(self.config.start, self.config.end)}.csv"
        raw_path = self.data_dir / "raw" / symbol / timeframe / file_name
        if not raw_path.exists():
            data_mod.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=self.config.start.to_pydatetime(),
                end=self.config.end.to_pydatetime(),
                dest=raw_path,
            )
        tf_map = {
            "htf1": self.tf_settings.get("htf1", "15m"),
            "htf2": self.tf_settings.get("htf2", "1h"),
            "stoch": self.tf_settings.get("stoch", timeframe),
            "regime": self.tf_settings.get("regime", "1h"),
        }
        df, htf_data = data_mod.prepare_dataset(raw_path, timeframe, tf_map)
        return df, htf_data

    def optimize(self, symbol: str, timeframe: str) -> None:
        df, htf_data = self._prepare_data(symbol, timeframe)
        period_label = timestamp_to_kst_label(self.config.start, self.config.end)
        report_manager = ReportManager(Path(self.reports_dir), symbol, timeframe, period_label)
        resume_state = load_resume_state(report_manager.output_dir) if self.config.resume else None
        start_trial = resume_state.next_trial if resume_state else 0

        trial_records: List[TrialRecord] = []
        record_lock = threading.Lock()

        def maybe_adapt_space() -> None:
            if len(trial_records) < self.config.adapt_interval:
                return
            if len(trial_records) % self.config.adapt_interval != 0:
                return
            updated = self._build_adapted_space(trial_records)
            if updated:
                self.param_sampler.update_space(updated)
                if self.gemini.available():
                    gemini_space = self.gemini.propose(
                        self._select_top_records(trial_records),
                        self.param_sampler.get_base_space(),
                    )
                    if isinstance(gemini_space, dict):
                        merged = self._merge_external_space(updated, gemini_space)
                        if merged:
                            self.param_sampler.update_space(merged)

        def objective(trial: optuna.Trial) -> float:
            sampled = self.param_sampler.sample(trial)
            params = StrategyParams.from_dict(sampled)
            engine = StrategyEngine(
                params=params,
                fee=trial.study.user_attrs.get("fee", 0.0005),
                slippage=trial.study.user_attrs.get("slippage", 0.0),
                initial_capital=trial.study.user_attrs.get("initial_capital", 10000),
                objective_weights=self.config.objective_weights,
                timeframe=timeframe,
                tf_settings=self.tf_settings,
            )
            prepared = engine.prepare(df, htf_data)
            result = engine.run(df, prepared)
            record = TrialRecord(
                trial_id=start_trial + trial.number,
                params=sampled,
                metrics=result.metrics,
                score=result.score,
                log=result.logs,
            )
            report_manager.add_record(record)
            with record_lock:
                trial_records.append(record)
                maybe_adapt_space()
            return result.score

        pruner = optuna.pruners.MedianPruner() if self.config.pruner == "median" else optuna.pruners.HyperbandPruner()
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.set_user_attr("fee", 0.0005)
        study.set_user_attr("slippage", 0.0)
        study.set_user_attr("initial_capital", 10000)
        study.optimize(objective, n_trials=self.config.trials_per_combo, n_jobs=self.config.n_jobs)
        report_manager.finalize()

    def _select_top_records(self, records: List[TrialRecord]) -> List[TrialRecord]:
        if not records:
            return []
        sorted_records = sorted(records, key=lambda record: record.score, reverse=True)
        top_n = max(1, int(len(sorted_records) * self.config.top_ratio))
        return sorted_records[:top_n]

    def _build_adapted_space(self, records: List[TrialRecord]) -> Optional[Dict[str, Any]]:
        top_records = self._select_top_records(records)
        if not top_records:
            return None

        base_space = self.param_sampler.get_base_space()
        current_space = json.loads(json.dumps(self.param_sampler.get_space()))
        low_q, high_q = self.config.adapt_quantile

        def clamp(val: float, low: float, high: float) -> float:
            return max(low, min(high, val))

        for group, params in current_space.items():
            for name, spec in params.items():
                ptype = spec.get("type", "float")
                values: List[Any] = [
                    record.params.get(group, {}).get(name)
                    for record in top_records
                    if name in record.params.get(group, {})
                ]
                values = [v for v in values if v is not None]
                if not values:
                    continue

                base_spec = base_space[group][name]
                if ptype in {"float", "int"}:
                    arr = np.array(values, dtype=float)
                    lo = float(np.quantile(arr, low_q))
                    hi = float(np.quantile(arr, high_q))
                    lo = clamp(lo, base_spec["low"], base_spec["high"])
                    hi = clamp(hi, base_spec["low"], base_spec["high"])
                    if lo == hi:
                        hi = min(base_spec["high"], lo + (base_spec.get("step", 0.1) or 0.1))
                    if ptype == "int":
                        lo = int(round(lo))
                        hi = int(round(hi))
                        if lo == hi:
                            hi = min(base_spec["high"], lo + max(1, base_spec.get("step", 1)))
                    spec["low"] = lo
                    spec["high"] = hi
                elif ptype == "categorical":
                    unique: List[Any] = []
                    for value in values:
                        if value not in unique:
                            unique.append(value)
                    filtered = [v for v in unique if v in base_spec["choices"]]
                    spec["choices"] = filtered or base_spec["choices"]
        return current_space

    def _merge_external_space(
        self,
        current: Dict[str, Any],
        external: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        merged = json.loads(json.dumps(current))
        base = self.param_sampler.get_base_space()
        for group, params in external.items():
            if group not in merged:
                continue
            for name, payload in params.items():
                if name not in merged[group]:
                    continue
                ptype = merged[group][name].get("type", "float")
                base_spec = base[group][name]
                try:
                    if ptype in {"float", "int"} and isinstance(payload, dict):
                        low = float(payload.get("low", base_spec["low"]))
                        high = float(payload.get("high", base_spec["high"]))
                        if ptype == "int":
                            merged[group][name]["low"] = int(round(low))
                            merged[group][name]["high"] = int(round(high))
                        else:
                            merged[group][name]["low"] = low
                            merged[group][name]["high"] = high
                    elif ptype == "categorical" and isinstance(payload, list):
                        filtered = [item for item in payload if item in base_spec["choices"]]
                        if filtered:
                            merged[group][name]["choices"] = filtered
                except (TypeError, ValueError):
                    continue
        return merged


__all__ = ["OptimizationConfig", "OptimizationManager"]

