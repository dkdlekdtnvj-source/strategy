"""리포팅 유틸리티."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

import yaml

from ..basicmodule import metrics
from ..basicmodule.utils import ensure_dir


@dataclass
class TrialRecord:
    trial_id: int
    params: Dict[str, object]
    metrics: metrics.Metrics
    score: float
    log: pd.DataFrame


@dataclass
class ReportManager:
    base_dir: Path
    symbol: str
    timeframe: str
    period_label: str
    records: List[TrialRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        ensure_dir(self.output_dir)

    @property
    def output_dir(self) -> Path:
        return self.base_dir / f"{self.symbol}_{self.timeframe}_{self.period_label}"

    def add_record(self, record: TrialRecord) -> None:
        self.records.append(record)
        self._write_trial(record)

    def _write_trial(self, record: TrialRecord) -> None:
        trial_name = f"trial_{record.trial_id:04d}"
        csv_path = self.output_dir / f"{trial_name}.csv"
        json_path = self.output_dir / f"{trial_name}_metrics.json"
        record.log.to_csv(csv_path, index=False)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "params": record.params,
                    "metrics": record.metrics.as_dict(),
                    "score": record.score,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def finalize(self) -> None:
        if not self.records:
            return
        existing = None
        summary_path = self.output_dir / "summary.csv"
        if summary_path.exists():
            existing = pd.read_csv(summary_path)

        summary_rows = [
            {
                "trial_id": rec.trial_id,
                **rec.metrics.as_dict(),
                "score": rec.score,
                "params": rec.params,
            }
            for rec in self.records
        ]
        summary_df = pd.DataFrame(summary_rows)
        if existing is not None and not existing.empty:
            summary_df = pd.concat([existing, summary_df], ignore_index=True)
        summary_csv = summary_path
        summary_xlsx = self.output_dir / "summary.xlsx"
        summary_df.to_csv(summary_csv, index=False)
        summary_df.to_excel(summary_xlsx, index=False)
        best = summary_df.sort_values("score", ascending=False).iloc[0]
        best_path = self.output_dir / "best_params.yaml"
        best_dict = {"trial_id": int(best["trial_id"]), "score": float(best["score"]), "params": best["params"]}
        with best_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(best_dict, f, allow_unicode=True, sort_keys=False)


__all__ = ["ReportManager", "TrialRecord"]
