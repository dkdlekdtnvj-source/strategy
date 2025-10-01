"""재시작 및 워밍업 로직."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..basicmodule.utils import ensure_dir


@dataclass
class ResumeState:
    next_trial: int
    warm_params: Optional[pd.DataFrame]


def load_resume_state(report_dir: Path) -> ResumeState:
    ensure_dir(report_dir)
    summary_csv = report_dir / "summary.csv"
    if not summary_csv.exists():
        return ResumeState(next_trial=0, warm_params=None)
    summary = pd.read_csv(summary_csv)
    next_trial = int(summary["trial_id"].max()) + 1 if not summary.empty else 0
    top = summary.sort_values("score", ascending=False).head(10)
    return ResumeState(next_trial=next_trial, warm_params=top)


__all__ = ["ResumeState", "load_resume_state"]
