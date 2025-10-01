"""공통 유틸리티 함수 모음.

시간대 처리, 로깅, 체크포인트, 난수 시드 고정 등 전략 엔진 전반에서 공통으로 사용한다.
"""
from __future__ import annotations

import json
import logging
import os
import random
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import yaml

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from pytz import timezone as ZoneInfo  # type: ignore

KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc


@dataclass
class Checkpoint:
    """실행 재개를 위한 체크포인트 정보."""

    path: Path
    payload: Dict[str, Any]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["Checkpoint"]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(path=path, payload=payload)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path | str, data: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def to_kst(ts: datetime | pd.Timestamp) -> datetime:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(KST)


def from_kst(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=KST)
    return ts.astimezone(UTC)


def random_run_id(prefix: str = "run") -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{suffix}"


def chunked(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    merged.update({k: v for k, v in update.items() if v is not None})
    return merged


def summarize_params(params: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def timestamp_to_kst_label(start: datetime, end: datetime) -> str:
    start_kst = to_kst(start)
    end_kst = to_kst(end)
    return f"{start_kst:%Y%m%d}-{end_kst:%Y%m%d}"


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - 방어적 처리
        return None


__all__ = [
    "Checkpoint",
    "KST",
    "UTC",
    "chunked",
    "dump_yaml",
    "ensure_dir",
    "from_kst",
    "get_logger",
    "load_yaml",
    "merge_dicts",
    "random_run_id",
    "safe_read_csv",
    "set_global_seed",
    "summarize_params",
    "timestamp_to_kst_label",
    "to_float",
    "to_kst",
    "utc_now",
]
