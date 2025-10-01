"""데이터 수집 및 전처리 모듈."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd
import requests

from . import indicators
from .utils import UTC

BINANCE_REST = "https://api.binance.com/api/v3"


def get_top_symbols(quote: str = "USDT", limit: int = 50) -> List[str]:
    url = f"{BINANCE_REST}/ticker/24hr"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    filtered = [
        item for item in data if item.get("symbol", "").endswith(quote) and float(item.get("quoteVolume", 0)) > 0
    ]
    sorted_items = sorted(filtered, key=lambda x: float(x["quoteVolume"]), reverse=True)[:limit]
    return [item["symbol"] for item in sorted_items]


def ccxt_client() -> ccxt.binance:
    return ccxt.binance({"enableRateLimit": True})


def timeframe_to_minutes(tf: str) -> int:
    mapping = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60}
    if tf not in mapping:
        raise ValueError(f"지원하지 않는 타임프레임: {tf}")
    return mapping[tf]


def fetch_ohlcv(symbol: str, timeframe: str, start: datetime, end: datetime, dest: Path) -> Path:
    client = ccxt_client()
    ms_per_minute = 60 * 1000
    tf_minutes = timeframe_to_minutes(timeframe)
    since = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    records: List[List[float]] = []

    while since < end_ms:
        batch = client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        records.extend(batch)
        since = batch[-1][0] + tf_minutes * ms_per_minute

    if not records:
        raise RuntimeError(f"데이터 수집 실패: {symbol} {timeframe}")

    df = pd.DataFrame(records, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[(df["timestamp"] >= start.astimezone(UTC)) & (df["timestamp"] <= end.astimezone(UTC))]
    df.set_index("timestamp", inplace=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest)
    return dest


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df = df.sort_index()
    return df


def resample_htf(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = timeframe
    resampled = df.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled = resampled.shift(1)
    return resampled.dropna(how="all")


def add_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = indicators.heikin_ashi(df)
    return pd.concat([df, ha], axis=1)


def prepare_dataset(
    raw_path: Path,
    timeframe: str,
    htf_map: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    df = load_ohlcv(raw_path)
    df = add_heikin_ashi(df)
    htf_data = {name: resample_htf(df, tf) for name, tf in htf_map.items()}
    return df, htf_data


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def read_manifest(path: Path) -> Dict[str, any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path: Path, payload: Dict[str, any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


__all__ = [
    "BINANCE_REST",
    "add_heikin_ashi",
    "ccxt_client",
    "fetch_ohlcv",
    "get_top_symbols",
    "load_ohlcv",
    "prepare_dataset",
    "read_manifest",
    "resample_htf",
    "save_dataframe",
    "timeframe_to_minutes",
    "write_manifest",
]
