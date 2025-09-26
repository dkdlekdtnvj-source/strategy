"""Caching helper for Binance OHLCV downloads."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .binance_client import BinanceClient

LOGGER = logging.getLogger(__name__)


@dataclass
class DataCache:
    root: Path
    futures: bool = False

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._client = BinanceClient(futures=self.futures)

    @staticmethod
    def _filename(symbol: str, timeframe: str, start: str, end: str) -> str:
        clean_symbol = symbol.replace(":", "_").replace("/", "")
        return f"BINANCE_{clean_symbol}_{timeframe}_{start.replace('-', '')}_{end.replace('-', '')}.csv"

    def _full_path(self, symbol: str, timeframe: str, start: str, end: str) -> Path:
        return self.root / self._filename(symbol, timeframe, start, end)

    def get(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        allow_partial: bool = False,
    ) -> pd.DataFrame:
        path = self._full_path(symbol, timeframe, start, end)
        if path.exists():
            LOGGER.info("Loading cached data from %s", path)
            frame = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        else:
            LOGGER.info("Downloading %s %s from Binance", symbol, timeframe)
            frame = self._client.fetch_ohlcv(symbol, timeframe, start, end)
            self._persist(path, frame)
        frame = frame.sort_index().loc[:, ["open", "high", "low", "close", "volume"]]
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.dropna(how="any") if not allow_partial else frame
        return frame

    def _persist(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        export = frame.copy()
        export.index.name = "timestamp"
        export.to_csv(path)
        LOGGER.info("Saved %s rows to %s", len(export), path)
