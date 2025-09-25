"""Client utilities for fetching OHLCV data from Binance with retries."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency handled via requirements
    raise RuntimeError("ccxt is required for Binance data access") from exc

LOGGER = logging.getLogger(__name__)


def _to_milliseconds(dt: datetime | str | pd.Timestamp) -> int:
    ts = pd.Timestamp(dt, tz="UTC")
    return int(ts.timestamp() * 1000)


def _parse_symbol(symbol: str) -> str:
    if symbol.upper().startswith("BINANCE:"):
        return symbol.split(":", 1)[1]
    return symbol


@dataclass
class BinanceClient:
    """Thin wrapper around ccxt.binance with retry and rate-limit handling."""

    futures: bool = False
    max_retries: int = 5
    retry_wait: float = 2.0
    rate_limit: float = 0.2

    def __post_init__(self) -> None:
        exchange_class = ccxt.binanceusdm if self.futures else ccxt.binance
        self._client = exchange_class({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | str,
        end: datetime | str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        market_symbol = _parse_symbol(symbol)
        since = _to_milliseconds(start)
        end_ms = _to_milliseconds(end)
        all_rows: List[List[float]] = []

        while since < end_ms:
            for attempt in range(1, self.max_retries + 1):
                try:
                    batch = self._client.fetch_ohlcv(
                        market_symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=limit,
                    )
                    break
                except ccxt.NetworkError as err:  # pragma: no cover - network failure path
                    LOGGER.warning("Network error from Binance: %s (attempt %d)", err, attempt)
                    time.sleep(self.retry_wait * attempt)
                except ccxt.ExchangeError as err:
                    LOGGER.error("Exchange error from Binance: %s", err)
                    raise
            else:  # no break
                raise RuntimeError("Exceeded maximum retries while fetching data")

            if not batch:
                break

            all_rows.extend(batch)
            since = batch[-1][0] + self._client.parse_timeframe(timeframe) * 1000
            time.sleep(self.rate_limit)

        if not all_rows:
            raise ValueError("No OHLCV data returned from Binance")

        frame = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()
        mask = (frame.index >= pd.Timestamp(start, tz="UTC")) & (frame.index <= pd.Timestamp(end, tz="UTC"))
        frame = frame.loc[mask]
        return frame


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: datetime | str,
    end: datetime | str,
    futures: bool = False,
    **kwargs,
) -> pd.DataFrame:
    client = BinanceClient(futures=futures, **kwargs)
    return client.fetch_ohlcv(symbol, timeframe, start, end)
