"""Regime feature extraction and contextual bandit helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def compute_regime_features(df: pd.DataFrame) -> Dict[str, float]:
    """Derive descriptive statistics from price data for regime labelling."""

    if df.empty:
        return {"atr_pct": 0.0, "trend_slope": 0.0, "session": "unknown"}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr = (high - low).rolling(window=14, min_periods=5).mean()
    atr_pct = float((atr / close).iloc[-1] * 100) if not atr.empty else 0.0

    ema_fast = close.ewm(span=34, adjust=False).mean()
    ema_slow = close.ewm(span=144, adjust=False).mean()
    slope_window = min(100, len(close) - 1)
    if slope_window > 1:
        slope = float((ema_fast.iloc[-1] - ema_fast.iloc[-slope_window]) / ema_fast.iloc[-slope_window])
    else:
        slope = 0.0

    timestamp = close.index[-1] if isinstance(close.index, pd.DatetimeIndex) else None
    if timestamp is not None:
        hour = timestamp.tz_convert("UTC").hour if timestamp.tzinfo else timestamp.hour
        if 12 <= hour < 20:
            session = "us"
        elif 7 <= hour < 12:
            session = "eu"
        else:
            session = "asia"
    else:
        session = "unknown"

    return {"atr_pct": atr_pct, "trend_slope": slope, "session": session}


def label_regime(features: Dict[str, float]) -> str:
    atr_pct = float(features.get("atr_pct", 0.0))
    slope = float(features.get("trend_slope", 0.0))
    session = features.get("session", "unknown")

    if atr_pct >= 1.5:
        vol = "high"
    elif atr_pct <= 0.6:
        vol = "low"
    else:
        vol = "mid"

    if slope >= 0.01:
        trend = "up"
    elif slope <= -0.01:
        trend = "down"
    else:
        trend = "flat"

    return f"vol_{vol}|trend_{trend}|session_{session}"


@dataclass
class BanditArm:
    count: int = 0
    total: float = 0.0

    def update(self, reward: float) -> None:
        self.count += 1
        self.total += reward

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass
class RegimeBandit:
    exploration: float = 0.6
    regimes: Dict[str, Dict[str, BanditArm]] = field(default_factory=dict)
    total_updates: int = 0

    def update(self, regime: str, arm_id: str, reward: float) -> None:
        arms = self.regimes.setdefault(regime, {})
        arm = arms.setdefault(arm_id, BanditArm())
        arm.update(reward)
        self.total_updates += 1

    def select(self, regime: str) -> Optional[str]:
        arms = self.regimes.get(regime)
        if not arms:
            return None
        best_score = float("-inf")
        best_arm: Optional[str] = None
        total = max(self.total_updates, 1)
        for arm_id, arm in arms.items():
            exploration_bonus = self.exploration * np.sqrt(np.log(total) / max(arm.count, 1))
            value = arm.mean + exploration_bonus
            if value > best_score:
                best_score = value
                best_arm = arm_id
        return best_arm

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"exploration": self.exploration, "total_updates": self.total_updates, "regimes": {}}
        for regime, arms in self.regimes.items():
            payload["regimes"][regime] = {arm_id: {"count": arm.count, "total": arm.total} for arm_id, arm in arms.items()}
        return payload

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "RegimeBandit":
        bandit = cls()
        if not data:
            return bandit
        bandit.exploration = float(data.get("exploration", bandit.exploration))
        bandit.total_updates = int(data.get("total_updates", 0))
        regimes = data.get("regimes", {})
        if isinstance(regimes, dict):
            for regime, arms in regimes.items():
                if not isinstance(arms, dict):
                    continue
                for arm_id, payload in arms.items():
                    count = int(payload.get("count", 0))
                    total = float(payload.get("total", 0.0))
                    if count <= 0:
                        continue
                    arm = bandit.regimes.setdefault(regime, {}).setdefault(arm_id, BanditArm())
                    arm.count = count
                    arm.total = total
        return bandit


def summarise_regimes(bandit: RegimeBandit) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for regime, arms in bandit.regimes.items():
        for arm_id, arm in arms.items():
            summary.append({"regime": regime, "arm": arm_id, "count": arm.count, "mean_reward": arm.mean})
    return summary


__all__ = [
    "compute_regime_features",
    "label_regime",
    "RegimeBandit",
    "summarise_regimes",
]
