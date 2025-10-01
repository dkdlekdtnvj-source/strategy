"""전략 파라미터 스키마 정의."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass
class UTBotParams:
    utKeyEff: float = 4.0
    utAtrEff: int = 10
    utHA: bool = False
    noRP: bool = True
    ibs: bool = False


@dataclass
class StochRsiParams:
    rsiLen: int = 14
    stLen: int = 14
    kLen: int = 3
    dLen: int = 3
    obEff: float = 80.0
    osEff: float = 20.0
    stMode: Literal["Bounce", "Cross"] = "Bounce"


@dataclass
class FilterParams:
    useMA100: bool = True
    maType: Literal["EMA", "SMA", "WMA"] = "EMA"
    useMicroTrend: bool = True
    emaFastLenBase: int = 21
    emaSlowLenBase: int = 55
    bbLen: int = 18
    bbMult: float = 1.2
    kcLen: int = 21
    useMomConfirm: bool = True
    useCHoCH: bool = True
    pL: int = 2
    pR: int = 3
    useVolumeFilter: bool = True
    volumeLookback: int = 34
    volumeMultiplier: float = 1.3
    useCandleFilter: bool = True
    candleBodyRatio: float = 55
    useEWO: bool = True
    ewoFast: int = 16
    ewoSlow: int = 26
    ewoSignal: int = 9
    useChop: bool = True
    chopLen: int = 14
    chopMax: float = 61.8
    useVolatility: bool = True
    minAtrPerc: float = 0.8
    maxAtrPerc: float = 8.0
    useVolumeBoost: bool = True
    volLen: int = 20
    useStructure: bool = True
    structureLen: int = 20
    useTrendBias: bool = True
    trendLenBase: int = 200
    useConfBias: bool = True
    confLenBase: int = 55


@dataclass
class ExitParams:
    atrLen: int = 14
    initKEff: float = 1.8
    tp1RR: float = 1.0
    tp2RR: float = 2.0
    tp1PctEff: float = 50.0
    trailKEff: float = 2.5
    beOffsetEff: float = 0.0
    useSwingSL: bool = False
    swN: int = 5
    usePercentStops: bool = True
    stopPctEff: float = 1.5
    takePctEff: float = 2.5
    trailStartEff: float = 1.0
    trailGapEff: float = 0.5
    maxBarsHoldEff: int = 0
    utFlipExit: bool = True
    roi1MinEff: int = 50
    roi1PctEff: float = 2.4
    roi2MinEff: int = 100
    roi2PctEff: float = 2.0
    roi3MinEff: int = 150
    roi3PctEff: float = 1.6


@dataclass
class RiskParams:
    lossCooldownBars: int = 10
    maxTradesPerDay: int = 0


@dataclass
class StrategyParams:
    ut: UTBotParams = field(default_factory=UTBotParams)
    stoch: StochRsiParams = field(default_factory=StochRsiParams)
    filters: FilterParams = field(default_factory=FilterParams)
    exits: ExitParams = field(default_factory=ExitParams)
    risk: RiskParams = field(default_factory=RiskParams)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyParams":
        def build(dataclass_type, values):
            if isinstance(values, dict):
                return dataclass_type(**values)
            return dataclass_type()

        return cls(
            ut=build(UTBotParams, data.get("ut", {})),
            stoch=build(StochRsiParams, data.get("stoch", {})),
            filters=build(FilterParams, data.get("filters", {})),
            exits=build(ExitParams, data.get("exits", {})),
            risk=build(RiskParams, data.get("risk", {})),
        )


def flatten_params(params: StrategyParams) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for group_name, group in params.__dict__.items():
        if hasattr(group, "__dict__"):
            for key, value in group.__dict__.items():
                flat[f"{group_name}_{key}"] = value
    return flat


__all__ = [
    "StrategyParams",
    "UTBotParams",
    "StochRsiParams",
    "FilterParams",
    "ExitParams",
    "RiskParams",
    "flatten_params",
]
