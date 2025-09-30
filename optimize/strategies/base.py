"""전략 플러그인 인터페이스를 정의합니다."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - 순환 참조 방지용 타입 힌트
    from optimize.run import DatasetSpec


class StrategyModel(ABC):
    """최적화 파이프라인이 사용하는 전략 구현 기본 클래스."""

    def prepare_data(
        self,
        df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        params: Dict[str, object],
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """전략 실행 전에 데이터 전처리를 수행합니다."""

        return df, htf_df

    @abstractmethod
    def run_backtest(
        self,
        df: pd.DataFrame,
        params: Dict[str, float | bool],
        fees: Dict[str, float],
        risk: Dict[str, float],
        *,
        htf_df: Optional[pd.DataFrame] = None,
        min_trades: Optional[int] = None,
    ) -> Dict[str, float]:
        """주어진 파라미터로 백테스트를 실행하고 주요 지표를 반환합니다."""


__all__ = ["StrategyModel"]
