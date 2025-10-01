"""CLI 엔트리 포인트."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from ..basicmodule import data as data_mod
from ..basicmodule.utils import load_yaml
from .optimizer import OptimizationConfig, OptimizationManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex PPP 백테스트 실행기")
    parser.add_argument("--config", type=Path, required=True, help="백테스트 설정 YAML 경로")
    return parser.parse_args()


def resolve_symbols(cfg: dict) -> List[str]:
    symbols = cfg.get("symbols")
    if symbols == "top50":
        top_list = data_mod.get_top_symbols(limit=50)
        print("거래량 상위 50개 티커:")
        for idx, sym in enumerate(top_list, start=1):
            print(f"{idx:2d}. {sym}")
        selected = input("사용할 티커 번호를 콤마로 입력하세요 (예: 1,3,10): ").strip()
        if not selected:
            return top_list
        chosen = []
        for token in selected.split(","):
            try:
                index = int(token) - 1
            except ValueError:
                continue
            if 0 <= index < len(top_list):
                chosen.append(top_list[index])
        return chosen or top_list
    if isinstance(symbols, list):
        return symbols
    raise ValueError("symbols 설정이 필요합니다.")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    opt_config = OptimizationConfig.from_dict(cfg)
    param_space = load_yaml(Path(cfg.get("params_space", "config/params_space.yaml")))
    tf_settings = cfg.get(
        "timeframes_settings",
        {"stoch": "15m", "htf1": "15m", "htf2": "1h", "regime": "1h"},
    )
    manager = OptimizationManager(
        config=opt_config,
        param_space={k: v for k, v in param_space.items() if k in {"ut", "stoch", "filters", "exits", "risk"}},
        data_dir=Path(cfg.get("data_dir", "project/data")),
        reports_dir=Path(cfg.get("reports_dir", "project/reports")),
        tf_settings=tf_settings,
    )

    symbols = resolve_symbols(cfg)
    timeframes = cfg.get("timeframes", ["1m"])

    for symbol in symbols:
        for tf in timeframes:
            manager.optimize(symbol, tf)


if __name__ == "__main__":
    main()
