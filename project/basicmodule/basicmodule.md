# Codex PPP Vishva x KASIA Python 백테스트 설계 요약

본 문서는 Pine Script 전략 "PPP Vishva x KASIA"에서 추출한 신호/필터/청산 로직을 Python 백테스트 엔진으로 이식하기 위한 구조와 규격을 요약한다. ADX, 세션/요일과 같은 시간 필터는 제외하고 불리언 스위치와 파라미터만 모듈화하였다.

## 디렉터리 구조
```
basicmodule/
  indicators.py   # 지표 계산 (UTBot, StochRSI, EWO, Choppiness 등)
  filters.py      # 신호 필터, 마이크로 트렌드, 구조 필터
  exits.py        # 손절/익절, 트레일링, ROI/Time, 퍼센트 스탑 로직
  engine.py       # 포지션 상태 머신과 동일봉 1회 진입 제어
  metrics.py      # Profit Factor, Sortino, Net Profit 및 보조 지표
  data.py         # Binance/CCXT 데이터 수집 및 리샘플
  utils.py        # 공통 유틸리티 (로그, 타임존, 체크포인트)
  schema.py       # 파라미터 스키마 및 탐색 범위 정의
```

## 지표 & 신호 요약
- **UTBot**: `utKeyEff`, `utAtrEff`, `utHA`, `noRP`, `ibs`
- **StochRSI**: `rsiLen`, `stLen`, `kLen`, `dLen`, `obEff`, `osEff`, `stMode`
- **필터**: MA100 방향, MicroTrend EMA 클라우드, Squeeze & Momentum, CHoCH 구조, Volume/Candle, 추가 필터(EWO, Choppiness, Volatility%, VolumeBoost, Structure, Trend/Confirm EMA)
- **청산**: ATR 기반, TP1/TP2, Trail/Breakeven, Percent Stops, Swing SL, ROI/Time, UT Flip Exit, Cooldown & Daily Limits

각 지표는 확정봉 데이터만 사용하며, HTF 리샘플은 `lookahead=off`에 해당하는 시프팅 처리를 Python에서 `shift(1)`로 구현한다.

## 백테스트 규격
- 데이터 기간: 2024-01-01 ~ 2025-10-01 (KST, 내부 UTC)
- 타임프레임: 저차 1m/3m/5m, 상위 확인용 15m/1h 리샘플
- 수수료/슬리피지: 구성 파일에서 설정 (기본 0.05%, 1틱)
- 동일봉 재진입 금지: `noRP` / `ibs` 로직으로 제어
- 저장 규격: 각 트라이얼마다 `trial_<id>.csv`, `trial_<id>_metrics.json`, 실행 요약 `summary.csv`, `summary.xlsx`, `best_params.yaml`

## 테스트 기준
- ETHUSDT 1m 구간 200 트라이얼 시 충돌 없이 요약 파일 생성
- 재시작 시 진행 중단 트라이얼 이후부터 이어서 실행
- Pine Script 대비 주요 청산 이벤트 타임스탬프 95% 이상 일치 (샘플 100개 비교)
- 병렬 최적화 중 파일 손상/경쟁 조건 없이 저장 유지

## 의존성
- `pandas`, `numpy`, `scipy`, `optuna`, `ccxt`, `pyyaml`, `tqdm`, `openpyxl`, `requests`
- Gemini API 사용 시 환경 변수 `GEMINI_API_KEY`만으로 REST 호출 (추가 패키지 불필요)

## 기본 테스트 명령
```bash
python -m codex_runner.run --config config/backtest.yaml
```

