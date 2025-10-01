# Codex PPP Vishva x KASIA 백테스트 시스템

Pine Script 전략 "PPP Vishva x KASIA"의 신호/필터/청산 로직을 Python 백테스터로 이식하였다. ADX, 세션/요일 등의 시간 필터는 제외하고 핵심 불리언과 파라미터만 재현한다.

## 구성
- `basicmodule/`: 지표 계산, 필터, 청산, 엔진, 데이터 및 유틸 모듈
- `codex_runner/`: CLI 실행, Optuna 기반 최적화, 리포팅 및 재시작 로직
- `config/`: 기본 백테스트 및 파라미터 탐색 범위 설정
- `data/`: Binance 원본 및 캐시 데이터 저장 경로
- `reports/`: 심볼/타임프레임별 최적화 결과 저장

## 빠른 시작
```bash
python -m project.codex_runner.run --config project/config/backtest.yaml
```

실행 시 `config/backtest.yaml`에 정의된 기간(2024-01-01~2025-10-01 KST), 타임프레임(저차 1m/3m/5m), 심볼(Top50 USDT 페어)을 기준으로 데이터를 수집하고 Optuna 최적화를 수행한다. 상위 타임프레임 신호는 15m/60m 리샘플 데이터를 사용한다. 각 트라이얼 로그는 `reports/<TICKER>_<TF>_<기간>/trial_XXXX.csv`에 기록되고, 요약 및 최고 파라미터는 `summary.csv`, `summary.xlsx`, `best_params.yaml`로 저장된다.

## 주요 특징
- UTBot, StochRSI, 구조/모멘텀/거래량 필터 등 Pine 신호 로직을 Python으로 변환
- ROI/시간, ATR/퍼센트 스탑, 트레일링, UT Flip Exit 등 청산 로직 구현
- Optuna TPE + Median Pruner 기반 병렬 최적화(구성에 따라 Hyperband 선택 가능)
- 재시작 시 기존 summary를 읽어 다음 트라이얼 인덱스를 이어감
- Gemini API 보조 탐색은 환경 변수 `GEMINI_API_KEY` 설정 후 확장 가능 (기본값 비활성화)
- 적응형 탐색: 상위 성능 트라이얼을 기반으로 파라미터 범위를 재조정하고 필요 시 Gemini 제안을 병합

## 주의사항
- 대규모 데이터 수집 시 Binance API Rate Limit을 고려하여 실행 전 `ccxt` RateLimit 옵션을 유지하십시오.
- Gemini API 키는 로그나 리포트에 기록하지 않도록 코드에서 차단되어 있다.
- 본 예제는 파라미터 사이징 및 일부 가드 로직을 단순화하였으므로 실거래 적용 전 충분한 검증이 필요하다.
