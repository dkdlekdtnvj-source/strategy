# Pine Strategy Optimisation Toolkit

This repository provides a reproducible pipeline to optimise a Pine Script strategy with
Binance market data. It includes a deterministic Pine script profile, a Python backtest
model, Optuna-based search utilities, reporting helpers, and walk-forward evaluation.

## Project layout

```
strategy/strategy.pine         # Pine reference strategy (optimiser-ready inputs)
config/params.yaml             # Single profile configuration
config/backtest.yaml           # Batch/sweep configuration
optimize/run.py                # CLI entry point
optimize/strategy_model.py     # Python equivalent of the Pine logic
optimize/metrics.py            # Performance metrics and scoring
optimize/search_spaces.py      # YAML → Optuna helper
optimize/wf.py                 # Walk-forward analysis
optimize/report.py             # CSV/JSON/heatmap reporting
optimize/llm.py                # Gemini 기반 후보 제안 도우미
datafeed/binance_client.py     # Binance downloader with retries
datafeed/cache.py              # CSV cache layer
reports/                       # Optimisation output (ignored by git)
studies/                       # Optuna SQLite storage (auto-created)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Ensure Binance OHLCV data is cached under `data/` (the CLI will download missing
   ranges automatically).
2. Adjust optimisation inputs in `config/params.yaml`. The sample profile now sweeps
   the squeeze-momentum core (oscillator length, signal length, BB/KC lengths & multipliers),
   directional-flux smoothing, dynamic threshold options, higher-timeframe confirmation,
   and exit modules (ATR stop, fixed % stop, swing/pivot stops, ATR trail,
   breakeven, time stop). You can also pre-define `overrides` to pin any parameter
   on/off before a run and enable Top-K walk-forward re-ranking via `search.top_k`.
3. Configure the sweep universe in `config/backtest.yaml`. By default it contains
  nine Binance USDT perpetual pairs (ENA, ETH, BTC, SOL, **XPL**, **ASTER**, DOGE,
  XRP, SUI) with lower timeframes 1m/3m/5m, higher timeframes 15m/1h, and a single
  2024-01-01 → 2025-09-25 창. The optimiser now treats the LTF/HTF selections as
  categorical parameters, so each Optuna trial chooses one combination while the
  reports highlight which pairing delivers the strongest Sortino/Profit Factor.
  The `symbol_aliases` mapping lets you keep the newly-listed ticker names you
  prefer (`XPLUSDT`, `ASTERUSDT`) while the optimiser automatically fetches data
  using Binance's current instruments (`XPLAUSDT`, `ASTRUSDT`).
4. Launch the optimiser non-interactively or with prompts:

   ```bash
   python -m optimize.run --params config/params.yaml --backtest config/backtest.yaml
   ```

   or

   ```bash
   python -m optimize.run --interactive
   ```

   The interactive mode lets you pick the symbol, evaluation window, leverage,
   position size, and the boolean filters (HTF sync, ATR trail, pivot stops,
   breakeven, etc.) from the terminal. Command-line overrides are also
   available:

   - `--symbol`, `--timeframe`, `--htf`, `--start`, `--end`
   - `--leverage`, `--qty-pct`
   - `--n-trials`
   - `--enable name1,name2`, `--disable name3`
   - `--top-k 10` to re-rank the best Optuna trials by walk-forward out-of-sample
     performance.

  Outputs are written to `reports/` (`results.csv`, `results_datasets.csv`,
  `results_timeframe_summary.csv`, `results_timeframe_rankings.csv`, `best.json`,
  `heatmap.png`) along with a machine-readable `trials/` 폴더(`trials.jsonl`,
  `trials_live.csv`, `best.yaml`, `trials_final.csv`). `trials_live.csv` 는 각
  트라이얼이 끝날 때마다 즉시 행이 추가되므로, 중간에 실행을 중단하더라도
  이미 탐색한 파라미터/지표 히스토리를 엑셀에서 바로 열어볼 수 있습니다.
  The `results_datasets.csv`
  file is especially useful for answering
  “어떤 LTF/HTF 조합이 가장 좋은가요?” because every dataset row lists the symbol,
  LTF, HTF, and the full metric set (Net Profit, Sortino, Profit Factor, MaxDD,
  Win Rate, Weekly Net Profit, 등). The automatically generated
  `results_timeframe_summary.csv`/`results_timeframe_rankings.csv` pair then pivots
  those rows into 평균·중앙값 테이블과 정렬 리스트 so you can immediately compare
  1m/3m/5m vs 15m/60m 조합. The `best.json` payload also includes the Top-K candidate
  summary with their walk-forward scores.

### Quick-start helper

If you would like a guided experience without memorising the CLI switches, run:

```bash
python -m optimize.quickstart
```

The helper will prompt for the symbol, backtest window, leverage, position size,
boolean filter toggles, and trial count. Once the questions are answered it
forwards the selections to `optimize.run` and the
reports appear under `reports/` just like the direct CLI entry point.

## LLM 보조(선택)

`config/params.yaml` 의 `llm` 블록을 활성화하면 일정 수(`initial_trials`) 만큼의
Optuna 트라이얼을 먼저 수행한 뒤 Gemini API에 "탑 트라이얼 요약 + 탐색 공간"을
전달해 유망한 파라미터 조합을 JSON 배열로 받아옵니다. 응답은 경계·스텝·카테고리
규칙을 모두 통과했을 때만 `study.enqueue_trial()` 로 큐에 넣어 남은 트라이얼에서
평가합니다.

- API 키는 `GEMINI_API_KEY` 환경변수 또는 `llm.api_key` 항목에서 읽습니다. 샘플
  프로필에는 요청하신 무료 키가 기본값으로 포함돼 있지만, 운영 환경에서는 환경
  변수 사용을 권장합니다.
- 기본 모델은 `gemini-2.0-flash-exp` 이며 `top_n`/`count` 값으로 참고할 트라이얼
  수와 제안 받을 후보 수를 제어할 수 있습니다.
- `google-genai` 패키지가 설치돼 있지 않으면 경고만 출력하고 LLM 단계를 건너뜁니다.

예시:

```bash
export GEMINI_API_KEY="AIzaSyDD1i5TbCqfWEMFunoxtvnpnr0VW3XZtsY"
python -m optimize.run --params config/params.yaml --backtest config/backtest.yaml
```

실행 후 `reports/<timestamp>.../trials/trials.jsonl` 에서는 각 트라이얼의 상태, 점수,
파라미터를 줄 단위 JSON 으로 확인할 수 있고 `trials_final.csv` 는 Excel/BI 도구에서
바로 열 수 있도록 준비됩니다.

## Testing

```bash
pytest
```

## Notes

- The Pine script uses `process_orders_on_close=true`, `request.security(..., lookahead_off)`,
  and confirmed-bar guards to prevent repainting. Commission, slippage, leverage, minimum
  tradable capital, and liquidation buffers are exposed as optimiser-friendly inputs.
- Walk-forward parameters (train/test window length and step) can be configured under
  `walk_forward` in `config/params.yaml`. The Python model now mirrors the Pine exits more
  closely with ATR trailing, pivot stops, breakeven, time-stop handling, minimum hold
  enforcement, and optional event-window gating.
- Metrics include Net Profit, Max Drawdown, Sortino, Sharpe, Profit Factor, Win Rate,
  weekly net profit, expectancy, RR, average MFE/MAE, and average holding period. The
  optimiser combines weighted objectives with penalties for breaching the risk gates and
  can optionally re-score the top trials by walk-forward OOS mean. 각 목표 지표는
  `direction`(maximize/minimize)을 명시할 수 있으며, `search.multi_objective=true`
  설정 시 Optuna 파레토 탐색으로 전환됩니다.
- `strategy` 블록에서 사용할 파이썬 전략 클래스를 모듈/클래스로 지정할 수 있어
  동일한 최적화 파이프라인으로 다양한 전략을 플러그인 형태로 교체할 수 있습니다.
- `search.n_jobs` 를 `auto` 로 두면 시스템 CPU 코어 수에 맞춰 병렬 최적화가 동작하며,
  다중 스레드 환경에서도 로그/결과 기록이 안전하게 직렬화됩니다.
- `search.best_metric` / `search.best_metric_direction` 으로 최종 베스트 후보를
  어떤 지표(기본: ProfitFactor)와 방향으로 선정할지 정의할 수 있습니다. 내부 스코어와
  무관하게 지정한 지표가 가장 우수한 트라이얼이 `best.json` 등에 기록됩니다.
- `search.refine` 블록을 활성화하면 일정 주기마다 현재 상위 트라이얼을 기준으로
  국소 돌연변이 파라미터를 자동 큐잉해 탐색이 빠르게 수렴하도록 돕습니다.
- `combine_metrics` 는 이제 각 데이터셋의 수익률 시리즈와 트레이드 리스트를 사용해
  실제 포트폴리오 기준으로 성과·위험 지표(ProfitFactor, MaxDD 등)를 재계산합니다.
- 최적화 중 `trials.jsonl` 에는 NetProfit/ProfitFactor/Sortino/MaxDD/Trades 등 핵심 지표가
  함께 기록되며, 리포트 생성 시 Optuna 파라미터 중요도 JSON과 시각화가 추가됩니다.
- `validation.in_objective` 옵션을 사용하면 지정한 주기로 경량 Walk-forward 점수를
  산출해 본 점수와 가중 평균하거나, 로그에 별도 기록하여 과최적화를 조기 감지할 수
  있습니다.
- Optimisation state is stored in `studies/<symbol>_<ltf>_<htf>.db` (SQLite + heartbeat)
  so 중단 후 재실행 시 자동으로 이어달리기(warm start)가 됩니다. JSONL/YAML 로그는
  최적화 도중 예기치 못한 종료가 발생해도 남도록 `trials/` 폴더에 즉시 기록됩니다.
