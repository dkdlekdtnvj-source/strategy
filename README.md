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
datafeed/binance_client.py     # Binance downloader with retries
datafeed/cache.py              # CSV cache layer
reports/                       # Optimisation output (ignored by git)
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
2. Adjust optimisation inputs in `config/params.yaml`. The sample profile exposes
   WaveTrend/ATR thresholds, higher-timeframe confirmation, fade-mode selection,
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
  `heatmap.png`). The `results_datasets.csv` file is especially useful for answering
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
  can optionally re-score the top trials by walk-forward OOS mean.
