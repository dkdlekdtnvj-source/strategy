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
   WaveTrend/ATR thresholds, higher-timeframe confirmation, exit modules (ATR trail,
   pivot stop, breakeven, time stop), and penalty settings. You can also pre-define
   `overrides` to pin any parameter on/off before a run and enable Top-K walk-forward
   re-ranking via `search.top_k`.
3. (Optional) Configure multi-symbol / multi-timeframe sweeps and shared risk
   assumptions in `config/backtest.yaml`. Each trial is evaluated across every
   symbol × timeframe × period combination to avoid overfitting to a single
   dataset.
4. Launch the optimiser non-interactively or with prompts:

   ```bash
   python -m optimize.run --params config/params.yaml --backtest config/backtest.yaml
   ```

   or

   ```bash
   python -m optimize.run --interactive
   ```

   The interactive mode lets you pick the symbol, lower timeframe, higher timeframe,
   evaluation window, leverage, position size, and boolean filters (e.g. HTF sync,
   ATR trail, pivot stops) from the terminal. Command-line overrides are also
   available:

   - `--symbol`, `--timeframe`, `--htf`, `--start`, `--end`
   - `--leverage`, `--qty-pct`
   - `--enable name1,name2`, `--disable name3`
   - `--top-k 10` to re-rank the best Optuna trials by walk-forward out-of-sample
     performance.

   Outputs are written to `reports/` (`results.csv`, `results_datasets.csv`,
   `best.json`, `heatmap.png`). The `best.json` payload now includes the Top-K
   candidate summary with their walk-forward scores.

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
