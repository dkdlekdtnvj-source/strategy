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
2. Adjust optimisation inputs in `config/params.yaml`. The example enables Bayesian
   optimisation across WaveTrend/ATR thresholds, HTF synchronisation, and indicator
   toggles while exposing leverage, commission, slippage, liquidation buffer, and
   walk-forward constraints. Risk gates such as minimum trades, minimum average
   holding period, and maximum consecutive losses feed directly into the scoring
   penalties.
3. (Optional) Configure multi-symbol / multi-timeframe sweeps and shared risk
   assumptions in `config/backtest.yaml`. Each trial is evaluated across every
   symbol × timeframe × period combination to avoid overfitting to a single
   dataset.
4. Launch the optimiser:

   ```bash
   python -m optimize.run --params config/params.yaml --backtest config/backtest.yaml
   ```

   Outputs will be saved in `reports/` as `results.csv`, `results_datasets.csv`,
   `best.json`, and `heatmap.png`.

## Testing

```bash
pytest
```

## Notes

- The Pine script uses `process_orders_on_close=true`, `request.security(..., lookahead_off)`,
  and confirmed-bar guards to prevent repainting. Commission, slippage, leverage, and
  liquidation buffers are exposed as optimiser-friendly inputs.
- Walk-forward parameters (train/test window length and step) can be configured under
  `walk_forward` in `config/params.yaml`. The Python model mirrors the Pine signals
  with confirmed higher-timeframe alignment.
- Metrics include Net Profit, Max Drawdown, Sortino, Sharpe, Profit Factor, Win Rate,
  weekly net profit, expectancy, RR, average MFE/MAE, and average holding period. The
  optimiser combines weighted objectives with penalties for breaching the risk gates.
