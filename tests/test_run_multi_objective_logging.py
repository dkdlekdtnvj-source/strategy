import json

import pytest

pytest.importorskip("pandas")

import pandas as pd
import yaml

from optimize.metrics import Trade
from optimize.run import DatasetSpec, optimisation_loop


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multi_objective_logging_and_snapshots(tmp_path, monkeypatch):
    index = pd.date_range("2024-01-01", periods=3, freq="1min")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        },
        index=index,
    )
    dataset = DatasetSpec(
        symbol="BINANCE:TESTUSDT",
        timeframe="1m",
        start="2024-01-01",
        end="2024-01-02",
        df=df,
        htf=None,
        htf_timeframe=None,
        source_symbol="BINANCE:TESTUSDT",
    )

    call_counter = {"count": 0}

    def _fake_backtest(df, params, fees, risk, *, htf_df=None):
        call_counter["count"] += 1
        base = 0.01 * call_counter["count"]
        returns = pd.Series([base, -base / 2, 0.0], index=index)
        trade = Trade(
            entry_time=index[0],
            exit_time=index[-1],
            direction="long",
            size=1.0,
            entry_price=100.0,
            exit_price=100.0 * (1 + base),
            profit=base,
            return_pct=base,
            mfe=base,
            mae=-base / 2,
            bars_held=len(index),
        )
        return {
            "Returns": returns,
            "TradesList": [trade],
            "Valid": True,
        }

    monkeypatch.setattr("optimize.run.run_backtest", _fake_backtest)

    params_cfg = {
        "search": {
            "multi_objective": True,
            "n_trials": 2,
            "algo": "random",
            "pruner": "nop",
            "seed": 1,
        },
        "space": {},
    }

    objectives = ["NetProfit", "Sharpe"]

    result = optimisation_loop(
        [dataset],
        params_cfg,
        objectives,
        fees={},
        risk={},
        log_dir=tmp_path,
    )

    assert result["multi_objective"] is True

    trial_log_path = tmp_path / "trials.jsonl"
    assert trial_log_path.exists()
    entries = [json.loads(line) for line in trial_log_path.read_text().splitlines() if line]
    assert entries, "trials.jsonl에 최소 한 건 이상의 기록이 필요합니다"
    for entry in entries:
        assert isinstance(entry["value"], list)
        assert len(entry["value"]) == len(objectives)

    best_yaml_path = tmp_path / "best.yaml"
    assert best_yaml_path.exists()
    snapshot = yaml.safe_load(best_yaml_path.read_text())
    assert isinstance(snapshot["best_value"], list)
    assert len(snapshot["best_value"]) == len(objectives)
    assert all(isinstance(val, float) for val in snapshot["best_value"])
