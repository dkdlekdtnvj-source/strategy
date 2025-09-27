import numpy as np
import pandas as pd

from optimize.context import RegimeBandit, compute_regime_features, label_regime
from optimize.pbt import generate_pbt_mutations
from optimize.search_spaces import build_space


def test_generate_pbt_mutations_stays_within_bounds() -> None:
    space = build_space(
        {
            "oscLen": {"type": "int", "min": 10, "max": 30},
            "thr": {"type": "float", "min": 0.5, "max": 1.5},
            "useFilter": {"type": "bool"},
            "mode": {"type": "choice", "values": ["a", "b", "c"]},
        }
    )
    seeds = [{"oscLen": 12, "thr": 0.9, "useFilter": True, "mode": "a"}]
    rng = np.random.default_rng(42)

    mutations = generate_pbt_mutations(seeds, space, scale=0.2, epsilon=0.0, rng=rng)

    assert len(mutations) == 1
    mutated = mutations[0]
    assert 10 <= mutated["oscLen"] <= 30
    assert 0.5 <= mutated["thr"] <= 1.5
    assert mutated["mode"] in {"a", "b", "c"}
    # With epsilon=0 we expect a single neighbour that differs from the seed.
    assert mutated != seeds[0]


def test_regime_bandit_serialisation_roundtrip() -> None:
    bandit = RegimeBandit(exploration=0.2)
    bandit.update("regime_a", "arm_1", 1.0)
    bandit.update("regime_a", "arm_1", 2.0)
    bandit.update("regime_a", "arm_2", 0.5)

    chosen = bandit.select("regime_a")
    assert chosen in {"arm_1", "arm_2"}

    restored = RegimeBandit.from_dict(bandit.to_dict())
    assert restored.select("regime_a") == bandit.select("regime_a")


def test_compute_regime_features_and_label() -> None:
    index = pd.date_range("2024-01-01", periods=200, freq="1min", tz="UTC")
    close = np.linspace(100, 110, num=200)
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=index)

    features = compute_regime_features(df)
    assert {"atr_pct", "trend_slope", "session"}.issubset(features.keys())

    regime = label_regime(features)
    assert regime.startswith("vol_")
    assert "|trend_" in regime
    assert "|session_" in regime
