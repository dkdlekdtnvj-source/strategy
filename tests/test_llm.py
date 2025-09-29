import json
from types import SimpleNamespace

import pytest

optuna = pytest.importorskip("optuna")

from optimize import llm


class DummyModels:
    def generate_content(self, model: str, contents: str):
        # 다중 목적 정보가 프롬프트에 포함되었는지 확인
        assert '"values"' in contents
        assert '"direction": "minimize"' in contents
        payload = json.dumps([{"alpha": 3}, {"alpha": 5}])
        return SimpleNamespace(text=payload)


class DummyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = DummyModels()


@pytest.fixture
def multi_objective_trials():
    distributions = {
        "alpha": optuna.distributions.IntUniformDistribution(low=0, high=10),
    }
    single = optuna.trial.create_trial(
        params={"alpha": 2},
        distributions=distributions,
        value=1.5,
        state=optuna.trial.TrialState.COMPLETE,
    )
    multi = optuna.trial.create_trial(
        params={"alpha": 4},
        distributions=distributions,
        values=[0.4, -0.1],
        state=optuna.trial.TrialState.COMPLETE,
    )
    return [single, multi]


def test_generate_llm_candidates_handles_multi_objective(monkeypatch, multi_objective_trials):
    monkeypatch.setattr(llm, "genai", SimpleNamespace(Client=DummyClient))

    space = {
        "alpha": {"type": "int", "min": 0, "max": 10},
    }
    config = {"enabled": True, "api_key": "dummy", "top_n": 5, "count": 2}
    objectives = [
        {"name": "NetProfit", "weight": 1.0, "direction": "maximize"},
        {"name": "MaxDD", "weight": 1.0, "direction": "minimize"},
    ]

    candidates = llm.generate_llm_candidates(space, multi_objective_trials, config, objectives)

    assert candidates == [{"alpha": 3}, {"alpha": 5}]
