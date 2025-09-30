import json
from types import SimpleNamespace

import optuna
import pytest

from optuna.distributions import FloatDistribution

from optimize.llm import generate_llm_candidates


class _DummyModels:
    def __init__(self, payload):
        self._payload = payload
        self.last_model = None
        self.last_contents = None

    def generate_content(self, model: str, contents: str) -> SimpleNamespace:
        self.last_model = model
        self.last_contents = contents
        return SimpleNamespace(text=json.dumps(self._payload))


class _DummyClient:
    def __init__(self, payload):
        self.models = _DummyModels(payload)


class _DummyGenAI:
    def __init__(self, payload):
        self._payload = payload
        self.last_client = None

    def Client(self, api_key: str) -> _DummyClient:
        client = _DummyClient(self._payload)
        client.api_key = api_key
        self.last_client = client
        return client


def _install_dummy_genai(monkeypatch: pytest.MonkeyPatch, payload):
    dummy = _DummyGenAI(payload)
    monkeypatch.setattr("optimize.llm.genai", dummy)
    return dummy


def test_generate_llm_candidates_single_objective(monkeypatch: pytest.MonkeyPatch):
    trials = [
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            params={"lr": 0.01},
            distributions={"lr": FloatDistribution(0.0, 1.0)},
            value=value,
        )
        for value in (0.25, 0.9, 0.6)
    ]
    space = {"lr": {"type": "float", "min": 0.0, "max": 1.0}}
    payload = [{"lr": 0.4}, {"lr": 0.2}]
    dummy_genai = _install_dummy_genai(monkeypatch, payload)

    result = generate_llm_candidates(
        space,
        trials,
        {"enabled": True, "api_key": "token", "top_n": 2, "count": 2},
    )

    assert result == [{"lr": 0.4}, {"lr": 0.2}]
    assert dummy_genai.last_client is not None
    contents = dummy_genai.last_client.models.last_contents
    json_blob = contents.split("objective values (higher is better):\n", 1)[1]
    trials_json = json_blob.split("\n\n", 1)[0]
    parsed = json.loads(trials_json)
    assert all("value" in entry for entry in parsed)
    assert all("values" not in entry for entry in parsed)


def test_generate_llm_candidates_multi_objective(monkeypatch: pytest.MonkeyPatch):
    trials = [
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            params={"lr": 0.01 * idx},
            distributions={"lr": FloatDistribution(0.0, 1.0)},
            values=values,
        )
        for idx, values in enumerate(
            ((0.8, 0.6), (0.7, 0.2), (0.9, 0.4)),
            start=1,
        )
    ]
    space = {"lr": {"type": "float", "min": 0.0, "max": 1.0}}
    payload = [{"lr": 0.3}]
    dummy_genai = _install_dummy_genai(monkeypatch, payload)

    result = generate_llm_candidates(
        space,
        trials,
        {
            "enabled": True,
            "api_key": "token",
            "top_n": 3,
            "count": 1,
            "objective_index": 1,
            "objective_direction": "minimize",
        },
    )

    assert result == [{"lr": 0.3}]
    assert dummy_genai.last_client is not None
    contents = dummy_genai.last_client.models.last_contents
    assert "lower is better" in contents
    json_blob = contents.split("objective values (lower is better):\n", 1)[1]
    trials_json = json_blob.split("\n\n", 1)[0]
    parsed = json.loads(trials_json)
    assert all("values" in entry for entry in parsed)
    assert parsed[0]["values"][1] == pytest.approx(0.2)
