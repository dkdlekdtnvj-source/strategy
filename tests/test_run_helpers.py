import pytest

from optimize.run import _normalise_periods, _resolve_symbol_entry, parse_args


def test_normalise_periods_returns_configured_entries():
    periods_cfg = [{"from": "2023-01-01", "to": "2023-06-30"}]
    base = {"from": "2022-01-01", "to": "2022-12-31"}

    result = _normalise_periods(periods_cfg, base)

    assert result == [{"from": "2023-01-01", "to": "2023-06-30"}]


def test_normalise_periods_falls_back_to_base_when_empty():
    base = {"from": "2022-01-01", "to": "2022-12-31"}

    result = _normalise_periods([], base)

    assert result == [{"from": "2022-01-01", "to": "2022-12-31"}]


@pytest.mark.parametrize(
    "payload",
    [
        [{"from": "2023-01-01"}],
        [{"to": "2023-06-30"}],
        ["2023-01-01/2023-06-30"],
    ],
)
def test_normalise_periods_raises_for_invalid_entries(payload):
    base = {}

    with pytest.raises(ValueError):
        _normalise_periods(payload, base)


def test_resolve_symbol_entry_uses_alias_map():
    alias, resolved = _resolve_symbol_entry(
        "BINANCE:XPLUSDT", {"BINANCE:XPLUSDT": "BINANCE:XPLAUSDT"}
    )

    assert alias == "BINANCE:XPLUSDT"
    assert resolved == "BINANCE:XPLAUSDT"


def test_resolve_symbol_entry_accepts_mapping_definition():
    entry = {"alias": "BINANCE:ASTERUSDT", "symbol": "BINANCE:ASTRUSDT"}
    alias, resolved = _resolve_symbol_entry(entry, {})

    assert alias == "BINANCE:ASTERUSDT"
    assert resolved == "BINANCE:ASTRUSDT"


def test_parse_args_accepts_trial_override():
    args = parse_args(["--n-trials", "25"])

    assert args.n_trials == 25
