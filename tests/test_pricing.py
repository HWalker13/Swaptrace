"""Unit tests for :mod:`swaptrace.pricing` -- the bundled table, overrides, and
the graceful-miss cost estimator.
"""

import json

import pytest

from swaptrace.pricing import estimate_cost_usd, load_overrides


def test_known_model_computes_expected_cost():
    # gpt-4o-mini @ (0.15, 0.60) USD / 1M tokens, 1000 prompt + 500 completion:
    # (1000 * 0.15 + 500 * 0.60) / 1_000_000 == 0.00045
    assert estimate_cost_usd("gpt-4o-mini", 1000, 500) == pytest.approx(0.00045)


def test_unknown_model_returns_none():
    assert estimate_cost_usd("no-such-model", 100, 100) is None


def test_none_token_counts_return_none():
    assert estimate_cost_usd("gpt-4o-mini", None, 50) is None
    assert estimate_cost_usd("gpt-4o-mini", 50, None) is None


def test_override_beats_bundled_rate():
    overrides = {"gpt-4o-mini": (1.0, 2.0)}
    # (1000 * 1.0 + 500 * 2.0) / 1_000_000 == 0.002
    got = estimate_cost_usd("gpt-4o-mini", 1000, 500, overrides=overrides)
    assert got == pytest.approx(0.002)
    assert got != pytest.approx(0.00045)  # not the bundled rate


def test_override_adds_new_model():
    overrides = {"custom-x": (10.0, 20.0)}
    assert estimate_cost_usd("custom-x", 100, 100) is None  # unknown without it
    # (100 * 10.0 + 100 * 20.0) / 1_000_000 == 0.003
    assert estimate_cost_usd("custom-x", 100, 100, overrides=overrides) == pytest.approx(0.003)


def test_partial_override_keeps_other_bundled_models():
    overrides = {"gpt-4o-mini": (99.0, 99.0)}
    # claude-haiku-4-5 @ (1.00, 5.00): (1000 * 1.0 + 1000 * 5.0) / 1_000_000 == 0.006
    got = estimate_cost_usd("claude-haiku-4-5", 1000, 1000, overrides=overrides)
    assert got == pytest.approx(0.006)


def test_load_overrides_missing_path_returns_empty(tmp_path):
    assert load_overrides(tmp_path / "nope.json") == {}


def test_load_overrides_parses_json_file(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"foo": [0.5, 1.5], "bar": [2, 4]}), encoding="utf-8")
    result = load_overrides(path)
    assert result == {"foo": (0.5, 1.5), "bar": (2.0, 4.0)}
    for rates in result.values():
        assert isinstance(rates, tuple) and len(rates) == 2
        assert all(isinstance(r, float) for r in rates)
