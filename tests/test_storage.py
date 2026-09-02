"""Round-trip tests for :mod:`swaptrace.storage` -- write then read is lossless.

Traces are built through the real ``core.py`` context-manager flow (reusing the
``run_cascade`` helper from ``test_core``), never hand-constructed dataclasses.
"""

import json

import pytest

from swaptrace import Trace
from swaptrace.storage import (
    TraceJSONEncoder,
    append_trace,
    iter_traces,
    read_traces,
    trace_from_dict,
    trace_to_dict,
)
from test_core import run_cascade  # pytest puts tests/ on sys.path

SUCCESS = (["retryable", "ok"], ["groq", "openai"])
EXHAUSTED = (["retryable", "retryable", "retryable"], ["groq", "openai", "anthropic"])


def _success_trace():
    return run_cascade(SUCCESS[0], providers=SUCCESS[1])


def _exhausted_trace():
    return run_cascade(EXHAUSTED[0], providers=EXHAUSTED[1])


def _non_retryable_trace():
    """A completed, exhausted trace whose one attempt raised a non-retryable error."""
    trace = Trace()
    with pytest.raises(ValueError):
        run_cascade(["fatal"], providers=["groq"], trace=trace)
    return trace


def test_round_trip_equality(tmp_path):
    """A written trace reads back equal by full dataclass equality."""
    written = _success_trace()
    path = append_trace(written, tmp_path / "traces.jsonl")
    (read,) = read_traces(path)
    assert read == written


def test_order_preserved(tmp_path):
    """Traces read back in the exact order they were appended."""
    a, b = _success_trace(), _exhausted_trace()
    path = tmp_path / "traces.jsonl"
    append_trace(a, path)
    append_trace(b, path)
    assert read_traces(path) == [a, b]


def test_nonexistent_file_yields_nothing(tmp_path):
    """Reading a path that was never created is empty, not an error."""
    missing = tmp_path / "never_written.jsonl"
    assert read_traces(missing) == []
    assert list(iter_traces(missing)) == []


def test_empty_file_yields_nothing(tmp_path):
    """An existing but empty file also yields nothing."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert read_traces(empty) == []


def test_none_fields_survive(tmp_path):
    """None round-trips as real None, not the string "None"."""
    written = _exhausted_trace()
    assert written.winning_provider is None
    assert written.attempts[0].cost_usd is None
    (read,) = read_traces(append_trace(written, tmp_path / "t.jsonl"))
    assert read.winning_provider is None
    assert read.attempts[0].cost_usd is None


def test_exact_datetime_equality(tmp_path):
    """No precision loss through the isoformat round-trip."""
    from datetime import datetime

    written = _success_trace()
    (read,) = read_traces(append_trace(written, tmp_path / "t.jsonl"))
    assert isinstance(read.started_at, datetime)
    assert read.started_at == written.started_at
    assert read.ended_at == written.ended_at
    assert read.attempts[0].started_at == written.attempts[0].started_at
    assert read.attempts[0].ended_at == written.attempts[0].ended_at


def test_pure_round_trip():
    """trace_from_dict(trace_to_dict(t)) == t, with zero file I/O."""
    t = _success_trace()
    assert trace_from_dict(trace_to_dict(t)) == t


def test_heterogeneous_traces_one_file(tmp_path):
    """Different trace shapes coexist in one file and all read back."""
    path = tmp_path / "mixed.jsonl"
    append_trace(_success_trace(), path)
    append_trace(_exhausted_trace(), path)
    append_trace(_non_retryable_trace(), path)
    read = read_traces(path)
    assert len(read) == 3
    assert [t.final_status for t in read] == ["success", "exhausted", "exhausted"]


def test_encoder_raises_on_unsupported_type():
    """TraceJSONEncoder fails loudly on a type it doesn't handle."""
    with pytest.raises(TypeError):
        json.dumps(object(), cls=TraceJSONEncoder)
    with pytest.raises(TypeError):
        json.dumps({1, 2, 3}, cls=TraceJSONEncoder)


def test_unrecognized_literal_raises():
    """A corrupted Literal value is rejected rather than silently loaded."""
    good = trace_to_dict(_success_trace())

    bad_status = {**good, "final_status": "garbage"}
    with pytest.raises(ValueError):
        trace_from_dict(bad_status)

    bad_outcome = trace_to_dict(_success_trace())
    bad_outcome["attempts"][0]["outcome"] = "weird"
    with pytest.raises(ValueError):
        trace_from_dict(bad_outcome)
