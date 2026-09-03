"""Tests for :mod:`swaptrace.cli` -- the ``swaptrace query`` command.

``main()`` is called directly (not via subprocess) with ``capsys`` capturing
output. The 5-trace dataset mirrors Session 11's manual verification: built
through the real ``run_cascade`` flow + ``storage.append_trace``, never
hand-written JSON, so costs are the real values from the Session 7 pricing table.
"""

import pytest

from swaptrace.cli import main, trace_matches
from swaptrace.core import Attempt, Trace
from swaptrace.storage import append_trace
from test_core import run_cascade  # pytest puts tests/ on sys.path

# gpt-4o-mini @ (0.15, 0.60)/1M, run_cascade's 10 prompt + 20 completion tokens
GPT_COST = (10 * 0.15 + 20 * 0.60) / 1_000_000            # 1.35e-5
# claude-haiku-4-5 @ (1.00, 5.00)/1M, same token counts
HAIKU_COST = (10 * 1.00 + 20 * 5.00) / 1_000_000          # 1.1e-4


@pytest.fixture
def trace_file(tmp_path):
    """Writes the 5-trace dataset to a JSONL file; returns (path, traces dict)."""
    path = tmp_path / ".swaptrace" / "traces.jsonl"
    traces = {}
    traces["t1"] = run_cascade(["ok"], providers=["groq"], model="gpt-4o-mini")
    traces["t2"] = run_cascade(
        ["retryable", "ok"], providers=["groq", "openai"], model="gpt-4o-mini"
    )
    traces["t3"] = run_cascade(
        ["retryable", "retryable", "retryable"],
        providers=["groq", "openai", "anthropic"],
    )
    t4 = Trace()
    with pytest.raises(ValueError):
        run_cascade(["fatal"], providers=["groq"], model="gpt-4o-mini", trace=t4)
    traces["t4"] = t4
    traces["t5"] = run_cascade(["ok"], providers=["anthropic"], model="claude-haiku-4-5")

    for trace in traces.values():
        append_trace(trace, path)
    return path, traces


def _query(path, *args):
    return main(["query", "--path", str(path), *args])


# --------------------------------------------------------------------------- #
def test_no_filters_shows_all(trace_file, capsys):
    path, traces = trace_file
    _query(path)
    out = capsys.readouterr().out
    for trace in traces.values():
        assert trace.trace_id[:8] in out
    assert "5 of 5 trace(s)." in out


def test_filter_by_provider(trace_file, capsys):
    path, traces = trace_file
    _query(path, "--provider", "groq")  # t1, t2, t3, t4 have a groq attempt; t5 does not
    out = capsys.readouterr().out
    assert "4 of 5 trace(s)." in out
    assert traces["t5"].trace_id[:8] not in out
    assert traces["t1"].trace_id[:8] in out


def test_filter_by_status(trace_file, capsys):
    path, traces = trace_file
    _query(path, "--status", "success")  # t1, t2, t5 have a success attempt
    out = capsys.readouterr().out
    assert "3 of 5 trace(s)." in out
    for key in ("t1", "t2", "t5"):
        assert traces[key].trace_id[:8] in out
    for key in ("t3", "t4"):
        assert traces[key].trace_id[:8] not in out


def test_filter_by_min_cost(trace_file, capsys):
    path, traces = trace_file
    # HAIKU_COST (~1.1e-4) passes; GPT_COST (~1.35e-5) does not
    _query(path, "--min-cost", "0.0001")
    out = capsys.readouterr().out
    assert "1 of 5 trace(s)." in out
    assert traces["t5"].trace_id[:8] in out
    assert traces["t1"].trace_id[:8] not in out


def test_combined_filters_matching(trace_file, capsys):
    path, traces = trace_file
    # t1: groq attempt + success attempt.  t2: groq attempt + (openai) success attempt.
    _query(path, "--provider", "groq", "--status", "success")
    out = capsys.readouterr().out
    assert "2 of 5 trace(s)." in out
    assert traces["t1"].trace_id[:8] in out
    assert traces["t2"].trace_id[:8] in out


def test_combined_filters_and_excludes_partial_match(trace_file, capsys):
    path, traces = trace_file
    # t3 has an anthropic attempt but NO success attempt -> AND must exclude it.
    # t5 has both -> included. This is what proves AND, not OR.
    _query(path, "--provider", "anthropic", "--status", "success")
    out = capsys.readouterr().out
    assert "1 of 5 trace(s)." in out
    assert traces["t5"].trace_id[:8] in out
    assert traces["t3"].trace_id[:8] not in out


def test_invalid_status_value(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["query", "--status", "bogus"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'bogus'" in err
    assert "--status" in err


def test_nonexistent_path(tmp_path, capsys):
    rc = main(["query", "--path", str(tmp_path / "nope.jsonl")])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "No traces found."


def test_existing_path_zero_matches(trace_file, capsys):
    path, _ = trace_file
    rc = main(["query", "--path", str(path), "--provider", "no-such-provider"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "No traces found."


def test_output_line_format(trace_file, capsys):
    path, traces = trace_file
    _query(path)
    out = capsys.readouterr().out
    t5 = traces["t5"]
    line = next(ln for ln in out.splitlines() if t5.trace_id[:8] in ln)
    assert "success" in line
    assert "anthropic" in line
    assert "1 attempt(s)" in line
    assert f"${t5.total_cost_usd:.6f}" in line  # e.g. "$0.000110"


def test_trace_matches_pure():
    trace = Trace(
        attempts=[
            Attempt(provider="groq", model="m", outcome="retryable_error"),
            Attempt(provider="openai", model="m", outcome="success"),
        ],
        total_cost_usd=0.005,
    )
    assert trace_matches(trace)  # no filters
    assert trace_matches(trace, provider="groq")
    assert not trace_matches(trace, provider="anthropic")
    assert trace_matches(trace, status="success")
    assert not trace_matches(trace, status="non_retryable_error")
    assert trace_matches(trace, min_cost=0.005)
    assert trace_matches(trace, min_cost=0.001)
    assert not trace_matches(trace, min_cost=0.01)
    # AND across filters
    assert trace_matches(trace, provider="groq", status="success")
    assert not trace_matches(trace, provider="anthropic", status="success")
    # the `total_cost_usd is None` guard in the min_cost branch
    assert not trace_matches(Trace(total_cost_usd=None), min_cost=0.0)


def test_main_returns_zero_on_success(trace_file):
    path, _ = trace_file
    assert main(["query", "--path", str(path)]) == 0
