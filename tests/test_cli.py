"""Tests for :mod:`swaptrace.cli` -- the ``swaptrace query`` command.

``main()`` is called directly (not via subprocess) with ``capsys`` capturing
output. The 5-trace dataset mirrors Session 11's manual verification: built
through the real ``run_cascade`` flow + ``storage.append_trace``, never
hand-written JSON, so costs are the real values from the Session 7 pricing table.
"""

import pytest

from swaptrace.cli import compare_providers, main, trace_matches
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


# --------------------------------------------------------------------------- #
# report --compare-providers
# --------------------------------------------------------------------------- #
def _a(provider, outcome, *, latency_ms=1.0, cost_usd=None):
    return Attempt(
        provider=provider, model="m", outcome=outcome,
        latency_ms=latency_ms, cost_usd=cost_usd,
    )


def _t(*attempts):
    return Trace(attempts=list(attempts))


# ---- pure compare_providers(), no file I/O -------------------------------- #
def test_compare_providers_always_succeeds():
    stats = compare_providers([_t(_a("groq", "success")), _t(_a("groq", "success"))])
    assert len(stats) == 1
    s = stats[0]
    assert s.provider == "groq"
    assert s.total_attempts == 2 and s.successes == 2
    assert s.success_rate == 1.0


def test_compare_providers_groups_across_all_traces():
    traces = [
        _t(_a("groq", "retryable_error")),
        _t(_a("groq", "success"), _a("openai", "success")),
        _t(_a("groq", "success")),
    ]
    by = {s.provider: s for s in compare_providers(traces)}
    assert by["groq"].total_attempts == 3  # one per trace, not grouped per-trace
    assert by["groq"].successes == 2
    assert by["groq"].success_rate == 2 / 3
    assert by["openai"].total_attempts == 1 and by["openai"].successes == 1


def test_compare_providers_never_succeeds():
    stats = compare_providers(
        [_t(_a("groq", "retryable_error")), _t(_a("groq", "non_retryable_error"))]
    )
    s = stats[0]
    assert s.successes == 0
    assert s.success_rate == 0.0
    assert s.avg_cost_per_success_usd is None  # not a ZeroDivisionError


def test_compare_providers_avg_latency_spans_failures():
    # groq: one failed attempt (100ms) + one success (200ms). Mean over BOTH is
    # 150; a success-only mean would be 200.
    stats = compare_providers(
        [_t(_a("groq", "retryable_error", latency_ms=100.0),
            _a("groq", "success", latency_ms=200.0))]
    )
    s = stats[0]
    assert s.total_attempts == 2 and s.successes == 1
    assert s.avg_latency_ms == 150.0


def test_compare_providers_sort_distinct_rates():
    traces = [
        _t(_a("low", "retryable_error"), _a("low", "retryable_error"), _a("low", "success")),  # 1/3
        _t(_a("high", "success"), _a("high", "success")),                                      # 2/2
        _t(_a("mid", "success"), _a("mid", "retryable_error")),                                # 1/2
    ]
    assert [s.provider for s in compare_providers(traces)] == ["high", "mid", "low"]


def test_compare_providers_sort_tie_break_is_alphabetical():
    # zebra and alpha each have exactly 1 success out of 3 attempts -> identical
    # success_rate (1/3, a repeating binary fraction computed the same way both
    # times, so genuinely equal, not merely equal-looking). Tie -> name order.
    traces = [
        _t(_a("zebra", "success"), _a("zebra", "retryable_error"), _a("zebra", "retryable_error")),
        _t(_a("alpha", "success"), _a("alpha", "retryable_error"), _a("alpha", "retryable_error")),
    ]
    stats = compare_providers(traces)
    by = {s.provider: s for s in stats}
    assert by["alpha"].success_rate == by["zebra"].success_rate  # genuinely equal
    assert by["alpha"].success_rate == 1 / 3
    assert [s.provider for s in stats] == ["alpha", "zebra"]


def test_compare_providers_empty_input():
    assert compare_providers([]) == []


# ---- CLI level: main(["report", ...]) ----------------------------------- #
@pytest.fixture
def report_file(tmp_path):
    """alpha 100% / beta 50% / gamma 0% (never succeeds -> $/success is '-')."""
    path = tmp_path / "traces.jsonl"
    traces = [
        _t(_a("alpha", "success", latency_ms=10.0, cost_usd=0.001)),
        _t(_a("alpha", "success", latency_ms=20.0, cost_usd=0.001),
           _a("beta", "retryable_error", latency_ms=30.0)),
        _t(_a("beta", "success", latency_ms=40.0, cost_usd=0.002),
           _a("gamma", "retryable_error", latency_ms=50.0)),
        _t(_a("gamma", "non_retryable_error", latency_ms=60.0)),
    ]
    for trace in traces:
        append_trace(trace, path)
    return path


def test_report_without_compare_providers_flag(tmp_path, capsys):
    rc = main(["report", "--path", str(tmp_path / "whatever.jsonl")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "Try --compare-providers." in captured.out
    assert captured.err == ""  # not a traceback


def test_report_nonexistent_path(tmp_path, capsys):
    rc = main(["report", "--path", str(tmp_path / "nope.jsonl"), "--compare-providers"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "No traces found."


def test_report_empty_file(tmp_path, capsys):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    rc = main(["report", "--path", str(path), "--compare-providers"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "No traces found."


def test_report_output_format(report_file, capsys):
    main(["report", "--path", str(report_file), "--compare-providers"])
    lines = capsys.readouterr().out.splitlines()

    header = lines[0]
    for col in ("PROVIDER", "ATTEMPTS", "SUCCESSES", "SUCCESS%",
                "AVG LATENCY", "TOTAL COST", "$/SUCCESS"):
        assert col in header

    assert [ln.split()[0] for ln in lines[1:]] == ["alpha", "beta", "gamma"]

    alpha = next(ln for ln in lines if ln.startswith("alpha"))
    assert "100.0%" in alpha and "$0.001000" in alpha

    gamma = next(ln for ln in lines if ln.startswith("gamma"))
    assert "0.0%" in gamma
    assert "$0.000000" in gamma          # TOTAL COST column
    assert gamma.rstrip().endswith("-")  # $/SUCCESS column is '-', not $0.000000


def test_report_returns_zero_on_success(report_file):
    assert main(["report", "--path", str(report_file), "--compare-providers"]) == 0
