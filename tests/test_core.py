"""Lifecycle regression tests for :mod:`swaptrace.core`.

These lock in the five scenarios hand-verified in Session 3: immediate success,
one retryable failure then success, full exhaustion (all retryable), a
non-retryable failure that raises out of the trace, and the defensive path where
an unclassified exception leaves an attempt block.

swaptrace is framework-agnostic, so the "retryable" exception type and the
provider-call helper are defined locally here -- nothing is imported from any
LLM framework.
"""

import pytest

from swaptrace import Trace

PROMPT_TOKENS = 10
COMPLETION_TOKENS = 20


class RetryableError(Exception):
    """Stand-in for a caller-owned "try the next provider" exception."""


def _call(behavior):
    """Fake provider call. ``behavior`` drives the outcome."""
    if behavior == "ok":
        return {"text": "ok"}
    if behavior == "retryable":
        raise RetryableError("rate limited")
    if behavior == "fatal":
        raise ValueError("bad request")
    if behavior == "boom":
        raise KeyError("unexpected internal state")
    raise AssertionError(f"unknown behavior: {behavior!r}")


def run_cascade(behaviors, providers=None, *, classify_unexpected=True, trace=None):
    """Run the exact Session 3 usage pattern over a list of provider behaviors.

    Returns the :class:`Trace`. Pass ``trace=Trace()`` when the cascade is
    expected to raise, so the instance can still be inspected afterwards.
    """
    if trace is None:
        trace = Trace()
    if providers is None:
        providers = [f"provider-{i}" for i in range(len(behaviors))]

    with trace:
        for provider, behavior in zip(providers, behaviors):
            with trace.attempt(provider=provider, model="test-model") as attempt:
                try:
                    response = _call(behavior)
                    attempt.record_success(
                        response,
                        prompt_tokens=PROMPT_TOKENS,
                        completion_tokens=COMPLETION_TOKENS,
                    )
                    break
                except RetryableError as exc:
                    attempt.record_failure(exc, retryable=True)
                    continue
                except Exception as exc:  # noqa: BLE001 - mirrors the caller pattern
                    if classify_unexpected:
                        attempt.record_failure(exc, retryable=False)
                    raise
    return trace


class TestImmediateSuccess:
    """First provider succeeds on the first try."""

    def test_final_status_and_winner(self):
        trace = run_cascade(["ok"], providers=["groq"])
        assert trace.final_status == "success"
        assert trace.winning_provider == "groq"
        assert trace.retry_count == 0
        assert len(trace.attempts) == 1

    def test_rollup_totals(self):
        trace = run_cascade(["ok"], providers=["groq"])
        assert trace.total_cost_usd == 0.0
        assert trace.attempts[0].cost_usd is None
        assert trace.total_latency_ms == sum(a.latency_ms for a in trace.attempts)

    def test_attempt_fields(self):
        attempt = run_cascade(["ok"], providers=["groq"]).attempts[0]
        assert attempt.outcome == "success"
        assert attempt.prompt_tokens == PROMPT_TOKENS
        assert attempt.completion_tokens == COMPLETION_TOKENS
        assert attempt.error_type is None
        assert attempt.error_message is None
        assert attempt.started_at is not None
        assert attempt.ended_at is not None
        assert attempt.latency_ms is not None and attempt.latency_ms >= 0.0


class TestRetryableThenSuccess:
    """First provider fails retryably, second succeeds."""

    def test_final_status_and_winner(self):
        trace = run_cascade(["retryable", "ok"], providers=["groq", "openai"])
        assert trace.final_status == "success"
        assert trace.winning_provider == "openai"
        assert trace.retry_count == 1
        assert len(trace.attempts) == 2

    def test_first_attempt_recorded_retryable(self):
        first = run_cascade(["retryable", "ok"], providers=["groq", "openai"]).attempts[0]
        assert first.outcome == "retryable_error"
        assert first.error_type == "RetryableError"
        assert first.error_message

    def test_second_attempt_is_clean_success(self):
        second = run_cascade(["retryable", "ok"], providers=["groq", "openai"]).attempts[1]
        assert second.outcome == "success"
        assert second.error_type is None
        assert second.error_message is None

    def test_total_latency_sums_all_attempts(self):
        trace = run_cascade(["retryable", "ok"], providers=["groq", "openai"])
        assert len(trace.attempts) == 2
        assert trace.total_latency_ms == sum(a.latency_ms for a in trace.attempts)


class TestFullExhaustionAllRetryable:
    """Every provider fails retryably; no exception escapes the trace."""

    BEHAVIORS = ["retryable", "retryable", "retryable"]
    PROVIDERS = ["groq", "openai", "anthropic"]

    def test_no_exception_escapes(self):
        trace = run_cascade(self.BEHAVIORS, providers=self.PROVIDERS)
        assert len(trace.attempts) == 3

    def test_final_status_and_winner(self):
        trace = run_cascade(self.BEHAVIORS, providers=self.PROVIDERS)
        assert trace.final_status == "exhausted"
        assert trace.winning_provider is None
        assert trace.retry_count == len(trace.attempts) - 1 == 2

    def test_every_attempt_retryable_error(self):
        trace = run_cascade(self.BEHAVIORS, providers=self.PROVIDERS)
        assert [a.outcome for a in trace.attempts] == ["retryable_error"] * 3

    def test_cost_zero(self):
        trace = run_cascade(self.BEHAVIORS, providers=self.PROVIDERS)
        assert trace.total_cost_usd == 0.0


class TestNonRetryableRaisesOut:
    """Caller classifies a failure as non-retryable and re-raises it."""

    def test_exception_propagates(self):
        trace = Trace()
        with pytest.raises(ValueError):
            run_cascade(["fatal"], providers=["groq"], trace=trace)

    def test_rollup_ran_despite_exception(self):
        trace = Trace()
        with pytest.raises(ValueError):
            run_cascade(["fatal"], providers=["groq"], trace=trace)
        assert trace.final_status == "exhausted"
        assert trace.ended_at is not None

    def test_attempt_recorded_non_retryable(self):
        trace = Trace()
        with pytest.raises(ValueError):
            run_cascade(["fatal"], providers=["groq"], trace=trace)
        assert trace.attempts[0].outcome == "non_retryable_error"
        assert trace.attempts[0].error_type == "ValueError"
        assert trace.retry_count == 0
        assert trace.winning_provider is None


class TestUnclassifiedExceptionDefensivePath:
    """An exception leaves ``with trace.attempt()`` without record_failure."""

    def test_exception_still_propagates(self):
        trace = Trace()
        with pytest.raises(KeyError):
            run_cascade(
                ["boom"], providers=["groq"], trace=trace, classify_unexpected=False
            )

    def test_attempt_defensively_classified(self):
        trace = Trace()
        with pytest.raises(KeyError):
            run_cascade(
                ["boom"], providers=["groq"], trace=trace, classify_unexpected=False
            )
        attempt = trace.attempts[0]
        assert attempt.outcome == "non_retryable_error"
        assert attempt.error_type == "KeyError"
        assert attempt.error_message is not None
        assert attempt.latency_ms is not None
        assert trace.final_status == "exhausted"
