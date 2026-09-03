"""Integration tests for :mod:`swaptrace.integrations.swapllm` (`traced()`).

Formalizes the seven scenarios hand-verified in Session 9. If the swapllm
optional extra is not installed the whole module skips cleanly rather than
erroring on import.
"""

import pytest

pytest.importorskip("swapllm")

from swapllm import (  # noqa: E402  (after importorskip, by design)
    AllProvidersFailedError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
    Router,
)
from swapllm.providers import Provider  # noqa: E402  -- swapllm's formal Provider Protocol
from pydantic import BaseModel  # noqa: E402

from swaptrace.integrations.swapllm import TracedRouter, traced  # noqa: E402

MSG = [{"role": "user", "content": "hi"}]

# behavior token -> swapllm exception class the fake raises for it
_RETRYABLE_TOKENS = {
    "rate_limit": ProviderRateLimitError,
    "timeout": ProviderTimeoutError,
    "server": ProviderServerError,
    "response_validation": ProviderResponseValidationError,
}
_TOKENS = {**_RETRYABLE_TOKENS, "request_error": ProviderRequestError}


class FakeProvider(Provider):
    """Implements swapllm's formal ``Provider`` Protocol (subclass + an
    ``isinstance`` guard in :func:`make_fake_provider`), so a future swapllm
    interface change surfaces here rather than silently passing.

    ``behavior`` is one of: a plain string (returned as the completion text),
    a token from ``_TOKENS`` (raises that exception), or a ``list`` of the
    above consumed one element per successive ``complete()`` call.
    """

    def __init__(self, name: str, model: str, behavior) -> None:
        self.name = name
        self.model = model
        self._behavior = behavior
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        b = self._behavior
        if isinstance(b, list):
            b = b[self.calls - 1]
        if b in _TOKENS:
            raise _TOKENS[b](self.name, f"fake {b}")
        return b


def make_fake_provider(name: str, model: str, behavior) -> FakeProvider:
    provider = FakeProvider(name, model, behavior)
    assert isinstance(provider, Provider)  # guards the structural contract
    return provider


def _router(*specs, order=None):
    """specs: (name, behavior) or (name, model, behavior)."""
    providers = []
    for spec in specs:
        if len(spec) == 2:
            name, behavior = spec
            model = f"{name}-model"
        else:
            name, model, behavior = spec
        providers.append(make_fake_provider(name, model, behavior))
    order = order or [p.name for p in providers]
    return providers, Router(providers=providers, fallback_order=order)


class Recipe(BaseModel):
    title: str
    servings: int


# --------------------------------------------------------------------------- #
def test_immediate_success_no_schema():
    providers, router = _router(
        ("groq", "hello from groq"),
        ("openai", "request_error"),
        ("anthropic", "request_error"),
    )
    tr = traced(router)

    result = tr.complete(messages=MSG)

    assert result == "hello from groq"
    t = tr.last_trace
    assert t.final_status == "success"
    assert len(t.attempts) == 1
    assert t.attempts[0].outcome == "success"
    assert t.attempts[0].provider == "groq"
    assert t.attempts[0].model == "groq-model"
    assert t.attempts[0].cost_usd is None
    assert t.total_cost_usd == 0.0
    assert t.winning_provider == "groq"
    assert providers[1].calls == 0 and providers[2].calls == 0


def test_fallback_then_success():
    providers, router = _router(
        ("groq", "rate_limit"),
        ("openai", "hello from openai"),
        ("anthropic", "request_error"),
    )
    tr = traced(router)

    result = tr.complete(messages=MSG)

    assert result == "hello from openai"
    t = tr.last_trace
    assert t.final_status == "success"
    assert [a.outcome for a in t.attempts] == ["retryable_error", "success"]
    assert t.attempts[0].error_type == "ProviderRateLimitError"
    assert t.attempts[0].provider == "groq"
    assert t.retry_count == 1
    assert t.winning_provider == "openai"
    assert providers[2].calls == 0


def test_full_exhaustion_all_retryable():
    providers, router = _router(
        ("groq", "rate_limit"),
        ("openai", "timeout"),
        ("anthropic", "server"),
    )
    tr = traced(router)

    with pytest.raises(AllProvidersFailedError) as ei:
        tr.complete(messages=MSG)

    failures = ei.value.failures
    assert [f.provider for f in failures] == ["groq", "openai", "anthropic"]
    assert isinstance(failures[0], ProviderRateLimitError)
    assert isinstance(failures[1], ProviderTimeoutError)
    assert isinstance(failures[2], ProviderServerError)
    t = tr.last_trace
    assert t.final_status == "exhausted"
    assert t.winning_provider is None
    assert [a.outcome for a in t.attempts] == ["retryable_error"] * 3


def test_non_retryable_propagates_and_last_trace_not_stale():
    # call 1 succeeds, call 2 hits a non-retryable error on the same TracedRouter:
    # last_trace must advance to call 2's finalized trace, not stay stale (this
    # locks in Session 9's try/finally fix).
    groq = make_fake_provider("groq", "groq-model", ["ok call one", "request_error"])
    openai = make_fake_provider("openai", "openai-model", ["unused", "unreachable"])
    tr = traced(Router(providers=[groq, openai], fallback_order=["groq", "openai"]))

    tr.complete(messages=MSG)
    first = tr.last_trace
    assert first.final_status == "success"

    with pytest.raises(ProviderRequestError) as ei:
        tr.complete(messages=MSG)
    assert not isinstance(ei.value, AllProvidersFailedError)
    assert ei.value.provider == "groq"

    t = tr.last_trace
    assert t is not None and t is not first
    assert t.trace_id != first.trace_id
    assert len(t.attempts) == 1
    assert t.attempts[0].outcome == "non_retryable_error"
    assert t.attempts[0].error_type == "ProviderRequestError"
    assert t.final_status == "exhausted"
    assert t.ended_at is not None
    assert groq.calls == 2 and openai.calls == 0


def test_provider_raised_response_validation_error_is_retryable():
    # ProviderResponseValidationError straight from a provider adapter (null
    # content), NO schema in play -- must still be treated as retryable.
    # Regression test for the narrow catch-tuple Session 9 corrected.
    providers, router = _router(
        ("groq", "response_validation"),
        ("openai", "hello from openai"),
    )
    tr = traced(router)

    result = tr.complete(messages=MSG)

    assert result == "hello from openai"
    t = tr.last_trace
    assert [a.outcome for a in t.attempts] == ["retryable_error", "success"]
    assert t.attempts[0].error_type == "ProviderResponseValidationError"
    assert t.winning_provider == "openai"


def test_schema_validation_failure_then_success():
    providers, router = _router(
        ("groq", "not valid json"),
        ("openai", '{"title": "Soup", "servings": 4}'),
    )
    tr = traced(router)

    result = tr.complete(messages=MSG, schema=Recipe)

    assert result == Recipe(title="Soup", servings=4)
    t = tr.last_trace
    assert t.attempts[0].outcome == "retryable_error"
    # the real wrapped type, not a raw pydantic.ValidationError leaking through
    assert t.attempts[0].error_type == "ProviderResponseValidationError"
    assert t.attempts[1].outcome == "success"


def test_schema_validation_strips_full_markdown_fence():
    providers, router = _router(("groq", '```json\n{"title": "Stew", "servings": 2}\n```'))
    tr = traced(router)

    assert tr.complete(messages=MSG, schema=Recipe) == Recipe(title="Stew", servings=2)
    assert tr.last_trace.attempts[0].outcome == "success"


def test_schema_validation_all_providers_fail():
    providers, router = _router(
        ("groq", "garbage"),
        ("openai", '{"title": "only a title"}'),
        ("anthropic", "also not json"),
    )
    tr = traced(router)

    with pytest.raises(AllProvidersFailedError) as ei:
        tr.complete(messages=MSG, schema=Recipe)

    failures = ei.value.failures
    assert [f.provider for f in failures] == ["groq", "openai", "anthropic"]
    assert all(isinstance(f, ProviderResponseValidationError) for f in failures)
    assert tr.last_trace.final_status == "exhausted"


def test_malformed_input_never_creates_trace():
    calls = []
    providers, router = _router(("groq", "unused"))
    tr = TracedRouter(router, on_trace=calls.append)  # fresh instance -> last_trace is None
    bad = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "hi"},
    ]

    with pytest.raises(ValueError) as ei:
        tr.complete(messages=bad)
    assert not isinstance(ei.value, AllProvidersFailedError)

    assert providers[0].calls == 0
    assert tr.last_trace is None
    assert calls == []


def test_last_trace_advances_across_calls():
    groq = make_fake_provider("groq", "groq-model", "rate_limit")
    openai = make_fake_provider("openai", "openai-model", "ok")
    tr = traced(Router(providers=[groq, openai], fallback_order=["groq", "openai"]))

    tr.complete(messages=MSG)
    first = tr.last_trace
    tr.complete(messages=MSG)
    second = tr.last_trace

    assert first is not second
    assert first.trace_id != second.trace_id
    assert second is tr.last_trace


def test_on_trace_fires_every_call_including_exhaustion():
    captured = []
    # lists are indexed by each provider's OWN call count: groq is called on all
    # three TracedRouter calls; openai only on calls 2 and 3.
    groq = make_fake_provider("groq", "groq-model", ["s1", "rate_limit", "server"])
    openai = make_fake_provider("openai", "openai-model", ["s2", "server"])
    tr = traced(
        Router(providers=[groq, openai], fallback_order=["groq", "openai"]),
        on_trace=captured.append,
    )

    assert tr.complete(messages=MSG) == "s1"          # 1: immediate success
    assert tr.complete(messages=MSG) == "s2"          # 2: fallback then success
    with pytest.raises(AllProvidersFailedError):
        tr.complete(messages=MSG)                     # 3: exhaustion

    assert len(captured) == 3
    assert [c.final_status for c in captured] == ["success", "success", "exhausted"]
    assert captured[-1] is tr.last_trace


def test_parity_with_real_router():
    # identical fakes through Router.complete() and traced(router).complete()
    def success_run(run):
        _, r = _router(("groq", "rate_limit"), ("openai", "the answer"))
        return run(r)

    assert success_run(lambda r: r.complete(MSG)) == "the answer"
    assert success_run(lambda r: traced(r).complete(messages=MSG)) == "the answer"

    def exhaustion_failures(run):
        _, r = _router(("groq", "rate_limit"), ("openai", "server"))
        try:
            run(r)
        except AllProvidersFailedError as e:
            return [(f.provider, type(f).__name__, f.args[0]) for f in e.failures]
        raise AssertionError("expected AllProvidersFailedError")

    assert exhaustion_failures(lambda r: r.complete(MSG)) == exhaustion_failures(
        lambda r: traced(r).complete(messages=MSG)
    )
