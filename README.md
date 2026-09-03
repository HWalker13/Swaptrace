# swaptrace

**Fallback-cascade observability for multi-provider LLM routers — see every attempt, not just the final answer.**

## The problem

When a router falls back across providers — Groq rate-limits, so it tries OpenAI,
which times out, so it tries Anthropic, which answers — the only thing you get
back is that final answer. The retry history is thrown away the moment a provider
succeeds. But that history is exactly what answers *"why did this request cost 3×
the usual"* and *"which provider is actually carrying the traffic."* `swaptrace`
records every attempt in a cascade — provider, model, outcome, latency, token
cost — and keeps them whether the trace ends in success or in exhaustion.

## Install

```
pip install swaptrace
```

For the [swapLLM](https://github.com/HWalker13/swapllm) integration:

```
pip install "swaptrace[swapllm]"
```

`swaptrace`'s core is standard-library only — no runtime dependencies. The
`swapllm` extra is opt-in.

## Quickstart

### Standalone

Wrap your own retry loop. `swaptrace` defines no exception types and makes no HTTP
calls — you classify each failure, it does the bookkeeping.

```python
from swaptrace import Trace


class RateLimited(Exception):
    """Your own exception type — swaptrace defines none of its own."""


def call_provider(name):
    if name == "groq":
        raise RateLimited("429 Too Many Requests")
    return {"text": "...", "prompt_tokens": 41, "completion_tokens": 12}


with Trace() as trace:
    for provider in ["groq", "openai"]:
        with trace.attempt(provider=provider, model="llama-3.1-8b-instant") as attempt:
            try:
                resp = call_provider(provider)
            except RateLimited as err:
                attempt.record_failure(err, retryable=True)
                continue
            attempt.record_success(
                resp,
                prompt_tokens=resp["prompt_tokens"],
                completion_tokens=resp["completion_tokens"],
            )
            break

print(trace.final_status)       # success
print(trace.winning_provider)   # openai
print(trace.retry_count)        # 1
for a in trace.attempts:
    print(a.attempt_index, a.provider, a.outcome, a.error_type, a.cost_usd)
# 0 groq retryable_error RateLimited None
# 1 openai success None 3.0100000000000004e-06
```

`cost_usd` is estimated from a small built-in per-token price table
(`swaptrace.pricing.DEFAULT_PRICING`). For models it doesn't know, pass
`pricing_overrides={"model-name": (input_rate, output_rate)}` to
`record_success` — rates are USD per million tokens.

### swapLLM integration

`traced()` wraps a `swapllm.Router` and records a `Trace` for every `.complete()`
call — without changing what `.complete()` returns or which exceptions it raises,
so existing call sites are untouched.

```python
from swapllm import Router, GroqProvider, OpenAIProvider
from swaptrace import storage
from swaptrace.integrations.swapllm import traced

router = Router(
    providers=[
        GroqProvider(api_key=..., model="llama-3.1-8b-instant"),
        OpenAIProvider(api_key=..., model="gpt-4o-mini"),
    ],
    fallback_order=["groq", "openai"],
)

# record every trace to a JSONL file as it completes
router = traced(
    router,
    on_trace=lambda t: storage.append_trace(t, ".swaptrace/traces.jsonl"),
)

answer = router.complete(messages=[{"role": "user", "content": "..."}])

router.last_trace.winning_provider
# "openai"
[(a.provider, a.outcome) for a in router.last_trace.attempts]
# [('groq', 'retryable_error'), ('openai', 'success')]
```

`traced()` re-implements swapLLM's fallback loop instead of wrapping
`Router.complete()`: swapLLM discards per-attempt information once a provider
succeeds, so the only way to observe it is to drive the loop. `AllProvidersFailedError`,
`ProviderRequestError`, and schema-validation behaviour are preserved exactly.
swapLLM's provider adapters return only text, so `cost_usd` stays `None` for
swapLLM-traced attempts.

## CLI

`swaptrace` never writes trace files on its own — wire `storage.append_trace`
into `on_trace` as shown above. It reads `.swaptrace/traces.jsonl` (relative to
the working directory — traces are per-project, like `.git/`) by default;
override with `--path`.

### `swaptrace query`

List traces, optionally filtered. `--provider` and `--status` match a trace if
*any* of its attempts matches; `--min-cost` is checked against the trace total.
Active filters combine with AND.

```
$ swaptrace query
2026-09-03T19:35:13.726568+00:00  a60d166a  success    groq          1 attempt(s)  $0.000025  25.0ms
2026-09-03T19:35:13.752308+00:00  a34ff498  success    openai        2 attempt(s)  $0.000221  45.1ms
2026-09-03T19:35:13.798053+00:00  4847ffd3  success    anthropic     3 attempt(s)  $0.001980  70.5ms
2026-09-03T19:35:13.869284+00:00  c627f556  exhausted  -             3 attempt(s)  $0.000000  68.6ms
2026-09-03T19:35:13.939634+00:00  9647ec87  exhausted  -             1 attempt(s)  $0.000000  22.1ms
5 of 5 trace(s).

$ swaptrace query --min-cost 0.001
2026-09-03T19:35:13.798053+00:00  4847ffd3  success    anthropic     3 attempt(s)  $0.001980  70.5ms
1 of 5 trace(s).
```

### `swaptrace report --compare-providers`

Flatten every attempt across every trace, grouped by provider — reliability,
latency, and cost side by side, most reliable first.

```
$ swaptrace report --compare-providers
PROVIDER     ATTEMPTS  SUCCESSES  SUCCESS%   AVG LATENCY    TOTAL COST    $/SUCCESS
anthropic           2          1     50.0%        22.3ms     $0.001980    $0.001980
groq                4          1     25.0%        23.6ms     $0.000025    $0.000025
openai              4          1     25.0%        23.1ms     $0.000221    $0.000221
```

`SUCCESS%` is per-attempt reliability (successes ÷ attempts, across all traces);
`AVG LATENCY` is over every attempt, successful or not; `$/SUCCESS` is `-` for a
provider that has never succeeded.

## Instrumentation overhead

Wrapping a provider call in swaptrace's `Trace`/`Attempt` bookkeeping adds a
measured **7.75 µs per call at the median and 7.96 µs at p95** (CPython 3.12.8,
macOS arm64; 5 trials × 1000 iterations, warm-up, GC disabled, `time.perf_counter`).
That overhead is a fixed cost of swaptrace's own work — a couple of `uuid4()` and
`datetime.now()` calls, the trace rollup, a pricing-table lookup — and does not
scale with the wrapped call, so against a typical 200 ms–2 s LLM API call it is
**well under 0.01%** (≈0.008% at 100 ms, ≈0.0004% at 2 s). Reproduce with
`python benchmarks/benchmark_overhead.py`; the full raw dataset is in
`benchmarks/results/`.

## What this isn't

- No hosted dashboard or web UI — `swaptrace query` / `report` are the interface.
- No OpenTelemetry / OTLP exporter yet.
- No streaming-response support.
- No multi-agent span tracing (agentrace-ai / spyllm cover that space) — a
  `swaptrace` trace is scoped to a single provider cascade.

## Development

```
pip install -e ".[dev]"
pytest
```

runs the core suite — **61 passed, 1 skipped** (the swapLLM integration tests
skip when the extra isn't installed). For the full **73**:

```
pip install -e ".[dev,swapllm]"
pytest
```

Requires Python 3.11+.

## License

MIT — see [LICENSE](LICENSE).
