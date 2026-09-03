# swapLLM integration research (for Session 9)

Read-only investigation of the real `swapllm` package, done in Session 8. Session 9
builds `swaptrace`'s integration adapter (`traced()`) against this.

## Source examined

- Repo: `https://github.com/HWalker13/swapllm.git`
- Commit: `9348c259b90206d5342e2f843b9d2b0a9ef83c14` ("Fix relative LICENSE link in README…", 2026-07-14)
- Cloned to `/tmp/swapllm-reference` (outside this repo; not a dependency).
- Package version `0.1.0`, `requires-python = ">=3.10"`.
- Runtime deps: `pydantic>=2,<3`, `groq>=1,<2`, `openai>=1,<3`, `anthropic>=0.30,<1`.
- Package layout: `swapllm/{__init__,router,exceptions,_normalize}.py` + `swapllm/providers/{base,groq,openai,anthropic}.py`.

---

## 1. The hinge question — does the success path expose failed attempts?

**No. On a successful call, every trace of the providers that failed first is
discarded. That information survives *only* on total failure, inside
`AllProvidersFailedError.failures`.**

### Evidence (from `swapllm/router.py`)

`Router.complete()` is the entire fallback loop, ~18 lines:

```python
# router.py:156
def complete(self, messages: list[Message], schema: type[_SchemaT] | None = None) -> str | _SchemaT:
    failures: list[ProviderError] = []                       # router.py:170  <-- LOCAL variable
    for name in self.fallback_order:
        provider = self._by_name[name]
        try:
            text = provider.complete(messages)               # router.py:174
        except _RETRYABLE as exc:
            failures.append(exc)                             # router.py:176
            continue
        if schema is None:
            return text                                      # router.py:180  <-- returns; `failures` dropped
        try:
            return schema.model_validate_json(_strip_markdown_json_fence(text))   # router.py:183  <-- returns; `failures` dropped
        except ValidationError as exc:
            failures.append(ProviderResponseValidationError(name, str(exc), original=exc))  # router.py:185
    raise AllProvidersFailedError(failures)                   # router.py:187  <-- only escape for `failures`
```

- `failures` is a **local** in `complete()` (router.py:170). It is never assigned to
  `self` and never returned on a success path.
- Both success returns (router.py:180 no-schema, router.py:183 schema) return the
  answer alone — `str` or the validated model — and drop `failures`.
- `Router.__init__` (router.py:129–154) sets **only** `self.providers`,
  `self.fallback_order`, `self._by_name`. There is no `self._attempts`,
  `self._history`, `self._last_result`, or any mutable accumulator. The instance
  is effectively immutable after construction.
- `complete()` also returns **no indication of which provider answered** — the
  caller gets `str | SchemaT` with zero metadata.
- There are **no hooks, callbacks, events, or instrumentation points** anywhere
  in swapllm (`grep -niE 'cost|usage|track|hook|callback|instrument|metric'` over
  the source finds nothing but the SPEC's own "out of scope" line). `SPEC.md §2`
  lists "Usage/cost tracking" as explicitly out of scope for v1 — confirmed by the
  code, not just the planning doc.

### Consequence for Session 9

A thin wrapper that calls `Router.complete()` and inspects the `Router` afterward
**cannot work** — on the (common) success path there is nothing to inspect: no
winning-provider name, no failed-attempt list, no per-attempt timing. `traced()`
must drive the fallback loop itself. See §6.

---

## 2. Exception classes — corrected against real source

The original swaptrace spec guessed `RateLimitError`, `TimeoutError`,
`ServerError`, `ConfigError`. **All four names are wrong.** Real hierarchy
(`swapllm/exceptions.py`, all exported from `swapllm/__init__.py`):

```
Exception
└── SwapLLMError                        # base for everything swapllm raises
    ├── ProviderError                   # one provider's request failed
    │   ├── ProviderRateLimitError      # 429
    │   ├── ProviderTimeoutError        # timeout OR connection failure (DNS/refused/dropped) — one type on purpose
    │   ├── ProviderServerError         # 5xx
    │   ├── ProviderResponseValidationError   # 200 OK but unusable: null content (adapter) OR schema-invalid (router)
    │   └── ProviderRequestError        # 400/401/403/unexpected SDK error — caller misconfig
    └── AllProvidersFailedError         # every provider in fallback_order failed
```

Plus: a **plain `ValueError`** (not a `SwapLLMError`) from
`swapllm.providers.base.validate_messages()` for malformed message lists
(>1 system message, or system message not at index 0).

### `ProviderError` shape (exceptions.py:29–32)

```python
def __init__(self, provider: str, message: str, *, original: Exception | None = None):
    super().__init__(f"[{provider}] {message}")
    self.provider = provider          # short id, matches fallback_order entries: "groq" / "openai" / "anthropic"
    self.original = original          # the raw vendor SDK exception (or None)
```

### What triggers fallback vs. what propagates (router.py:63–68)

```python
_RETRYABLE = (ProviderRateLimitError, ProviderTimeoutError,
              ProviderServerError, ProviderResponseValidationError)
```

- **Advances to next provider** (`_RETRYABLE`): rate limit, timeout/connection,
  5xx, unusable/schema-invalid response.
- **Propagates immediately, no fallback**: `ProviderRequestError`, and the plain
  `ValueError` from `validate_messages()`.
- **No same-provider retry, ever** — `Router.__init__` even rejects a duplicate
  name in `fallback_order` (router.py:143–145) to prevent bypassing that rule.
  Also rejects an empty `fallback_order` and names not present in `providers`.

### Mapping (for `traced()`'s `record_failure(..., retryable=?)`)

| swapllm exception | swaptrace `retryable=` |
|---|---|
| `ProviderRateLimitError`, `ProviderTimeoutError`, `ProviderServerError`, `ProviderResponseValidationError` | `True` |
| `ProviderRequestError` | `False` |
| `ValueError` (validate_messages) | `False` — and it's raised on the *first* provider only, before any real call |

Vendor→normalized translation happens in `swapllm/_normalize.py::normalize_exception`
(timeout checked before connection because every vendor's `APITimeoutError`
subclasses its `APIConnectionError`); adapters call it at their SDK boundary, so
the Router (and `traced()`) only ever see the normalized types above.

---

## 3. `Router.complete()` signature and return type

```python
def complete(self, messages: list[Message], schema: type[_SchemaT] | None = None) -> str | _SchemaT
```

- `messages`: `list[Message]` where `Message = TypedDict("Message", {"role": str, "content": str})` (`providers/base.py:23`).
- `schema`: optional `type[pydantic.BaseModel]`.
  - **omitted** → returns the winning provider's raw text as `str`.
  - **provided** → the raw text is JSON-parsed and validated (`schema.model_validate_json`),
    and the **validated model instance** is returned — never raw text alongside a schema.
  - A single wrapping markdown fence (```` ```json\n…\n``` ```` or bare ```` ```\n…\n``` ````)
    around the *entire* text is stripped before validation
    (`router.py::_strip_markdown_json_fence`, regex `_MARKDOWN_JSON_FENCE_RE`).
    Text with anything before/after the fence is **not** stripped and fails validation.
  - Schema-validation failure is raised **by the Router** (router.py:185) as a
    freshly constructed `ProviderResponseValidationError(name, str(exc), original=exc)`
    and appended to `failures` — it is a retryable trigger, so the Router moves on
    to the next provider. Note this happens *after* `provider.complete()` has
    already returned successfully.
- No timeout/retry/temperature kwargs — plain text/chat only (SPEC §2).

---

## 4. `fallback_order` strings → provider instances

`Router.__init__` (router.py:147): `by_name = {p.name: p for p in providers}`,
stored as `self._by_name`. The loop then does `provider = self._by_name[name]`
for each `name` in `fallback_order`.

Each adapter carries `name` as a **class attribute**:

| class | `name` | other attrs |
|---|---|---|
| `GroqProvider` | `"groq"` | `.model` (str, instance attr), `._client` |
| `OpenAIProvider` | `"openai"` | `.model`, `._client` |
| `AnthropicProvider` | `"anthropic"` | `.model`, `.max_tokens` (default 4096), `._client` |

`Provider` is a `@runtime_checkable` `Protocol` (`providers/base.py:61`):
```python
class Provider(Protocol):
    name: str
    def complete(self, messages: list[Message]) -> str: ...
```
Constructors: `XProvider(api_key: str, model: str, *, http_client: httpx.Client | None = None)`
(Anthropic also takes `*, max_tokens: int = 4096`).

**For `traced()`:** the provider-name string swaptrace's `Attempt.provider` needs is
exactly `provider.name` (equivalently, the `fallback_order` entry — they are the
same string by construction). `Attempt.model` should be `provider.model`.
`traced()` can rebuild the map from public attributes:
`{p.name: p for p in router.providers}` (avoids touching the private `_by_name`),
and iterate `router.fallback_order` (public).

---

## 5. `AllProvidersFailedError` payload

```python
# exceptions.py:109
def __init__(self, failures: list[ProviderError]) -> None:
    self.failures = failures
    detail = "; ".join(f"{f.provider}: {type(f).__name__}: {f.args[0]}" for f in failures)
    super().__init__(f"All providers failed: {detail}")
```

- `.failures`: `list[ProviderError]`, **one per attempted provider, in attempt
  order** (confirmed by `tests/test_router.py::test_all_providers_failed_error_populates_failures_with_provider_and_original`:
  `[f.provider for f in failures] == ["groq", "openai", "anthropic"]`).
- Each element is a `_RETRYABLE`-type `ProviderError` with `.provider`, `.original`
  (always non-`None` in practice for real vendor failures; the router-constructed
  schema-validation ones carry the `pydantic.ValidationError` as `.original`).
- Raised only after **every** provider in `fallback_order` has failed a retryable
  way. A non-retryable failure short-circuits before this is reached.
- There is **no** per-attempt latency, token, or cost data on it — only the
  exception objects. Timing must be measured by whoever drives the calls.

---

## 6. Recommended integration strategy for `traced()`

Three strategies were considered:

### A. Thin outer wrapper around `Router.complete()` — REJECTED

Call `Router.complete()`, catch `AllProvidersFailedError` for the failure path.
**Fails the §1 finding:** on success there is no winning-provider name, no
failed-attempt list, and no per-attempt timing to read — `complete()` returns a
bare `str`/model and the `Router` keeps no history. This yields an almost-empty
trace for the most common case. Not viable.

### B. Instrument the provider objects, let `Router.complete()` drive — WORKABLE, LEAKY

Wrap each `provider.complete` (on the instances, with `try/finally` teardown) to
open a `trace.attempt(...)` per call, then call `Router.complete()` unchanged.
Per-attempt timing comes for free at the wrapped call site.
**Seam problem:** schema validation runs in the Router *after*
`provider.complete()` returned successfully (router.py:183–185), so the wrapper
records a `record_success()` for a provider whose output the Router then rejects
and skips. Reconciling "recorded success that didn't actually win" requires
post-hoc inference (diffing against `AllProvidersFailedError.failures`, or against
the final return value). Also: monkey-patching third-party instances is intrusive.

### C. Reimplement the fallback loop in `traced()` using swapllm's provider + exception primitives — RECOMMENDED

`traced(router, messages, schema=None, *, pricing_overrides=None, path=None)` drives
its own loop that mirrors `Router.complete()`'s ~18 lines, wrapping each provider
call in `with trace.attempt(provider=p.name, model=p.model) as attempt:`:

- `text = provider.complete(messages)` inside the `attempt` block.
- `except (ProviderRateLimitError, ProviderTimeoutError, ProviderServerError, ProviderResponseValidationError) as exc:`
  → `attempt.record_failure(exc, retryable=True)`; append to a local `failures`; `continue`.
- `except (ProviderRequestError, ValueError) as exc:` → `attempt.record_failure(exc, retryable=False)`; re-raise (no more providers).
- schema `None` → `attempt.record_success(text, prompt_tokens=?, completion_tokens=?)`; return `text`.
- schema given → `schema.model_validate_json(strip_fence(text))`; on success `record_success(...)` + return the model;
  on `pydantic.ValidationError` → build `ProviderResponseValidationError(name, str(exc), original=exc)`,
  `attempt.record_failure(that, retryable=True)`, append, `continue` (matches router.py:183–185 exactly).
- loop exhausted → `raise AllProvidersFailedError(failures)`.

**Why C is the right call:**
- It is the only option that populates `Attempt.provider`, `Attempt.outcome`,
  `winning_provider`, and per-attempt timing correctly on the **success** path —
  which §1 proves is otherwise impossible.
- Schema-validation failures get classified correctly and attributed to the right
  provider, with no reconciliation guesswork (fixes B's seam).
- The duplicated surface is genuinely small and stable: `Router.complete()` is
  ~18 lines, SPEC §2 says "do not let this creep", and v1 is frozen. The only
  subtleties to mirror are the `_RETRYABLE` set (reconstructable from the 4
  public exception classes — no need to import `swapllm.router._RETRYABLE`) and
  the markdown-fence strip (only relevant when `schema=` is used).

**Open decisions for Session 9 (not resolved here):**
- **Dependency.** C needs `swapllm` importable (for the exception classes and the
  provider protocol) and, *only when `schema=` is passed*, `pydantic`. Both are
  already present transitively for anyone using `swapllm` with schemas
  (`swapllm` depends on `pydantic>=2`). Options: a `swaptrace[swapllm]` optional
  extra, or import-guarded (`try: import swapllm`) so the core package stays
  zero-dep. Recommend the optional-extra route and keeping the adapter in a new
  `swaptrace/integrations/` package that is never imported by core.
- **Token counts.** swapllm's `provider.complete()` returns **only `str`** — it
  throws away the vendor `usage` block (visible in `tests/test_router.py` bodies:
  the mock responses include `usage`, but the adapters return `content` only). So
  `traced()` has **no token counts to pass** to `record_success()` unless Session 9
  adds its own usage extraction (would require going below `provider.complete()`,
  e.g. per-vendor response parsing — likely out of scope). Expect
  `prompt_tokens=None, completion_tokens=None` → `cost_usd=None` for now, which
  the pricing layer already handles gracefully. Worth a short note in Session 9's
  plan and possibly a `swaptrace` roadmap item.
- **Fence-strip parity.** Either vendor a 3-line copy of `_strip_markdown_json_fence`
  or accept minor divergence for the `schema=` + fenced-output case. Low stakes.
- **API shape.** `traced(router, ...)` (reads `router.providers` / `router.fallback_order`)
  vs. `traced(providers=[...], fallback_order=[...], ...)`. The former reuses an
  already-built `Router` and its `__init__` validation; recommend it.
