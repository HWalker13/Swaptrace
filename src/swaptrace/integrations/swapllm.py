"""swaptrace <-> swapllm integration: a traced, drop-in wrapper for ``Router``.

Requires the ``swaptrace[swapllm]`` extra (``pip install swaptrace[swapllm]``).
Nothing in ``swaptrace``'s core imports this module.

Why this reimplements swapllm's fallback loop instead of wrapping
``Router.complete()``: on a successful call ``Router.complete()`` discards every
piece of per-attempt information -- its ``failures`` list is a local, it returns
a bare ``str``/model with no winning-provider metadata, and the ``Router``
instance keeps no history (see ``docs/swapllm-integration-notes.md`` S1). The
only way to trace attempts on the success path is to drive the loop ourselves.

The loop below mirrors ``swapllm.router.Router.complete()`` (swapllm 0.1.0,
~18 lines). **If swapllm changes its fallback classification or schema-handling,
this must be updated to match.** Kept in sync deliberately:

* ``_RETRYABLE_EXCEPTIONS`` mirrors ``swapllm.router._RETRYABLE`` -- rebuilt from
  the public exception classes rather than imported (it's just a tuple of stable
  public names).
* ``_strip_markdown_json_fence`` IS imported from ``swapllm.router`` (private but
  a standalone pure function) so fence-stripping stays automatically in sync.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError
from swapllm import (
    AllProvidersFailedError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
    Router,
)
from swapllm.providers import Message
from swapllm.providers.base import validate_messages
from swapllm.router import _strip_markdown_json_fence

from swaptrace.core import Trace

__all__ = ["traced", "TracedRouter"]

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

# Mirrors swapllm.router._RETRYABLE (swapllm 0.1.0): the failure classes that
# mean "this provider, not the request, is the problem" and advance to the next
# provider. ProviderRequestError and the plain ValueError from
# validate_messages() are deliberately absent -- they propagate.
_RETRYABLE_EXCEPTIONS: tuple[type[ProviderError], ...] = (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderServerError,
    ProviderResponseValidationError,
)


class TracedRouter:
    """Drop-in wrapper around a swapllm :class:`~swapllm.Router` that records a
    :class:`~swaptrace.core.Trace` for every :meth:`complete` call.

    :meth:`complete` takes the same arguments and returns exactly what a plain
    ``Router.complete()`` would -- raising the same ``AllProvidersFailedError``
    on exhaustion and the same ``ProviderRequestError`` / ``ValueError`` on a
    non-retryable failure -- so existing call sites, including
    ``except AllProvidersFailedError`` handlers, keep working unchanged. The
    recorded trace is exposed out-of-band:

    * :attr:`last_trace` -- the most recent call's ``Trace`` (``None`` before the
      first call).
    * ``on_trace`` -- an optional callback invoked with the ``Trace`` after every
      call (e.g. ``swaptrace.storage.append_trace``).

    Both are updated on **every** exit path: success, exhaustion, and a
    propagating non-retryable failure.

    swapllm token/usage data is not available (``provider.complete()`` returns
    only text), so every attempt's ``cost_usd`` stays ``None``.
    """

    def __init__(
        self, router: Router, *, on_trace: Callable[[Trace], None] | None = None
    ) -> None:
        self._router = router
        self._on_trace = on_trace
        # Rebuilt from the public `providers` list -- not router._by_name (private).
        self._by_name = {p.name: p for p in router.providers}
        self.last_trace: Trace | None = None

    def complete(
        self,
        *,
        messages: list[Message],
        schema: type[_SchemaT] | None = None,
    ) -> str | _SchemaT:
        # swapllm's adapters each run validate_messages() as the first line of
        # their own complete(); we run it up front so a malformed-input
        # ValueError is raised before any Trace exists -- no provider is
        # attempted, so there is nothing to trace. The per-provider calls below
        # re-run it harmlessly.
        validate_messages(messages)

        trace = Trace()
        failures: list[ProviderError] = []  # mirrors Router.complete()'s own local
        response: str | _SchemaT | None = None

        try:
            with trace:
                for name in self._router.fallback_order:
                    provider = self._by_name[name]
                    with trace.attempt(
                        provider=provider.name, model=provider.model
                    ) as attempt:
                        try:
                            raw = provider.complete(messages)
                        except _RETRYABLE_EXCEPTIONS as exc:
                            attempt.record_failure(exc, retryable=True)
                            failures.append(exc)
                            continue
                        except (ProviderRequestError, ValueError) as exc:
                            attempt.record_failure(exc, retryable=False)
                            raise

                        if schema is not None:
                            # Mirror Router's order: the provider call succeeds
                            # first, schema validation happens after, on the
                            # returned text. A schema failure is a provider
                            # failure (retryable), not a caller error.
                            try:
                                validated = schema.model_validate_json(
                                    _strip_markdown_json_fence(raw)
                                )
                            except ValidationError as exc:
                                wrapped = ProviderResponseValidationError(
                                    provider.name, str(exc), original=exc
                                )
                                attempt.record_failure(wrapped, retryable=True)
                                failures.append(wrapped)
                                continue
                            attempt.record_success(
                                validated, prompt_tokens=None, completion_tokens=None
                            )
                            response = validated
                        else:
                            attempt.record_success(
                                raw, prompt_tokens=None, completion_tokens=None
                            )
                            response = raw
                        break  # success -- stop trying providers
        finally:
            # Runs on every path: success, exhaustion, and a non-retryable
            # failure propagating through `with trace:`. `with trace:` has
            # already finalized `trace` by the time we get here.
            self.last_trace = trace
            if self._on_trace is not None:
                self._on_trace(trace)

        if trace.final_status == "exhausted":
            # The loop ran out with no success and no exception propagated --
            # mirror Router's exhaustion behavior. `failures` is byte-for-byte
            # what Router.complete() would have collected.
            raise AllProvidersFailedError(failures)

        return response


def traced(
    router: Router, *, on_trace: Callable[[Trace], None] | None = None
) -> TracedRouter:
    """Wrap ``router`` so every ``.complete()`` call records a swaptrace ``Trace``.

    ``on_trace``, if given, is called with the ``Trace`` after every completion.
    """
    return TracedRouter(router, on_trace=on_trace)
