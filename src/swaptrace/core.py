"""Core data model for swaptrace.

The :class:`Trace` and :class:`Attempt` dataclasses plus their context managers.
A ``Trace`` records one "swap" sequence -- the ordered attempts made against a
cascade of providers to get a usable LLM response -- and finalizes a set of
rolled-up totals when its ``with`` block exits.

swaptrace is framework-agnostic on purpose: it defines no exception hierarchy.
The caller catches its own errors and classifies each one as retryable or not
via :meth:`Attempt.record_failure`.

Stdlib only: ``dataclasses``, ``uuid``, ``datetime``, ``typing.Literal``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from swaptrace.pricing import estimate_cost_usd

__all__ = ["Trace", "Attempt", "AttemptOutcome", "TraceStatus"]

AttemptOutcome = Literal["success", "retryable_error", "non_retryable_error"]
TraceStatus = Literal["success", "exhausted"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass(kw_only=True)
class Attempt:
    """A single provider/model call within a :class:`Trace`.

    Created by :meth:`Trace.attempt` and used directly as a context manager::

        with trace.attempt(provider="openai", model="gpt-4o") as attempt:
            response = call_provider(...)
            attempt.record_success(response, prompt_tokens=12, completion_tokens=34)

    ``__enter__`` stamps ``started_at``; ``__exit__`` stamps ``ended_at`` and
    ``latency_ms``. ``outcome`` and the token / error fields are set by
    :meth:`record_success` / :meth:`record_failure`, or -- if an exception
    leaves the block without the caller classifying it -- by ``__exit__``.

    Fields are declared in the project spec's data-model order. ``started_at`` /
    ``ended_at`` / ``latency_ms`` / ``outcome`` are typed ``| None`` because the
    context manager fills them in after construction.
    """

    attempt_id: str = field(default_factory=_new_id)
    trace_id: str = ""
    provider: str = ""
    model: str = ""
    attempt_index: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    outcome: AttemptOutcome | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __enter__(self) -> Attempt:
        self.started_at = _now()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ended_at = _now()
        if self.started_at is not None:
            self.latency_ms = (
                self.ended_at - self.started_at
            ).total_seconds() * 1000.0
        # If an exception is leaving this block and the caller never classified
        # it (record_success / record_failure both set ``outcome``, so an unset
        # ``outcome`` means neither ran), record it as a non-retryable failure
        # rather than let it pass through with the attempt looking unfinished.
        if exc is not None and self.outcome is None:
            self.outcome = "non_retryable_error"
            self.error_type = type(exc).__name__
            self.error_message = str(exc)
        return False  # never suppress the caller's exception

    def record_success(
        self,
        response,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        pricing_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Mark this attempt successful, store the token counts, and price it.

        ``response`` (the provider's response object) is accepted for a stable
        call signature but not persisted -- there is no field for it.

        ``cost_usd`` is filled in from :func:`swaptrace.pricing.estimate_cost_usd`
        using this attempt's ``model`` and the token counts; it stays ``None``
        when the model is not in the pricing table (or either token count is
        ``None``). ``pricing_overrides`` (``{model: (input_rate, output_rate)}``,
        USD per 1M tokens) is passed straight through -- keep a local dict in
        your retry loop and hand it in on each call.
        """
        self.outcome = "success"
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = estimate_cost_usd(
            self.model,
            prompt_tokens,
            completion_tokens,
            overrides=pricing_overrides,
        )

    def record_failure(self, error, *, retryable: bool) -> None:
        """Mark this attempt failed, using the caller's own classification.

        ``retryable=True`` -> ``outcome = "retryable_error"`` (the trace may try
        the next provider); ``retryable=False`` -> ``outcome =
        "non_retryable_error"`` (the caller is expected to re-raise).
        """
        self.outcome = "retryable_error" if retryable else "non_retryable_error"
        self.error_type = type(error).__name__
        self.error_message = str(error)


@dataclass(kw_only=True)
class Trace:
    """Top-level record of one swap sequence: the ordered attempts made to get
    a usable LLM response, plus the rolled-up totals.

    Used as a context manager::

        with Trace() as trace:
            for provider in providers:
                with trace.attempt(provider=provider, model=model) as attempt:
                    ...

    ``__enter__`` stamps ``started_at``. ``__exit__`` stamps ``ended_at`` and
    then finalizes the roll-up fields (:meth:`_finalize`) -- this runs whether
    or not an exception is propagating out of the ``with`` block, and never
    suppresses one.
    """

    trace_id: str = field(default_factory=_new_id)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    attempts: list[Attempt] = field(default_factory=list)
    final_status: TraceStatus | None = None
    winning_provider: str | None = None
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    retry_count: int = 0

    def __enter__(self) -> Trace:
        self.started_at = _now()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ended_at = _now()
        # Finalize bookkeeping even when a non-retryable failure is about to
        # raise out of the whole ``with Trace():`` block.
        self._finalize()
        return False  # never suppress the caller's exception

    def _finalize(self) -> None:
        """Aggregate ``self.attempts`` into the roll-up fields."""
        winner = next(
            (a for a in self.attempts if a.outcome == "success"), None
        )
        # Only two end states exist: a success happened, or it didn't. "Didn't"
        # covers both running out of retryable options and hitting a
        # non-retryable error.
        self.final_status = "success" if winner is not None else "exhausted"
        self.winning_provider = winner.provider if winner is not None else None
        # Retries = attempts beyond the first.
        self.retry_count = max(len(self.attempts) - 1, 0)
        # Total time spent across the whole cascade, retry overhead included --
        # the sum of the attempts' own latencies, deliberately NOT the trace's
        # wall-clock duration (which would hide time lost to retries).
        self.total_latency_ms = sum(a.latency_ms or 0.0 for a in self.attempts)
        # Sum of per-attempt cost, treating None as 0.0. Every cost_usd is None
        # until Session 7 lands pricing.py, so this is correctly 0.0 for now.
        self.total_cost_usd = sum(a.cost_usd or 0.0 for a in self.attempts)

    def attempt(self, provider: str, model: str) -> Attempt:
        """Create the next :class:`Attempt` for this trace and return it.

        The returned object is itself a context manager::

            with trace.attempt(provider, model) as attempt:
                ...

        The attempt is appended to :attr:`attempts` immediately (before the
        ``with`` block runs) so ``attempt_index`` stays stable.
        """
        att = Attempt(
            trace_id=self.trace_id,
            provider=provider,
            model=model,
            attempt_index=len(self.attempts),
        )
        self.attempts.append(att)
        return att
