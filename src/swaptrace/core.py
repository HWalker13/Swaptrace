"""Core data model for swaptrace.

Session 2 scope: the :class:`Trace` and :class:`Attempt` dataclasses plus
*skeleton* context managers. The success / failure / exhaustion branching and
the roll-up of per-attempt numbers into the parent :class:`Trace` are left as
``# TODO: Session 3`` markers and ``NotImplementedError`` stubs on purpose --
Session 3 builds against this, it does not inherit a finished implementation.

Stdlib only: ``dataclasses``, ``uuid``, ``datetime``, ``typing.Literal``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

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

        with trace.attempt("openai", "gpt-4o") as attempt:
            response = call_provider(...)
            attempt.record_success(prompt_tokens=12, completion_tokens=34)

    ``__enter__`` stamps ``started_at``; ``__exit__`` stamps ``ended_at`` and
    ``latency_ms``. The remaining fields are populated by
    :meth:`record_success` / :meth:`record_failure` (Session 3).

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
        # TODO: Session 3 -- if an exception propagated (exc is not None) and
        # record_failure/record_success was never called, classify it here.
        return False

    def record_success(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Mark this attempt successful and store token/cost data.

        Stub -- Session 3 owns the real logic (set ``outcome = "success"``,
        store the numbers, signal the parent trace to stop retrying).
        """
        raise NotImplementedError("Session 3")

    def record_failure(
        self,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        retryable: bool,
    ) -> None:
        """Mark this attempt failed.

        ``retryable=True`` -> ``outcome = "retryable_error"`` (the trace may try
        the next provider); ``retryable=False`` -> ``outcome =
        "non_retryable_error"`` (the trace stops).

        Stub -- Session 3 owns the branching and the exhaustion detection.
        """
        raise NotImplementedError("Session 3")


@dataclass(kw_only=True)
class Trace:
    """Top-level record of one swap sequence: the ordered attempts made to get
    a usable LLM response, plus the rolled-up totals.

    Used as a context manager::

        with Trace() as trace:
            with trace.attempt("openai", "gpt-4o") as attempt:
                ...

    ``__enter__`` stamps ``started_at``; ``__exit__`` stamps ``ended_at``. The
    roll-up fields (``final_status``, ``winning_provider``, ``total_cost_usd``,
    ``total_latency_ms``, ``retry_count``) are computed in Session 3 and are
    left unset / zeroed here.
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
        # TODO: Session 3 -- aggregate self.attempts into final_status
        # ("success" vs "exhausted"), winning_provider, total_cost_usd,
        # total_latency_ms and retry_count.
        return False

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
