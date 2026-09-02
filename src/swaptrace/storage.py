"""Append-only JSONL persistence for swaptrace traces.

Each :class:`~swaptrace.core.Trace` is serialized to a single JSON object and
appended as one line to a ``.jsonl`` file, so every line is independently
parseable. This module holds both directions:

Writer (Session 5):

* :func:`trace_to_dict` -- pure ``Trace -> dict`` (no I/O).
* :func:`append_trace` -- append one trace as a line, creating parent dirs.
* :class:`TraceJSONEncoder` -- renders ``datetime`` as ISO-8601.

Reader (Session 6), mirroring the same pure/impure split:

* :func:`trace_from_dict` -- pure ``dict -> Trace`` (the inverse of
  :func:`trace_to_dict`); the round-trip is lossless.
* :func:`iter_traces` / :func:`read_traces` -- read traces back from a file.

Stdlib only: ``dataclasses``, ``json``, ``pathlib``, ``datetime``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from swaptrace.core import Attempt, Trace

__all__ = [
    "trace_to_dict",
    "append_trace",
    "TraceJSONEncoder",
    "trace_from_dict",
    "iter_traces",
    "read_traces",
]

_VALID_FINAL_STATUS = frozenset({"success", "exhausted"})
_VALID_OUTCOME = frozenset({"success", "retryable_error", "non_retryable_error"})


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
class TraceJSONEncoder(json.JSONEncoder):
    """JSON encoder that renders :class:`datetime.datetime` as an ISO-8601 string.

    ``datetime`` is the only non-JSON-native type in the data model. Anything
    else falls through to ``super().default()`` and raises ``TypeError`` -- a
    loud early warning if a new non-serializable type ever enters the model,
    rather than a silent bad write.
    """

    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def trace_to_dict(trace: Trace) -> dict:
    """Convert a :class:`~swaptrace.core.Trace` to a plain ``dict``.

    Uses :func:`dataclasses.asdict`, which recurses into the nested
    :class:`~swaptrace.core.Attempt` objects in ``trace.attempts`` for free.
    ``datetime`` values are left as ``datetime`` objects here;
    :class:`TraceJSONEncoder` converts them at ``json.dumps`` time.
    """
    return dataclasses.asdict(trace)


def append_trace(trace: Trace, path: str | Path) -> Path:
    """Append ``trace`` as one JSON line to the JSONL file at ``path``.

    The parent directory is created if it does not exist, so callers need not
    pre-create anything. ``path`` is a required, explicit argument -- this layer
    has no opinion about where trace files should live (that is a CLI-session
    decision). Returns the :class:`~pathlib.Path` written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(trace_to_dict(trace), cls=TraceJSONEncoder)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
def _parse_dt(value: str | datetime | None) -> datetime | None:
    """Coerce a stored timestamp to a ``datetime``.

    ``None`` passes through. An ISO-8601 string is parsed with
    :meth:`datetime.datetime.fromisoformat`. A ``datetime`` passes through
    unchanged -- so :func:`trace_from_dict` accepts both the output of
    :func:`trace_to_dict` (``dataclasses.asdict`` keeps ``datetime`` objects)
    and a dict parsed from a JSON line (ISO strings). This is what makes the
    pure ``trace_from_dict(trace_to_dict(t)) == t`` round-trip hold.
    """
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _check_literal(value, allowed: frozenset, field_name: str):
    """Fail loudly (matching :class:`TraceJSONEncoder`) on an unrecognized
    ``Literal`` value that Python would otherwise load silently. ``None`` is
    tolerated -- it is a real dataclass default for an in-flight attempt.
    """
    if value is not None and value not in allowed:
        raise ValueError(
            f"storage: unrecognized {field_name} {value!r} "
            f"(expected one of {sorted(allowed)} or None)"
        )
    return value


def _attempt_from_dict(data: dict) -> Attempt:
    """Reconstruct one :class:`~swaptrace.core.Attempt` from its stored dict.

    Every field is passed explicitly -- no default factory runs, so the stored
    ``attempt_id`` / ``trace_id`` are preserved exactly.
    """
    return Attempt(
        attempt_id=data["attempt_id"],
        trace_id=data["trace_id"],
        provider=data["provider"],
        model=data["model"],
        attempt_index=data["attempt_index"],
        started_at=_parse_dt(data["started_at"]),
        ended_at=_parse_dt(data["ended_at"]),
        latency_ms=data["latency_ms"],
        prompt_tokens=data["prompt_tokens"],
        completion_tokens=data["completion_tokens"],
        cost_usd=data["cost_usd"],
        outcome=_check_literal(data["outcome"], _VALID_OUTCOME, "attempt outcome"),
        error_type=data["error_type"],
        error_message=data["error_message"],
    )


def trace_from_dict(data: dict) -> Trace:
    """Reconstruct a :class:`~swaptrace.core.Trace` from a dict produced by
    :func:`trace_to_dict`. The inverse conversion -- ``trace_from_dict`` of
    ``trace_to_dict`` returns an equal ``Trace``.

    Every field is passed explicitly so the stored ``trace_id`` is kept rather
    than a freshly generated one.
    """
    return Trace(
        trace_id=data["trace_id"],
        started_at=_parse_dt(data["started_at"]),
        ended_at=_parse_dt(data["ended_at"]),
        attempts=[_attempt_from_dict(a) for a in data["attempts"]],
        final_status=_check_literal(
            data["final_status"], _VALID_FINAL_STATUS, "final_status"
        ),
        winning_provider=data["winning_provider"],
        total_cost_usd=data["total_cost_usd"],
        total_latency_ms=data["total_latency_ms"],
        retry_count=data["retry_count"],
    )


def iter_traces(path: str | Path) -> Iterator[Trace]:
    """Yield each :class:`~swaptrace.core.Trace` stored in the JSONL file at ``path``.

    A path that does not exist yields nothing -- "no traces logged yet" is a
    normal state, not an error. Blank lines (e.g. a trailing newline) are
    skipped. A malformed line is *not* recovered from: ``json.JSONDecodeError``
    propagates.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield trace_from_dict(json.loads(line))


def read_traces(path: str | Path) -> list[Trace]:
    """Eager :func:`iter_traces` -- the whole file as a list."""
    return list(iter_traces(path))
