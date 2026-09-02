"""Append-only JSONL persistence for swaptrace traces.

Session 5 scope: the **writer** only. Each :class:`~swaptrace.core.Trace` is
serialized to a single JSON object and appended as one line to a ``.jsonl``
file, so every line is independently parseable. The matching reader is Session 6.

Serialization is split in two on purpose:

* :func:`trace_to_dict` -- pure ``Trace -> dict``, no I/O, trivially testable and
  the shape Session 6's reader parses back.
* :func:`append_trace` -- the filesystem side (open, append, newline, mkdir).

Stdlib only: ``dataclasses``, ``json``, ``pathlib``, ``datetime``.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

from swaptrace.core import Trace

__all__ = ["trace_to_dict", "append_trace", "TraceJSONEncoder"]


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
