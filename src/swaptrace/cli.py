"""``swaptrace`` command-line interface.

Subcommands:

* ``swaptrace query`` -- read a JSONL trace file and print the *traces* that
  match some filters.
* ``swaptrace report --compare-providers`` -- flatten every *attempt* across
  every trace, group by provider, and print a reliability / latency / cost
  comparison.

Trace files
-----------
swaptrace never writes traces on its own. The library records a
:class:`~swaptrace.core.Trace` per call; persisting it is the caller's job,
typically by wiring :func:`swaptrace.storage.append_trace` into the ``on_trace``
callback of :func:`swaptrace.integrations.swapllm.traced`::

    from swaptrace import storage
    from swaptrace.integrations.swapllm import traced

    path = ".swaptrace/traces.jsonl"
    router = traced(router, on_trace=lambda t: storage.append_trace(t, path))

``swaptrace query`` reads ``.swaptrace/traces.jsonl`` relative to the current
directory by default (see :data:`DEFAULT_TRACE_PATH`), overridable with
``--path``. A fresh install has nothing to query until something has written
there.

Stdlib only: ``argparse``, ``pathlib``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from swaptrace.core import Attempt, Trace
from swaptrace.storage import read_traces

__all__ = [
    "main",
    "build_parser",
    "trace_matches",
    "compare_providers",
    "ProviderStats",
    "DEFAULT_TRACE_PATH",
]

# Where ``swaptrace query`` looks by default. Relative to the working directory:
# traces are meaningful per-project (which app, which repo), not global -- the
# same way ``.git/`` is per-repo, not per-machine. ``storage.py`` stays
# opinion-free about file locations; ``cli.py`` is where that opinion belongs.
DEFAULT_TRACE_PATH = ".swaptrace/traces.jsonl"

_STATUS_CHOICES = ["success", "retryable_error", "non_retryable_error"]


def trace_matches(
    trace: Trace,
    *,
    provider: str | None = None,
    status: str | None = None,
    min_cost: float | None = None,
) -> bool:
    """Whether ``trace`` passes every active filter (unset filters are ignored;
    active ones combine with AND).

    ``provider`` and ``status`` are attempt-level fields: a trace matches if
    *any* of its attempts matches. ``min_cost`` is checked against the trace's
    rolled-up ``total_cost_usd``.
    """
    if provider is not None and not any(a.provider == provider for a in trace.attempts):
        return False
    if status is not None and not any(a.outcome == status for a in trace.attempts):
        return False
    if min_cost is not None and (
        trace.total_cost_usd is None or trace.total_cost_usd < min_cost
    ):
        return False
    return True


def _format_trace_line(trace: Trace) -> str:
    started = trace.started_at.isoformat() if trace.started_at is not None else "?"
    short_id = trace.trace_id[:8]
    status = (trace.final_status or "?").ljust(9)
    winner = trace.winning_provider or "-"
    n = len(trace.attempts)
    cost = f"${trace.total_cost_usd:.6f}"
    latency = f"{trace.total_latency_ms:.1f}ms"
    return (
        f"{started}  {short_id}  {status}  {winner:<12}  "
        f"{n} attempt(s)  {cost}  {latency}"
    )


def cmd_query(args: argparse.Namespace) -> int:
    traces = read_traces(args.path)
    matches = [
        t
        for t in traces
        if trace_matches(
            t, provider=args.provider, status=args.status, min_cost=args.min_cost
        )
    ]
    if not matches:
        # Missing file, empty file, or nothing matched -- all a normal "nothing
        # to show" state, not an error (mirrors iter_traces on a missing path).
        print("No traces found.")
        return 0
    for trace in matches:
        print(_format_trace_line(trace))
    print(f"{len(matches)} of {len(traces)} trace(s).")
    return 0


# --------------------------------------------------------------------------- #
# report --compare-providers
# --------------------------------------------------------------------------- #
@dataclass
class ProviderStats:
    """Per-provider aggregate over every attempt that used it. Derived and
    thrown away after printing -- not part of the persisted data model.
    """

    provider: str
    total_attempts: int
    successes: int
    success_rate: float  # successes / total_attempts
    avg_latency_ms: float  # mean over ALL attempts, success or failure
    total_cost_usd: float  # sum of cost_usd, None treated as 0.0
    avg_cost_per_success_usd: float | None  # None when successes == 0


def compare_providers(traces: list[Trace]) -> list[ProviderStats]:
    """Flatten every attempt across ``traces``, group by ``attempt.provider``,
    and return one :class:`ProviderStats` per provider, sorted by
    ``success_rate`` descending (ties broken by provider name).
    """
    groups: dict[str, list[Attempt]] = {}
    for trace in traces:
        for attempt in trace.attempts:
            groups.setdefault(attempt.provider, []).append(attempt)

    stats: list[ProviderStats] = []
    for provider, attempts in groups.items():
        total = len(attempts)
        successes = sum(1 for a in attempts if a.outcome == "success")
        latencies = [a.latency_ms for a in attempts if a.latency_ms is not None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        total_cost = sum(a.cost_usd or 0.0 for a in attempts)
        stats.append(
            ProviderStats(
                provider=provider,
                total_attempts=total,
                successes=successes,
                success_rate=successes / total,
                avg_latency_ms=avg_latency,
                total_cost_usd=total_cost,
                avg_cost_per_success_usd=(
                    total_cost / successes if successes else None
                ),
            )
        )
    stats.sort(key=lambda s: (-s.success_rate, s.provider))
    return stats


_REPORT_COLUMNS = ("PROVIDER", "ATTEMPTS", "SUCCESSES", "SUCCESS%", "AVG LATENCY", "TOTAL COST", "$/SUCCESS")


def _report_row(provider, attempts, successes, success_pct, latency, total_cost, per_success) -> str:
    return (
        f"{provider:<12}{attempts:>9}{successes:>11}{success_pct:>10}"
        f"{latency:>14}{total_cost:>14}{per_success:>13}"
    )


def _format_stats_row(s: ProviderStats) -> str:
    per_success = (
        "-"
        if s.avg_cost_per_success_usd is None
        else f"${s.avg_cost_per_success_usd:.6f}"
    )
    return _report_row(
        s.provider,
        s.total_attempts,
        s.successes,
        f"{s.success_rate * 100:.1f}%",
        f"{s.avg_latency_ms:.1f}ms",
        f"${s.total_cost_usd:.6f}",
        per_success,
    )


def cmd_report(args: argparse.Namespace) -> int:
    if not args.compare_providers:
        print("No report type specified. Try --compare-providers.")
        return 2
    stats = compare_providers(read_traces(args.path))
    if not stats:
        print("No traces found.")
        return 0
    print(_report_row(*_REPORT_COLUMNS))
    for row in stats:
        print(_format_stats_row(row))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swaptrace", description="Trace and compare LLM swap attempts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser(
        "query",
        help="print traces from a JSONL file, optionally filtered",
        description=(
            "Read a JSONL trace file and print the traces matching the given "
            "filters (combined with AND). Prints 'No traces found.' when the "
            "file is missing or nothing matches. swaptrace does not write this "
            "file itself -- wire swaptrace.storage.append_trace into a traced() "
            "router's on_trace callback."
        ),
    )
    query.add_argument(
        "--path",
        default=DEFAULT_TRACE_PATH,
        help=f"trace file to read (default: {DEFAULT_TRACE_PATH})",
    )
    query.add_argument(
        "--provider",
        metavar="NAME",
        help="keep traces with at least one attempt on this provider",
    )
    query.add_argument(
        "--status",
        choices=_STATUS_CHOICES,
        help="keep traces with at least one attempt of this outcome",
    )
    query.add_argument(
        "--min-cost",
        type=float,
        metavar="USD",
        help="keep traces whose total_cost_usd is >= this value",
    )

    report = subparsers.add_parser(
        "report",
        help="aggregate stats across traces",
        description=(
            "Aggregate statistics across a JSONL trace file. Prints 'No traces "
            "found.' when the file is missing or empty. Currently one mode: "
            "--compare-providers."
        ),
    )
    report.add_argument(
        "--path",
        default=DEFAULT_TRACE_PATH,
        help=f"trace file to read (default: {DEFAULT_TRACE_PATH})",
    )
    report.add_argument(
        "--compare-providers",
        action="store_true",
        help="per-provider reliability / latency / cost comparison, "
        "sorted by success rate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``swaptrace`` console script. Returns an exit code;
    the setuptools-generated wrapper passes it to ``sys.exit()``.
    """
    args = build_parser().parse_args(argv)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "report":
        return cmd_report(args)
    return 1  # unreachable -- subparsers are required


if __name__ == "__main__":
    import sys

    sys.exit(main())
