"""Measure swaptrace's instrumentation overhead per call.

The project's spec asks for a *measured*, not estimated, figure for the added
latency of wrapping a provider call in swaptrace's ``Trace`` / ``Attempt``
bookkeeping. This script produces that number and writes the full raw dataset
so anyone can recompute the statistics independently.

What is measured
----------------
The **standalone core API only**::

    with Trace() as trace:
        with trace.attempt(provider="mock", model="mock") as attempt:
            response = mock_provider_call()
            attempt.record_success(response, prompt_tokens=10, completion_tokens=20)

vs. a bare ``mock_provider_call()``. This is one traced provider call in the
common single-attempt (first provider succeeds) shape -- so the per-call number
*includes* one ``Trace()`` construction + ``_finalize()``. A trace that wraps a
multi-attempt cascade amortizes that fixed cost over its attempts.

Deliberately NOT measured: the swapLLM adapter (this is about the core layer in
isolation) and ``storage.append_trace`` (disk I/O has its own highly variable
cost; folding it in would make the number meaningless).

Framing the number
------------------
swaptrace's overhead is a fixed cost of its own bookkeeping (a couple of
``uuid4()`` calls, ``datetime.now()`` calls, the ``_finalize()`` rollup, the
pricing dict merge) -- it does not depend on what the wrapped call does. So the
honest way to state it is as a fraction of a *real* LLM API call latency
(``REFERENCE_LATENCIES_S``: 100 ms - 2 s), computed by arithmetic from the
measured absolute overhead. The ratio against this script's no-op baseline is
also recorded but is not representative -- the baseline is a function that just
returns a string.

Run: ``python benchmarks/benchmark_overhead.py``
Stdlib only: ``gc``, ``json``, ``platform``, ``re``, ``statistics``, ``time``,
``datetime``, ``pathlib``.
"""

from __future__ import annotations

import gc
import json
import platform
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from swaptrace import Trace

N_WARMUP = 100
N_ITERATIONS = 1000
N_TRIALS = 5

# Realistic LLM API call latencies to express the overhead against (seconds).
# 100 ms - 2 s spans fast small models to slow large ones. No calls are actually
# made at these latencies -- the percentages are pure arithmetic on the measured
# absolute overhead.
REFERENCE_LATENCIES_S = [0.1, 0.2, 0.5, 1.0, 2.0]

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def mock_provider_call() -> str:
    """A stand-in provider call that does essentially zero work, so all measured
    time reflects swaptrace's overhead rather than the mock's."""
    return "ok"


def run_untraced_trial(n: int) -> list[float]:
    """Time ``n`` bare ``mock_provider_call()`` invocations, one by one."""
    perf = time.perf_counter
    samples: list[float] = []
    for _ in range(n):
        start = perf()
        mock_provider_call()
        samples.append(perf() - start)
    return samples


def run_traced_trial(n: int) -> list[float]:
    """Time ``n`` fully ``Trace``/``Attempt``-wrapped calls, one by one."""
    perf = time.perf_counter
    samples: list[float] = []
    for _ in range(n):
        start = perf()
        with Trace() as trace:
            with trace.attempt(provider="mock", model="mock") as attempt:
                response = mock_provider_call()
                attempt.record_success(
                    response, prompt_tokens=10, completion_tokens=20
                )
        samples.append(perf() - start)
    return samples


def _summarize(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "median_s": statistics.median(samples),
        # 95th percentile: quantiles(..., n=100) returns the 99 cut points
        # between percentiles; index 94 is the boundary at the 95th.
        "p95_s": statistics.quantiles(samples, n=100)[94],
        "mean_s": statistics.fmean(samples),
        "min_s": min(samples),
        "max_s": max(samples),
    }


def _platform_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _results_path(info: dict, when: datetime) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", info["platform"]).strip("-")
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    return _RESULTS_DIR / f"overhead_{slug}_{stamp}.json"


def main() -> None:
    info = _platform_info()
    when = datetime.now(timezone.utc)

    # --- Warm-up: exercise both paths and DISCARD the timings. The interpreter,
    # attribute-lookup caches, etc. make the first calls slower than steady
    # state; measuring from cold would bias the number upward. (No assignment
    # here -- the returned lists are thrown away.)
    run_untraced_trial(N_WARMUP)
    run_traced_trial(N_WARMUP)

    # --- Measured portion: GC off so a collection pause can't land mid-trial
    # and masquerade as swaptrace overhead. Re-enabled in `finally` no matter
    # what.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        untraced_trials = [run_untraced_trial(N_ITERATIONS) for _ in range(N_TRIALS)]
        traced_trials = [run_traced_trial(N_ITERATIONS) for _ in range(N_TRIALS)]
    finally:
        if gc_was_enabled:
            gc.enable()

    untraced_per_trial = [_summarize(t) for t in untraced_trials]
    traced_per_trial = [_summarize(t) for t in traced_trials]
    untraced_pooled = _summarize([s for t in untraced_trials for s in t])
    traced_pooled = _summarize([s for t in traced_trials for s in t])

    overhead_median_s = traced_pooled["median_s"] - untraced_pooled["median_s"]
    overhead_p95_s = traced_pooled["p95_s"] - untraced_pooled["p95_s"]
    overhead_mean_s = traced_pooled["mean_s"] - untraced_pooled["mean_s"]
    overhead = {
        "median_abs_s": overhead_median_s,
        "median_abs_us": overhead_median_s * 1e6,
        "p95_abs_s": overhead_p95_s,
        "p95_abs_us": overhead_p95_s * 1e6,
        "mean_abs_s": overhead_mean_s,
        "mean_abs_us": overhead_mean_s * 1e6,
        # Overhead as a fraction of a real LLM API call latency. THIS is the
        # headline framing.
        "pct_of_realistic_latency": {
            "reference_latencies_s": REFERENCE_LATENCIES_S,
            "median": {
                f"{r}s": overhead_median_s / r * 100 for r in REFERENCE_LATENCIES_S
            },
            "p95": {
                f"{r}s": overhead_p95_s / r * 100 for r in REFERENCE_LATENCIES_S
            },
        },
        # Ratio against this script's no-op baseline (~0.04 us). Recorded for
        # completeness, but NOT representative of real-world overhead -- the
        # baseline is a function that just returns a string. Use
        # pct_of_realistic_latency instead.
        "pct_of_synthetic_zero_cost_baseline": {
            "median": overhead_median_s / untraced_pooled["median_s"] * 100,
            "p95": overhead_p95_s / untraced_pooled["p95_s"] * 100,
            "note": (
                "NOT representative of real-world overhead -- baseline is a "
                "no-op function; see pct_of_realistic_latency"
            ),
        },
    }

    payload = {
        "kind": "swaptrace-instrumentation-overhead",
        "status": "final",
        "timestamp_utc": when.isoformat(),
        "config": {
            "n_warmup": N_WARMUP,
            "n_iterations": N_ITERATIONS,
            "n_trials": N_TRIALS,
            "gc_disabled_during_measurement": True,
            "timer": "time.perf_counter",
        },
        "platform": info,
        "raw_timings_s": {
            "untraced": untraced_trials,
            "traced": traced_trials,
        },
        "per_trial": {
            "untraced": untraced_per_trial,
            "traced": traced_per_trial,
        },
        "pooled": {
            "untraced": untraced_pooled,
            "traced": traced_pooled,
        },
        "overhead": overhead,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _results_path(info, when)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    us = lambda s: f"{s * 1e6:9.3f} us"  # noqa: E731 -- local formatting shortcut
    print(f"swaptrace instrumentation overhead ({payload['status']})")
    print("=" * 62)
    print(f"  python   : {info['python_implementation']} {info['python_version']}")
    print(f"  platform : {info['platform']}")
    print(f"  when     : {when.isoformat()}")
    print(f"  config   : {N_TRIALS} trials x {N_ITERATIONS} iters "
          f"(+{N_WARMUP} warm-up, GC off, perf_counter)")
    print()
    print(f"  {'':>10}  {'median':>12}  {'p95':>12}  {'mean':>12}")
    print(f"  {'untraced':>10}  {us(untraced_pooled['median_s'])}  "
          f"{us(untraced_pooled['p95_s'])}  {us(untraced_pooled['mean_s'])}")
    print(f"  {'traced':>10}  {us(traced_pooled['median_s'])}  "
          f"{us(traced_pooled['p95_s'])}  {us(traced_pooled['mean_s'])}")
    print()
    print(f"  ABSOLUTE overhead : {overhead['median_abs_us']:.3f} us/call median, "
          f"{overhead['p95_abs_us']:.3f} us/call p95")
    print()
    print("  as % of a realistic LLM API call latency  (the headline framing):")
    print(f"      {'call latency':>14}  {'median':>10}  {'p95':>10}")
    for r in REFERENCE_LATENCIES_S:
        key = f"{r}s"
        print(f"      {str(int(r * 1000)) + ' ms':>14}  "
              f"{overhead['pct_of_realistic_latency']['median'][key]:>9.4f}%  "
              f"{overhead['pct_of_realistic_latency']['p95'][key]:>9.4f}%")
    print()
    synth = overhead["pct_of_synthetic_zero_cost_baseline"]
    print(f"  (synthetic no-op-baseline ratio: {synth['median']:.0f}% median -- "
          "NOT representative; baseline is a no-op fn)")
    print()
    print("  per-trial traced medians (us):",
          ", ".join(f"{t['median_s'] * 1e6:.3f}" for t in traced_per_trial))
    print("  per-trial untraced medians (us):",
          ", ".join(f"{t['median_s'] * 1e6:.3f}" for t in untraced_per_trial))
    print()
    print(f"  raw data -> {out_path}")


if __name__ == "__main__":
    main()
