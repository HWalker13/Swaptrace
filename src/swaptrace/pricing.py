"""Per-token cost estimation for swaptrace attempts.

A small bundled price table, a JSON override loader, and a cost estimator that
returns ``None`` rather than raising when it can't price something. Wired into
:meth:`swaptrace.core.Attempt.record_success`.

Stdlib only: ``json``, ``pathlib``.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["DEFAULT_PRICING", "load_overrides", "estimate_cost_usd"]

# Price is USD per 1,000,000 tokens: (input_rate, output_rate).
# Verified 2026-09-02 against multiple independent trackers. Bundled prices
# are a convenience starting point, NOT a source of truth for real spend --
# provider pricing changes frequently. Case in point: at least one tracker
# reported Groq's Llama 3.1 8B Instant moved to Enterprise-only pricing on
# 2026-08-26, while five others (including one synced the same day this
# table was built) still showed it active. This exact kind of discrepancy
# is why override support exists below -- always verify against the
# provider's live pricing page before relying on this for a real budget.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "llama-3.1-8b-instant": (0.05, 0.08),   # Groq
    "gpt-4o-mini": (0.15, 0.60),             # OpenAI
    "claude-haiku-4-5": (1.00, 5.00),        # Anthropic
}


def load_overrides(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load model -> ``(input_rate, output_rate)`` overrides from a JSON file.

    The file is shaped ``{"model-name": [input_rate, output_rate], ...}``. A
    missing file returns ``{}`` -- "no overrides configured" is a normal state,
    not an error (same principle as ``iter_traces`` on a missing path). ``path``
    is a required, explicit argument; this module has no opinion about where an
    overrides file should live.
    """
    path = Path(path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        model: (float(rates[0]), float(rates[1]))
        for model, rates in raw.items()
    }


def estimate_cost_usd(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    overrides: dict | None = None,
) -> float | None:
    """Estimate the USD cost of one attempt, or ``None`` when it can't be priced.

    Returns ``None`` -- never raises -- if either token count is ``None`` or if
    ``model`` appears in neither ``overrides`` nor :data:`DEFAULT_PRICING`. An
    uncatalogued model is a normal, expected condition, not a data-integrity
    bug: crashing a whole trace over an unpriced model would be the wrong
    failure mode. (This is deliberately a different philosophy from the
    storage layer's fail-loud handling of *corrupted* data.)

    ``overrides`` is merged on top of the bundled table, so a partial override
    of one model leaves the others intact.
    """
    if prompt_tokens is None or completion_tokens is None:
        return None
    table = {**DEFAULT_PRICING, **(overrides or {})}
    rates = table.get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
