"""Cross-Modal Leakage Rate (CMLR).

CMLR(A→B) = |{f: retracted via A AND leaked via B}| / |F|

The core metric: what fraction of forgotten facts are recoverable
through a modality different from the retraction modality.
"""

from __future__ import annotations

from memleak.metrics.stats import MetricResult, bootstrap_ci
from memleak.schema.types import LeakageVerdict, ProbeResult


def compute_cmlr(
    results: list[ProbeResult],
    retract_modality: str | None = None,
    probe_modality: str | None = None,
) -> MetricResult:
    """Compute Cross-Modal Leakage Rate.

    Args:
        results: List of ProbeResult objects (already judged).
        retract_modality: Filter to retraction modality ("text" or "image"). None = all.
        probe_modality: Filter to probe modality ("text" or "image"). None = all.

    Returns:
        MetricResult with CMLR value, 95% CI, and per-category breakdowns.
    """
    filtered = results
    if retract_modality:
        filtered = [r for r in filtered if r.retract_modality == retract_modality]
    if probe_modality:
        filtered = [r for r in filtered if r.probe_modality == probe_modality]

    if not filtered:
        return MetricResult(name="CMLR", n=0)

    # Binary: 1 if leaked, 0 if not
    leakage_flags = [
        1.0 if r.ensemble_verdict == LeakageVerdict.LEAKED else 0.0
        for r in filtered
    ]

    rate = sum(leakage_flags) / len(leakage_flags)
    ci = bootstrap_ci(leakage_flags)

    # Per-probe-type breakdown
    breakdowns: dict[str, float] = {}
    by_type: dict[str, list[float]] = {}
    for r, flag in zip(filtered, leakage_flags):
        key = r.probe_type.value
        by_type.setdefault(key, []).append(flag)
    for key, flags in by_type.items():
        breakdowns[f"probe_type:{key}"] = sum(flags) / len(flags) if flags else 0.0

    return MetricResult(
        name="CMLR",
        value=rate,
        ci_lower=ci[0],
        ci_upper=ci[1],
        n=len(filtered),
        breakdowns=breakdowns,
    )


def compute_same_modal_leakage(results: list[ProbeResult]) -> MetricResult:
    """Compute same-modal leakage rate (baseline: retract and probe in same modality)."""
    same = [r for r in results if r.retract_modality == r.probe_modality]
    return compute_cmlr(same)


def compute_cross_modal_leakage(results: list[ProbeResult]) -> MetricResult:
    """Compute cross-modal leakage rate (retract and probe in different modalities)."""
    cross = [r for r in results if r.retract_modality != r.probe_modality]
    return compute_cmlr(cross)
