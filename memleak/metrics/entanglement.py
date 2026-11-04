"""Entanglement Ratio (ER).

ER(τ) = (Acc_before - Acc_after) / Acc_before

Measures collateral damage: does forgetting one fact degrade recall
of semantically similar retained facts? Plotted as a curve across
similarity thresholds τ ∈ {0.5, 0.7, 0.9}.

Uses RetentionJudge verdicts (RETAINED/NOT_RETAINED), not LeakageJudge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memleak.metrics.stats import MetricResult, bootstrap_ci
from memleak.schema.types import RetentionResult, RetentionVerdict


def compute_entanglement_ratio(
    before_results: list[RetentionResult],
    after_results: list[RetentionResult],
    similarity_threshold: float,
    similarity_map: dict[str, float] | None = None,
) -> MetricResult:
    """Compute Entanglement Ratio at a given similarity threshold.

    Args:
        before_results: RetentionJudge results on retained facts BEFORE retraction.
        after_results: RetentionJudge results on the SAME retained facts AFTER retraction.
        similarity_threshold: Only include retained facts with similarity > τ to some
            forgotten fact.
        similarity_map: Dict mapping fact_id -> max similarity to any forgotten fact.
            If None, includes all facts (no filtering).

    Returns:
        MetricResult with ER value.
    """
    # Filter by similarity threshold
    if similarity_map:
        relevant_ids = {fid for fid, sim in similarity_map.items() if sim > similarity_threshold}
        before_filtered = [r for r in before_results if r.fact_id in relevant_ids]
        after_filtered = [r for r in after_results if r.fact_id in relevant_ids]
    else:
        before_filtered = before_results
        after_filtered = after_results

    if not before_filtered or not after_filtered:
        return MetricResult(name=f"ER(τ={similarity_threshold})", n=0)

    # Match by fact_id
    before_by_id = {r.fact_id: r for r in before_filtered}
    after_by_id = {r.fact_id: r for r in after_filtered}
    common_ids = set(before_by_id) & set(after_by_id)

    if not common_ids:
        return MetricResult(name=f"ER(τ={similarity_threshold})", n=0)

    acc_before_flags = [
        1.0 if before_by_id[fid].verdict == RetentionVerdict.RETAINED else 0.0
        for fid in common_ids
    ]
    acc_after_flags = [
        1.0 if after_by_id[fid].verdict == RetentionVerdict.RETAINED else 0.0
        for fid in common_ids
    ]

    acc_before = sum(acc_before_flags) / len(acc_before_flags)
    acc_after = sum(acc_after_flags) / len(acc_after_flags)

    if acc_before == 0:
        er = 0.0  # can't compute ratio if baseline accuracy is 0
    else:
        er = (acc_before - acc_after) / acc_before

    return MetricResult(
        name=f"ER(τ={similarity_threshold})",
        value=er,
        n=len(common_ids),
        breakdowns={
            "acc_before": acc_before,
            "acc_after": acc_after,
            "threshold": similarity_threshold,
        },
    )


@dataclass
class EntanglementCurve:
    """ER values at multiple similarity thresholds for plotting."""
    thresholds: list[float] = field(default_factory=list)
    er_values: list[float] = field(default_factory=list)
    n_values: list[int] = field(default_factory=list)


def compute_entanglement_curve(
    before_results: list[RetentionResult],
    after_results: list[RetentionResult],
    similarity_map: dict[str, float] | None = None,
    thresholds: list[float] | None = None,
) -> EntanglementCurve:
    """Compute ER at multiple thresholds for the entanglement curve."""
    thresholds = thresholds or [0.5, 0.7, 0.9]
    curve = EntanglementCurve()

    for tau in thresholds:
        result = compute_entanglement_ratio(
            before_results, after_results, tau, similarity_map
        )
        curve.thresholds.append(tau)
        curve.er_values.append(result.value)
        curve.n_values.append(result.n)

    return curve
