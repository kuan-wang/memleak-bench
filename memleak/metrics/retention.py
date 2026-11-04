"""Retention Integrity (RI).

RI = Acc(unrelated retained facts | after) / Acc(unrelated retained facts | before)

Measures whether forgetting causes collateral damage to unrelated retained facts.
RI = 1.0 means no damage; RI < 1.0 means some retained facts were lost.
"""

from __future__ import annotations

from memleak.metrics.stats import MetricResult
from memleak.schema.types import RetentionResult, RetentionVerdict


def compute_retention_integrity(
    before_results: list[RetentionResult],
    after_results: list[RetentionResult],
) -> MetricResult:
    """Compute Retention Integrity.

    Args:
        before_results: RetentionJudge results on retained facts BEFORE retraction.
        after_results: RetentionJudge results on the SAME retained facts AFTER retraction.

    Returns:
        MetricResult with RI value.
    """
    if not before_results or not after_results:
        return MetricResult(name="RI", n=0)

    before_by_id = {r.fact_id: r for r in before_results}
    after_by_id = {r.fact_id: r for r in after_results}
    common = set(before_by_id) & set(after_by_id)

    if not common:
        return MetricResult(name="RI", n=0)

    acc_before = sum(
        1 for fid in common
        if before_by_id[fid].verdict == RetentionVerdict.RETAINED
    ) / len(common)

    acc_after = sum(
        1 for fid in common
        if after_by_id[fid].verdict == RetentionVerdict.RETAINED
    ) / len(common)

    ri = acc_after / acc_before if acc_before > 0 else 1.0

    return MetricResult(
        name="RI",
        value=ri,
        n=len(common),
        breakdowns={"acc_before": acc_before, "acc_after": acc_after},
    )
