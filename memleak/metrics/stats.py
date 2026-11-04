"""Statistical utilities: bootstrap CI and permutation tests."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """A metric value with confidence interval and optional breakdowns."""
    name: str = ""
    value: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n: int = 0
    breakdowns: dict[str, float] = Field(default_factory=dict)


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_resamples)
    ])
    alpha = (1 - confidence) / 2
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)))


def permutation_test(
    group_a: list[float],
    group_b: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test. Returns p-value."""
    if not group_a or not group_b:
        return 1.0
    rng = np.random.default_rng(seed)
    a = np.array(group_a)
    b = np.array(group_b)
    observed_diff = abs(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = abs(combined[:n_a].mean() - combined[n_a:].mean())
        if perm_diff >= observed_diff:
            count += 1

    return count / n_permutations
