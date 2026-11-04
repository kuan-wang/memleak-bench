"""Directional Asymmetry Score (DAS).

DAS = |CMLR(text→image) - CMLR(image→text)|

Measures whether forgetting is symmetric across modalities.
High DAS = one direction is structurally harder to forget.
"""

from __future__ import annotations

from memleak.metrics.cmlr import compute_cmlr
from memleak.metrics.stats import MetricResult, permutation_test
from memleak.schema.types import LeakageVerdict, ProbeResult


def compute_das(results: list[ProbeResult]) -> MetricResult:
    """Compute Directional Asymmetry Score with significance test.

    Returns:
        MetricResult with DAS value, p-value in breakdowns.
    """
    text_to_image = compute_cmlr(results, retract_modality="text", probe_modality="image")
    image_to_text = compute_cmlr(results, retract_modality="image", probe_modality="text")

    das = abs(text_to_image.value - image_to_text.value)

    # Permutation test for significance
    t2i_flags = [
        1.0 if r.ensemble_verdict == LeakageVerdict.LEAKED else 0.0
        for r in results
        if r.retract_modality == "text" and r.probe_modality == "image"
    ]
    i2t_flags = [
        1.0 if r.ensemble_verdict == LeakageVerdict.LEAKED else 0.0
        for r in results
        if r.retract_modality == "image" and r.probe_modality == "text"
    ]

    p_value = permutation_test(t2i_flags, i2t_flags) if t2i_flags and i2t_flags else 1.0

    return MetricResult(
        name="DAS",
        value=das,
        n=text_to_image.n + image_to_text.n,
        breakdowns={
            "cmlr_text_to_image": text_to_image.value,
            "cmlr_image_to_text": image_to_text.value,
            "p_value": p_value,
            "n_text_to_image": float(text_to_image.n),
            "n_image_to_text": float(image_to_text.n),
        },
    )
