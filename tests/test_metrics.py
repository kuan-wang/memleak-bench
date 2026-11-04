"""Tests for metrics computation."""

import pytest
from memleak.metrics.stats import MetricResult, bootstrap_ci, permutation_test
from memleak.metrics.cmlr import compute_cmlr, compute_same_modal_leakage, compute_cross_modal_leakage
from memleak.metrics.das import compute_das
from memleak.metrics.entanglement import compute_entanglement_ratio, compute_entanglement_curve
from memleak.metrics.retention import compute_retention_integrity
from memleak.schema.types import (
    LeakageVerdict, ProbeResult, ProbeType,
    RetentionResult, RetentionVerdict,
)


def _probe(fact_id: str, verdict: LeakageVerdict, retract: str = "text", probe: str = "image",
           probe_type: ProbeType = ProbeType.TEXT_RETRACT_IMAGE_PROBE) -> ProbeResult:
    return ProbeResult(
        profile_id="test", fact_id=fact_id, probe_type=probe_type,
        agent_response="test", ensemble_verdict=verdict,
        retract_modality=retract, probe_modality=probe,
    )


class TestBootstrap:
    def test_empty(self):
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_single_value(self):
        lo, hi = bootstrap_ci([0.5])
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_range(self):
        lo, hi = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        assert 0.0 <= lo <= 0.5
        assert 0.5 <= hi <= 1.0


class TestPermutationTest:
    def test_identical_groups(self):
        p = permutation_test([1, 1, 1], [1, 1, 1])
        assert p == 1.0

    def test_different_groups(self):
        p = permutation_test([0, 0, 0, 0], [1, 1, 1, 1])
        assert p < 0.05  # should be significant


class TestCMLR:
    def test_all_leaked(self):
        results = [_probe(f"f{i}", LeakageVerdict.LEAKED) for i in range(10)]
        m = compute_cmlr(results)
        assert m.value == 1.0
        assert m.n == 10

    def test_none_leaked(self):
        results = [_probe(f"f{i}", LeakageVerdict.NOT_LEAKED) for i in range(10)]
        m = compute_cmlr(results)
        assert m.value == 0.0

    def test_half_leaked(self):
        results = (
            [_probe(f"f{i}", LeakageVerdict.LEAKED) for i in range(5)] +
            [_probe(f"f{i+5}", LeakageVerdict.NOT_LEAKED) for i in range(5)]
        )
        m = compute_cmlr(results)
        assert m.value == pytest.approx(0.5)

    def test_filter_by_modality(self):
        results = [
            _probe("f1", LeakageVerdict.LEAKED, retract="text", probe="image"),
            _probe("f2", LeakageVerdict.NOT_LEAKED, retract="image", probe="text"),
        ]
        t2i = compute_cmlr(results, retract_modality="text", probe_modality="image")
        assert t2i.value == 1.0
        assert t2i.n == 1

        i2t = compute_cmlr(results, retract_modality="image", probe_modality="text")
        assert i2t.value == 0.0
        assert i2t.n == 1

    def test_empty(self):
        m = compute_cmlr([])
        assert m.n == 0
        assert m.value == 0.0

    def test_breakdown_by_probe_type(self):
        results = [
            _probe("f1", LeakageVerdict.LEAKED, probe_type=ProbeType.TEXT_RETRACT_IMAGE_PROBE),
            _probe("f2", LeakageVerdict.NOT_LEAKED, probe_type=ProbeType.INVISIBLE_PERSISTENCE),
        ]
        m = compute_cmlr(results)
        assert "probe_type:text_retract_image_probe" in m.breakdowns
        assert m.breakdowns["probe_type:text_retract_image_probe"] == 1.0
        assert m.breakdowns["probe_type:invisible_persistence"] == 0.0


class TestDAS:
    def test_symmetric(self):
        results = [
            _probe("f1", LeakageVerdict.LEAKED, retract="text", probe="image"),
            _probe("f2", LeakageVerdict.LEAKED, retract="image", probe="text"),
        ]
        m = compute_das(results)
        assert m.value == 0.0  # both directions = 100% leakage

    def test_asymmetric(self):
        results = [
            _probe("f1", LeakageVerdict.LEAKED, retract="text", probe="image"),
            _probe("f2", LeakageVerdict.NOT_LEAKED, retract="image", probe="text"),
        ]
        m = compute_das(results)
        assert m.value == 1.0  # max asymmetry


class TestEntanglementRatio:
    def test_no_damage(self):
        before = [RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED)]
        after = [RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED)]
        m = compute_entanglement_ratio(before, after, 0.5)
        assert m.value == 0.0  # no accuracy drop

    def test_full_damage(self):
        before = [RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED)]
        after = [RetentionResult(fact_id="r1", verdict=RetentionVerdict.NOT_RETAINED)]
        m = compute_entanglement_ratio(before, after, 0.5)
        assert m.value == 1.0  # complete accuracy drop

    def test_with_similarity_filter(self):
        before = [
            RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED),
            RetentionResult(fact_id="r2", verdict=RetentionVerdict.RETAINED),
        ]
        after = [
            RetentionResult(fact_id="r1", verdict=RetentionVerdict.NOT_RETAINED),
            RetentionResult(fact_id="r2", verdict=RetentionVerdict.RETAINED),
        ]
        sim_map = {"r1": 0.9, "r2": 0.3}  # r1 is similar, r2 is not

        m = compute_entanglement_ratio(before, after, 0.7, sim_map)
        assert m.value == 1.0  # only r1 is above threshold, and it was damaged
        assert m.n == 1

    def test_curve(self):
        before = [RetentionResult(fact_id=f"r{i}", verdict=RetentionVerdict.RETAINED) for i in range(5)]
        after = [RetentionResult(fact_id=f"r{i}", verdict=RetentionVerdict.NOT_RETAINED) for i in range(5)]
        curve = compute_entanglement_curve(before, after)
        assert len(curve.thresholds) == 3
        assert all(er == 1.0 for er in curve.er_values)  # all damaged


class TestRetentionIntegrity:
    def test_perfect_retention(self):
        before = [RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED)]
        after = [RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED)]
        m = compute_retention_integrity(before, after)
        assert m.value == 1.0

    def test_no_retention(self):
        before = [RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED)]
        after = [RetentionResult(fact_id="r1", verdict=RetentionVerdict.NOT_RETAINED)]
        m = compute_retention_integrity(before, after)
        assert m.value == 0.0

    def test_partial(self):
        before = [
            RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED),
            RetentionResult(fact_id="r2", verdict=RetentionVerdict.RETAINED),
        ]
        after = [
            RetentionResult(fact_id="r1", verdict=RetentionVerdict.RETAINED),
            RetentionResult(fact_id="r2", verdict=RetentionVerdict.NOT_RETAINED),
        ]
        m = compute_retention_integrity(before, after)
        assert m.value == pytest.approx(0.5)
