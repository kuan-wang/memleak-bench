"""Tests for schema types and validators."""

import json

import pytest

from memleak.schema.types import (
    EntangledFact,
    Fact,
    FactCategory,
    ImageEntry,
    Probe,
    ProbeType,
    Profile,
    RetractionPoint,
    RunConfig,
    Turn,
    VisualGrounding,
)
from memleak.schema.validators import ValidationError, validate_profile


def _make_fact(
    idx: int,
    category: FactCategory,
    forget: bool,
    grounding: VisualGrounding = VisualGrounding.EXPLICIT,
) -> Fact:
    """Helper: create a minimal valid fact."""
    images = []
    if grounding != VisualGrounding.ABSENT:
        images = [ImageEntry(path=f"PLACEHOLDER:img_{idx}", grounding_type=grounding)]

    entangled = []
    probes = []
    if forget:
        entangled = [
            EntangledFact(fact_id=f"ent_{idx}_low", similarity_level=0.5),
            EntangledFact(fact_id=f"ent_{idx}_med", similarity_level=0.7),
            EntangledFact(fact_id=f"ent_{idx}_high", similarity_level=0.9),
        ]
        probes = [
            Probe(
                probe_id=f"probe_{idx}",
                probe_type=ProbeType.TEXT_RETRACT_IMAGE_PROBE,
                query=f"Do you know about fact {idx}?",
            )
        ]

    return Fact(
        fact_id=f"fact_{idx:03d}",
        text=f"Fact number {idx} about the entity",
        category=category,
        visual_grounding=grounding,
        images=images,
        forget_target=forget,
        entangled_facts=entangled,
        probes=probes,
    )


def _make_valid_profile() -> Profile:
    """Create a profile that passes all validators."""
    categories = list(FactCategory)
    facts = []
    idx = 0

    # Grounding assignments for forget targets (10 total):
    # 4 explicit, 4 implicit, 2 absent
    forget_groundings = (
        [VisualGrounding.EXPLICIT] * 4
        + [VisualGrounding.IMPLICIT] * 4
        + [VisualGrounding.ABSENT] * 2
    )
    fg_idx = 0

    for cat in categories:
        # 2 forget targets per category
        for _ in range(2):
            facts.append(_make_fact(idx, cat, forget=True, grounding=forget_groundings[fg_idx]))
            fg_idx += 1
            idx += 1
        # 2 retained per category
        for _ in range(2):
            facts.append(_make_fact(idx, cat, forget=False, grounding=VisualGrounding.EXPLICIT))
            idx += 1

    # Interaction stream that introduces all facts before retraction
    turns = []
    for i, fact in enumerate(facts):
        turns.append(
            Turn(
                turn_id=i,
                role="user",
                text=f"Here is info: {fact.text}",
                facts_introduced=[fact.fact_id],
            )
        )
    turns.append(Turn(turn_id=len(facts), role="user", text="Filler turn"))

    retraction = RetractionPoint(
        turn_id=len(facts) + 1,
        instruction="Please forget the targeted facts.",
        target_facts=[f.fact_id for f in facts if f.forget_target],
    )

    return Profile(
        profile_id="test_001",
        name="Test Entity",
        facts=facts,
        interaction_stream=turns,
        retraction_point=retraction,
    )


class TestValidProfile:
    def test_valid_profile_passes(self):
        profile = _make_valid_profile()
        errors = validate_profile(profile, strict=False)
        assert errors == []

    def test_valid_profile_strict(self):
        profile = _make_valid_profile()
        validate_profile(profile, strict=True)  # should not raise


class TestFactCount:
    def test_wrong_fact_count(self):
        profile = _make_valid_profile()
        profile.facts = profile.facts[:15]
        errors = validate_profile(profile, strict=False)
        assert any("Expected 20 facts" in e for e in errors)

    def test_wrong_forget_count(self):
        profile = _make_valid_profile()
        # Make one more fact a forget target
        profile.facts[19].forget_target = True
        profile.facts[19].entangled_facts = [
            EntangledFact(fact_id="x", similarity_level=0.5)
        ]
        profile.facts[19].probes = [
            Probe(probe_id="p", probe_type=ProbeType.TEXT_RETRACT_IMAGE_PROBE, query="?")
        ]
        errors = validate_profile(profile, strict=False)
        assert any("Expected 10 forget targets" in e for e in errors)


class TestCategoryBalance:
    def test_imbalanced_categories(self):
        profile = _make_valid_profile()
        # Change a fact's category to break balance
        profile.facts[0].category = FactCategory.POSSESSION
        errors = validate_profile(profile, strict=False)
        assert any("Category" in e for e in errors)


class TestForgetTargetRequirements:
    def test_forget_target_without_entangled(self):
        profile = _make_valid_profile()
        profile.facts[0].entangled_facts = []
        errors = validate_profile(profile, strict=False)
        assert any("no entangled facts" in e for e in errors)

    def test_forget_target_without_probes(self):
        profile = _make_valid_profile()
        profile.facts[0].probes = []
        errors = validate_profile(profile, strict=False)
        assert any("no probes" in e for e in errors)


class TestUniqueFactIds:
    def test_duplicate_fact_ids(self):
        profile = _make_valid_profile()
        profile.facts[1].fact_id = profile.facts[0].fact_id
        errors = validate_profile(profile, strict=False)
        assert any("Duplicate fact IDs" in e for e in errors)


class TestRetractionPoint:
    def test_retraction_targets_unknown_fact(self):
        profile = _make_valid_profile()
        profile.retraction_point.target_facts.append("nonexistent_fact")
        errors = validate_profile(profile, strict=False)
        assert any("unknown fact" in e for e in errors)

    def test_retraction_targets_non_forget_fact(self):
        profile = _make_valid_profile()
        retained_id = next(f.fact_id for f in profile.facts if not f.forget_target)
        profile.retraction_point.target_facts.append(retained_id)
        errors = validate_profile(profile, strict=False)
        assert any("non-forget fact" in e for e in errors)


class TestChainValidation:
    def test_identical_chain_answers(self):
        profile = _make_valid_profile()
        from memleak.schema.types import ChainHop, MultiHopChain

        profile.facts[0].cross_modal_chains = [
            MultiHopChain(
                chain_id="c1",
                hops=[ChainHop(fact_id="fact_000", modality="text")],
                answer_if_remembered="Paris",
                answer_if_forgotten="Paris",  # same — should fail
            )
        ]
        errors = validate_profile(profile, strict=False)
        assert any("identical" in e for e in errors)


class TestSerialization:
    def test_roundtrip_json(self):
        profile = _make_valid_profile()
        json_str = profile.model_dump_json(indent=2)
        restored = Profile.model_validate_json(json_str)
        assert restored.profile_id == profile.profile_id
        assert len(restored.facts) == len(profile.facts)
        assert restored.facts[0].fact_id == profile.facts[0].fact_id

    def test_roundtrip_dict(self):
        profile = _make_valid_profile()
        d = profile.model_dump()
        restored = Profile.model_validate(d)
        assert restored == profile


class TestRunConfig:
    def test_defaults(self):
        config = RunConfig()
        assert config.retraction_variant == "explicit"
        assert config.min_recall_threshold == 0.8
        assert len(config.probe_types) == 3
        assert ProbeType.TEXT_RETRACT_IMAGE_PROBE in config.probe_types
        assert ProbeType.INVISIBLE_PERSISTENCE in config.probe_types
        assert ProbeType.CROSS_IMAGE_RECONSTRUCTION in config.probe_types

    def test_extended_probes(self):
        config = RunConfig(probe_types=list(ProbeType))
        assert len(config.probe_types) == 5


class TestStrictMode:
    def test_strict_raises(self):
        profile = _make_valid_profile()
        profile.facts = profile.facts[:15]
        with pytest.raises(ValidationError) as exc_info:
            validate_profile(profile, strict=True)
        assert "test_001" in str(exc_info.value)
