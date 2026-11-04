"""Validation rules for MemLeak profiles.

These validators enforce the structural invariants that the benchmark
design requires. Every generated profile must pass all validators
before being included in the dataset.
"""

from __future__ import annotations

from collections import Counter

from memleak.schema.types import FactCategory, Profile, VisualGrounding


class ValidationError(Exception):
    def __init__(self, profile_id: str, errors: list[str]):
        self.profile_id = profile_id
        self.errors = errors
        super().__init__(f"Profile {profile_id}: {len(errors)} error(s)\n" + "\n".join(errors))


def validate_profile(profile: Profile, strict: bool = True) -> list[str]:
    """Validate a profile against all structural invariants.

    Returns a list of error messages. Empty list = valid.
    If strict=True, raises ValidationError on any failure.
    """
    errors: list[str] = []

    # --- Fact count ---
    if len(profile.facts) != 20:
        errors.append(f"Expected 20 facts, got {len(profile.facts)}")

    # --- Forget target count ---
    forget_count = sum(1 for f in profile.facts if f.forget_target)
    if forget_count != 10:
        errors.append(f"Expected 10 forget targets, got {forget_count}")

    # --- Category balance: exactly 4 facts per category ---
    cat_counts = Counter(f.category for f in profile.facts)
    for cat in FactCategory:
        if cat_counts.get(cat, 0) != 4:
            errors.append(f"Category {cat.value}: expected 4 facts, got {cat_counts.get(cat, 0)}")

    # --- 2 forget targets per category ---
    forget_by_cat = Counter(f.category for f in profile.facts if f.forget_target)
    for cat in FactCategory:
        if forget_by_cat.get(cat, 0) != 2:
            errors.append(
                f"Category {cat.value}: expected 2 forget targets, "
                f"got {forget_by_cat.get(cat, 0)}"
            )

    # --- Grounding distribution among forget targets ---
    forget_grounding = Counter(
        f.visual_grounding for f in profile.facts if f.forget_target
    )
    explicit = forget_grounding.get(VisualGrounding.EXPLICIT, 0)
    implicit = forget_grounding.get(VisualGrounding.IMPLICIT, 0)
    absent = forget_grounding.get(VisualGrounding.ABSENT, 0)
    # Allow some flexibility: ~4 explicit, ~4 implicit, ~2 absent
    if explicit < 2 or explicit > 6:
        errors.append(f"Forget targets: expected 2-6 explicit grounding, got {explicit}")
    if implicit < 2 or implicit > 6:
        errors.append(f"Forget targets: expected 2-6 implicit grounding, got {implicit}")
    if absent < 1 or absent > 4:
        errors.append(f"Forget targets: expected 1-4 absent grounding, got {absent}")

    # --- Every forget target has entangled facts ---
    for fact in profile.facts:
        if fact.forget_target and len(fact.entangled_facts) == 0:
            errors.append(f"Fact {fact.fact_id}: forget target has no entangled facts")

    # --- Every forget target has at least one probe ---
    for fact in profile.facts:
        if fact.forget_target and len(fact.probes) == 0:
            errors.append(f"Fact {fact.fact_id}: forget target has no probes")

    # --- Multi-hop chain answers differ ---
    for fact in profile.facts:
        for chain in fact.cross_modal_chains:
            if chain.answer_if_remembered == chain.answer_if_forgotten:
                errors.append(
                    f"Fact {fact.fact_id}, chain {chain.chain_id}: "
                    f"remembered and forgotten answers are identical"
                )

    # --- Unique fact IDs within profile ---
    fact_ids = [f.fact_id for f in profile.facts]
    if len(fact_ids) != len(set(fact_ids)):
        dupes = [fid for fid, cnt in Counter(fact_ids).items() if cnt > 1]
        errors.append(f"Duplicate fact IDs: {dupes}")

    # --- Images have paths for non-absent grounding ---
    for fact in profile.facts:
        if fact.visual_grounding != VisualGrounding.ABSENT and len(fact.images) == 0:
            # Only warn — images may be placeholder during development
            if strict:
                errors.append(
                    f"Fact {fact.fact_id}: {fact.visual_grounding.value} grounding but no images"
                )

    # --- Retraction point references valid fact IDs ---
    if profile.retraction_point:
        valid_ids = set(fact_ids)
        for fid in profile.retraction_point.target_facts:
            if fid not in valid_ids:
                errors.append(f"Retraction targets unknown fact: {fid}")
            else:
                matching = next(f for f in profile.facts if f.fact_id == fid)
                if not matching.forget_target:
                    errors.append(f"Retraction targets non-forget fact: {fid}")

    # --- Interaction stream ordering ---
    if profile.interaction_stream and profile.retraction_point:
        retraction_turn = profile.retraction_point.turn_id
        introduced_before = set()
        for turn in profile.interaction_stream:
            if turn.turn_id < retraction_turn:
                introduced_before.update(turn.facts_introduced)
        forget_ids = {f.fact_id for f in profile.facts if f.forget_target}
        missing = forget_ids - introduced_before
        if missing and profile.interaction_stream:  # only check if stream is populated
            errors.append(f"Forget facts not introduced before retraction: {missing}")

    if strict and errors:
        raise ValidationError(profile.profile_id, errors)

    return errors
