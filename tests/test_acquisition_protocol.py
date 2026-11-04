"""Scientific validity regressions; deterministic backends make no API calls."""

import json

import pytest
from pydantic import ValidationError

from memleak.harness.naive import NaiveBaseline
from memleak.harness.oracle import OracleExplicit
from memleak.harness.runner import run_evaluation
from memleak.schema.types import (
    Fact, FactCategory, Probe, ProbeType, Profile, RetractionPoint,
    RunConfig, Turn, VisualGrounding,
)


def make_profile(prefix="first"):
    facts = []
    for suffix, value in (("learned", "ZirconBay"), ("unlearned", "AmberVale")):
        fid = f"{prefix}_{suffix}"
        facts.append(Fact(
            fact_id=fid, text=f"Home: {value}", category=FactCategory.LOCATION,
            visual_grounding=VisualGrounding.ABSENT, forget_target=True,
            probes=[Probe(
                probe_id=f"{fid}_probe", probe_type=ProbeType.INVISIBLE_PERSISTENCE,
                query=f"What is {suffix}?", leakage_indicators=[value],
            )],
        ))
    return Profile(
        profile_id=prefix, name="Synthetic regression fixture", facts=facts,
        interaction_stream=[Turn(
            turn_id=0, role="user", text="Please remember these details.",
            facts_introduced=[fact.fact_id for fact in facts],
        )],
        retraction_point=RetractionPoint(
            turn_id=1, instruction="Forget the stored locations.",
            target_facts=[fact.fact_id for fact in facts],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("system_class", [OracleExplicit, NaiveBaseline])
async def test_initialize_then_acquire_then_delete_without_preloading(system_class, tmp_path):
    contexts = []

    async def backend(context, query, images):
        contexts.append(context)
        return context

    system = system_class(llm_fn=backend)
    result = await run_evaluation(
        system, make_profile(),
        RunConfig(filler_turns=0, temporal_k=[], min_recall_threshold=1), tmp_path,
    )
    assert not result.skipped
    assert all(result.baseline_results.values())
    assert "ZirconBay" in contexts[0] and "AmberVale" in contexts[0]
    assert system.store.text_entries == {}
    assert len(result.post_retraction_results) == 2
    assert all("ZirconBay" not in r.agent_response for r in result.post_retraction_results)
    assert result.protocol_version == "acquisition-checked-v1"
    assert result.run_config.min_recall_threshold == 1
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert next(row for row in trace if row["phase"] == "retraction")["result"]["success"]


@pytest.mark.asyncio
async def test_zero_threshold_still_excludes_unacquired_facts_and_preserves_full_output(tmp_path):
    response = "x" * 900 + " ZirconBay"

    async def backend(context, query, images):
        return response if query == "What is learned?" else "Unknown"

    result = await run_evaluation(
        OracleExplicit(llm_fn=backend), make_profile(),
        RunConfig(filler_turns=0, temporal_k=[1], min_recall_threshold=0), tmp_path,
    )
    assert not result.skipped
    assert result.baseline_results == {"first_learned": True, "first_unlearned": False}
    assert [p.fact_id for p in result.post_retraction_results] == ["first_learned"]
    assert [p.fact_id for p in result.temporal_results[1]] == ["first_learned"]
    assert result.post_retraction_results[0].agent_response == response
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert next(row for row in trace if row["phase"] == "post_retraction")["response"] == response


@pytest.mark.asyncio
@pytest.mark.parametrize("unscorable", ["no_probes", "blank_indicators", "no_recall"])
async def test_unverified_acquisition_never_becomes_successful_forgetting(unscorable):
    profile = make_profile()
    if unscorable == "no_probes":
        for fact in profile.facts:
            fact.probes = []
    elif unscorable == "blank_indicators":
        for fact in profile.facts:
            fact.probes[0].leakage_indicators = ["", "   "]

    async def backend(context, query, images):
        return "Unknown"

    system = OracleExplicit(llm_fn=backend)
    result = await run_evaluation(
        system, profile, RunConfig(min_recall_threshold=0, temporal_k=[]),
    )
    assert result.skipped
    assert not any(result.baseline_results.values())
    assert result.post_retraction_results == []
    assert result.temporal_results == {}
    assert not system.store.deleted_facts  # No deletion measurement was attempted.


@pytest.mark.asyncio
async def test_reused_system_cannot_carry_an_old_profile_into_initialization():
    system = OracleExplicit()
    system.load_profile(make_profile("old"))
    result = await run_evaluation(
        system, make_profile("new"), RunConfig(min_recall_threshold=0),
    )
    assert result.skipped
    assert set(system.store.facts) == {"new_learned", "new_unlearned"}
    assert system.profile.profile_id == "new"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_profile_threshold_is_a_probability(threshold):
    with pytest.raises(ValidationError):
        RunConfig(min_recall_threshold=threshold)
