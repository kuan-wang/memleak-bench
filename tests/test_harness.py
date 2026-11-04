"""Tests for the evaluation harness: oracle + runner."""

import asyncio
import json
from pathlib import Path

import pytest

from memleak.harness.oracle import OracleExplicit
from memleak.harness.long_context import LongContextSystem
from memleak.harness.runner import run_evaluation
from memleak.schema.types import Profile, RunConfig


PILOT_DIR = Path(__file__).parent.parent / "data" / "pilots"


def _load_pilot(n: int) -> Profile:
    path = PILOT_DIR / f"pilot_{n:03d}.json"
    return Profile.model_validate_json(path.read_text())


async def _echo_reference_context(system_prompt, query, images):
    """Deterministic test double: report only the supplied memory context."""
    return system_prompt


class TestOracleExplicit:
    def test_reset(self):
        oracle = OracleExplicit()
        profile = _load_pilot(1)
        oracle.load_profile(profile)
        assert len(oracle.store.text_entries) == 20
        oracle.reset()
        assert len(oracle.store.text_entries) == 0

    def test_load_profile(self):
        oracle = OracleExplicit()
        profile = _load_pilot(1)
        oracle.load_profile(profile)
        assert len(oracle.store.facts) == 20
        assert len(oracle.store.text_entries) == 20

    @pytest.mark.asyncio
    async def test_forget_deletes_target_facts(self):
        oracle = OracleExplicit()
        profile = _load_pilot(1)
        oracle.load_profile(profile)

        forget_result = await oracle.forget(profile.retraction_point.instruction)
        assert forget_result.success
        assert len(forget_result.deleted_items) > 0
        assert forget_result.method == "oracle_explicit"

        # Verify target facts are deleted
        for fact in profile.facts:
            if fact.forget_target:
                assert fact.fact_id in oracle.store.deleted_facts
                assert fact.fact_id not in oracle.store.text_entries
            else:
                assert fact.fact_id not in oracle.store.deleted_facts
                assert fact.fact_id in oracle.store.text_entries

    @pytest.mark.asyncio
    async def test_probe_without_llm(self):
        oracle = OracleExplicit()  # no llm_fn
        profile = _load_pilot(1)
        oracle.load_profile(profile)

        response = await oracle.probe("What do you know about me?")
        assert "Oracle response" in response
        assert "20 active facts" in response

    def test_memory_state(self):
        oracle = OracleExplicit()
        profile = _load_pilot(1)
        oracle.load_profile(profile)

        state = oracle.get_memory_state()
        assert len(state["active_facts"]) == 20
        assert len(state["deleted_facts"]) == 0


class TestLongContextSystem:
    @pytest.mark.asyncio
    async def test_basic_flow(self):
        system = LongContextSystem()  # no llm_fn
        system.reset()

        from memleak.schema.types import Turn
        turn = Turn(turn_id=0, role="user", text="Hello, I live in Paris.")
        response = await system.inject_turn(turn)
        assert "placeholder" in response.lower()

        state = system.get_memory_state()
        assert state["user_messages"] == 1
        assert state["assistant_messages"] == 1

    @pytest.mark.asyncio
    async def test_forget_is_suppression(self):
        system = LongContextSystem()
        system.reset()

        result = await system.forget("Forget where I live.")
        assert result.success
        assert result.method == "behavioral_suppression"
        assert len(result.deleted_items) == 0  # nothing actually deleted

        state = system.get_memory_state()
        assert state["system_messages"] == 2  # initial + suppression


class TestRunner:
    @pytest.mark.asyncio
    async def test_full_loop_oracle(self, tmp_path):
        """Run the full 6-phase evaluation on pilot 1 with oracle."""
        oracle = OracleExplicit(llm_fn=_echo_reference_context)
        profile = _load_pilot(1)
        config = RunConfig(
            filler_turns=2,
            temporal_k=[5],  # short for testing
            min_recall_threshold=0.0,
        )

        result = await run_evaluation(
            system=oracle,
            profile=profile,
            config=config,
            output_dir=tmp_path / "oracle" / "pilot_001",
        )

        assert result.system_name == "OracleExplicit"
        assert result.profile_id == "pilot_001"
        assert not result.skipped
        assert len(result.post_retraction_results) > 0
        assert 5 in result.temporal_results

        # Check trace file was written
        trace_path = tmp_path / "oracle" / "pilot_001" / "trace.jsonl"
        assert trace_path.exists()
        lines = trace_path.read_text().strip().split("\n")
        assert len(lines) > 10  # should have many trace entries

    @pytest.mark.asyncio
    async def test_full_loop_long_context(self, tmp_path):
        """A placeholder backend must not produce forgetting measurements."""
        system = LongContextSystem()
        profile = _load_pilot(1)

        config = RunConfig(
            filler_turns=2,
            temporal_k=[5],
            min_recall_threshold=0.0,
        )

        result = await run_evaluation(
            system=system,
            profile=profile,
            config=config,
            output_dir=tmp_path / "long_context" / "pilot_001",
        )

        assert result.skipped
        assert result.post_retraction_results == []

    @pytest.mark.asyncio
    async def test_skips_on_low_recall(self, tmp_path):
        """Profile should be skipped if baseline recall is too low."""
        oracle = OracleExplicit()  # no LLM = responses won't contain indicators
        profile = _load_pilot(1)
        oracle.load_profile(profile)

        config = RunConfig(
            min_recall_threshold=0.9,  # high threshold
            filler_turns=1,
            temporal_k=[],
        )

        result = await run_evaluation(
            system=oracle,
            profile=profile,
            config=config,
        )

        # Oracle without LLM returns placeholder, so no indicators match
        # → baseline recall ≈ 0% → should skip
        assert result.skipped
        assert "baseline_recall" in result.skip_reason

    @pytest.mark.asyncio
    async def test_all_pilots_run(self, tmp_path):
        """Smoke test: all 5 pilots complete without error."""
        for i in range(1, 6):
            oracle = OracleExplicit(llm_fn=_echo_reference_context)
            profile = _load_pilot(i)
            oracle.load_profile(profile)

            config = RunConfig(
                filler_turns=1,
                temporal_k=[],
                min_recall_threshold=0.0,
            )

            result = await run_evaluation(
                system=oracle,
                profile=profile,
                config=config,
            )

            assert not result.skipped, f"Pilot {i} was skipped"
            assert len(result.post_retraction_results) > 0, f"Pilot {i} had no probes"
