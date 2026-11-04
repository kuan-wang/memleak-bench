"""Check that the revised entry point produces auditable, separate artifacts."""

import json

import pytest

from memleak.harness.oracle import OracleExplicit
from memleak.judge.leakage_judge import LeakageJudge
from memleak.schema.types import RunConfig
from scripts.run_generated import build_manifest, run_system_on_profiles
from tests.test_acquisition_protocol import make_profile


@pytest.mark.asyncio
async def test_generated_entry_point_writes_complete_trace_and_judged_result(tmp_path):
    async def backend(context, query, images):
        return context

    profile = make_profile()
    results = await run_system_on_profiles(
        "Oracle", lambda: OracleExplicit(llm_fn=backend), [profile],
        RunConfig(filler_turns=0, temporal_k=[]), LeakageJudge(), output_dir=tmp_path,
    )
    saved = json.loads((tmp_path / "Oracle" / "first" / "result.json").read_text())
    assert saved["baseline_results"] == {"first_learned": True, "first_unlearned": True}
    assert len(saved["post_retraction_results"]) == 2
    assert all(row["judge_results"] for row in saved["post_retraction_results"])
    assert (tmp_path / "Oracle" / "first" / "trace.jsonl").exists()
    assert results[0].protocol_version == "acquisition-checked-v1"


def test_manifest_fingerprints_inputs_without_collecting_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-must-not-appear-in-artifacts")
    profile = make_profile()
    first = build_manifest([profile], RunConfig(), "claude")
    profile.facts[0].text = "Changed input"
    second = build_manifest([profile], RunConfig(), "claude")
    assert first["profile_sha256"] != second["profile_sha256"]
    assert first["source_sha256"] == second["source_sha256"]
    assert len(first["source_sha256"]["memleak/harness/runner.py"]) == 64
    assert first["transport"] == "anthropic_api"
    assert first["requested_judge_model"] == "claude-sonnet-4-6"
    assert "secret-must-not-appear-in-artifacts" not in json.dumps(first)
