#!/usr/bin/env python3
"""Run Oracle + LongContext on the 19 profiles missing from the original run,
then merge into existing results for a consistent n=103 across all 5 systems.

Usage: python scripts/run_missing_profiles.py
"""

from __future__ import annotations
import asyncio, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.schema.types import Profile, RunConfig, ProbeType, LeakageVerdict
from memleak.harness.oracle import OracleExplicit
from memleak.harness.long_context import LongContextSystem
from memleak.harness.runner import run_evaluation
from memleak.judge.leakage_judge import LeakageJudge
from memleak.metrics.cmlr import compute_cmlr
from memleak.schema.validators import validate_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

GEN_DIR = Path(__file__).parent.parent / "data" / "generated"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

MISSING_IDS = {"p016","p020","p027","p029","p037","p044","p045","p056","p058",
               "p074","p087","p091","p093","p098","p104","p117","p120","p121","p122"}


async def run_system_on_profiles(system_name, make_system, profiles, config, judge):
    results = []
    for i, profile in enumerate(profiles):
        logger.info(f"  [{system_name}] {i+1}/{len(profiles)}: {profile.profile_id}")
        system = make_system()
        if hasattr(system, "load_profile"):
            system.load_profile(profile)
        result = await run_evaluation(system, profile, config)
        fact_map = {f.fact_id: f for f in profile.facts}
        for pr in result.post_retraction_results:
            fact = fact_map.get(pr.fact_id)
            if fact:
                matching = next((p for p in fact.probes if p.probe_type == pr.probe_type), None)
                if matching:
                    jr = await judge.judge(fact.text, matching.query, pr.agent_response,
                                           f"{pr.fact_id}_{pr.probe_type.value}")
                    pr.judge_results = [jr]
                    pr.ensemble_verdict = jr.verdict
        results.append(result)
    return results


async def main():
    from memleak.llm import call_claude, call_claude_with_system

    # Load missing profiles
    profiles = []
    for f in sorted(GEN_DIR.glob("*.json")):
        p = Profile.model_validate_json(f.read_text())
        if p.profile_id not in MISSING_IDS:
            continue
        if len(p.facts) != 20:
            continue
        errors = validate_profile(p, strict=False)
        critical = [e for e in errors if "Expected 20" in e or "Expected 10 forget" in e
                     or "expected 2 forget targets" in e]
        if not critical:
            profiles.append(p)

    logger.info(f"Loaded {len(profiles)} missing profiles")

    config = RunConfig(filler_turns=2, temporal_k=[], min_recall_threshold=0.0)

    async def judge_fn(prompt): return await call_claude(prompt=prompt, max_tokens=256)
    judge = LeakageJudge(llm_fn=judge_fn, model_name="claude-sonnet-4")

    # Run Oracle
    logger.info("=== Oracle ===")
    oracle_new = await run_system_on_profiles(
        "Oracle", lambda: OracleExplicit(llm_fn=call_claude_with_system),
        profiles, config, judge)

    # Run LongContext
    logger.info("=== LongContext ===")
    def make_lc():
        lc = LongContextSystem(llm_fn=call_claude, model_name="claude")
        lc.reset()
        return lc
    lc_new = await run_system_on_profiles("LongContext", make_lc, profiles, config, judge)

    # Merge with existing
    for sys_name, new_results, filename in [
        ("Oracle", oracle_new, "Oracle_generated.json"),
        ("LongContext", lc_new, "LongContext_generated.json"),
    ]:
        existing = json.loads((RESULTS_DIR / filename).read_text())
        merged = existing + [r.model_dump() for r in new_results]
        (RESULTS_DIR / filename).write_text(json.dumps(merged, indent=2, default=str))
        probes = [p for r in merged for p in r["post_retraction_results"]]
        leaked = sum(1 for p in probes if p.get("ensemble_verdict") == "leaked")
        logger.info(f"{sys_name}: {len(merged)} profiles, {len(probes)} probes, {leaked/len(probes)*100:.1f}% CMLR")


if __name__ == "__main__":
    asyncio.run(main())
