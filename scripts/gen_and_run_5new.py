#!/usr/bin/env python3
"""Generate 5 new profiles and run all 5 systems on them.

Usage: python scripts/gen_and_run_5new.py
"""

from __future__ import annotations
import asyncio, json, logging, os, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.datagen.profiles import generate_profile, PROFILE_THEMES
from memleak.schema.types import Profile, RunConfig
from memleak.schema.validators import validate_profile
from memleak.harness.oracle import OracleExplicit
from memleak.harness.long_context import LongContextSystem
from memleak.harness.naive import NaiveBaseline
from memleak.harness.letta_wrapper import LettaSimulated
from memleak.harness.mem0_wrapper import Mem0System
from memleak.harness.runner import run_evaluation
from memleak.judge.leakage_judge import LeakageJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

GEN_DIR = Path(__file__).parent.parent / "data" / "generated"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

NEW_IDS = ["p011", "p025", "p036", "p039", "p050"]
MAX_RETRIES = 8


async def generate_new_profiles() -> list[Profile]:
    """Generate 5 new valid profiles with unique themes."""
    # Pick 5 unused themes (shuffle with different seed to avoid collisions)
    themes = PROFILE_THEMES.copy()
    random.seed(99)
    random.shuffle(themes)

    profiles = []
    for i, prefix in enumerate(NEW_IDS):
        theme = themes[i]
        logger.info(f"Generating {prefix}: {theme[:60]}...")

        for attempt in range(1, MAX_RETRIES + 1):
            profile = await generate_profile(theme, prefix, model="gpt-4o-mini")
            if profile is None:
                logger.warning(f"  {prefix} attempt {attempt}: generation failed")
                continue

            errors = validate_profile(profile, strict=False)
            critical = [e for e in errors if "Expected 20" in e or "Expected 10 forget" in e
                         or "expected 2 forget targets" in e]

            if len(profile.facts) == 20 and not critical:
                path = GEN_DIR / f"{prefix}.json"
                path.write_text(profile.model_dump_json(indent=2))
                logger.info(f"  {prefix} attempt {attempt}: PASS")
                profiles.append(profile)
                break
            else:
                logger.warning(f"  {prefix} attempt {attempt}: FAIL "
                               f"(facts={len(profile.facts)}, critical={len(critical)})")
        else:
            logger.error(f"  {prefix}: FAILED after {MAX_RETRIES} attempts")

    return profiles


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
            if not fact:
                continue
            matching = next((p for p in fact.probes if p.probe_type == pr.probe_type), None)
            if matching:
                jr = await judge.judge(
                    fact=fact.text, query=matching.query,
                    response=pr.agent_response,
                    probe_id=f"{pr.fact_id}_{pr.probe_type.value}",
                )
                pr.judge_results = [jr]
                pr.ensemble_verdict = jr.verdict
        results.append(result)
    return results


async def main():
    from memleak.llm import call_claude, call_claude_with_system

    # Step 1: Generate profiles
    logger.info("=== Generating 5 new profiles ===")
    profiles = await generate_new_profiles()
    logger.info(f"Generated {len(profiles)} valid profiles")

    if not profiles:
        logger.error("No valid profiles generated!")
        return

    # Step 2: Run all 5 systems
    config = RunConfig(filler_turns=2, temporal_k=[], min_recall_threshold=0.0)

    async def judge_fn(prompt): return await call_claude(prompt=prompt, max_tokens=256)
    judge = LeakageJudge(llm_fn=judge_fn, model_name="claude-sonnet-4")

    llm_fn = call_claude
    llm_sys_fn = call_claude_with_system

    _mem0_counter = [0]
    def make_mem0():
        _mem0_counter[0] += 1
        return Mem0System(llm_fn=llm_fn, user_id=f"gen_{os.getpid()}_{_mem0_counter[0]}")

    systems = [
        ("Oracle", lambda: OracleExplicit(llm_fn=llm_sys_fn)),
        ("LongContext", lambda: LongContextSystem(llm_fn=llm_fn, model_name="claude")),
        ("Naive", lambda: NaiveBaseline(llm_fn=llm_sys_fn)),
        ("Letta", lambda: LettaSimulated(llm_fn=llm_fn)),
        ("Mem0", make_mem0),
    ]

    for sys_name, make_system in systems:
        logger.info(f"=== {sys_name} ===")
        new_results = await run_system_on_profiles(
            sys_name, make_system, profiles, config, judge)

        # Merge
        filename = f"{sys_name}_generated.json"
        existing = json.loads((RESULTS_DIR / filename).read_text())
        merged = existing + [r.model_dump() for r in new_results]
        (RESULTS_DIR / filename).write_text(json.dumps(merged, indent=2, default=str))

        probes = [p for r in merged for p in r["post_retraction_results"]]
        leaked = sum(1 for p in probes if p.get("ensemble_verdict") == "leaked")
        logger.info(f"{sys_name}: {len(merged)} profiles, {len(probes)} probes, "
                     f"{leaked/len(probes)*100:.1f}% CMLR")


if __name__ == "__main__":
    asyncio.run(main())
