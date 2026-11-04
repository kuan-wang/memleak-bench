#!/usr/bin/env python3
"""Run Mem0 only on the 5 new profiles, then merge."""

from __future__ import annotations
import asyncio, json, logging, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.schema.types import Profile, RunConfig
from memleak.schema.validators import validate_profile
from memleak.harness.mem0_wrapper import Mem0System
from memleak.harness.runner import run_evaluation
from memleak.judge.leakage_judge import LeakageJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

GEN_DIR = Path(__file__).parent.parent / "data" / "generated"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

NEW_IDS = {"p011", "p025", "p036", "p039", "p050"}


async def main():
    from memleak.llm import call_claude

    profiles = []
    for f in sorted(GEN_DIR.glob("*.json")):
        p = Profile.model_validate_json(f.read_text())
        if p.profile_id not in NEW_IDS:
            continue
        if len(p.facts) != 20:
            continue
        errors = validate_profile(p, strict=False)
        critical = [e for e in errors if "Expected 20" in e or "Expected 10 forget" in e
                     or "expected 2 forget targets" in e]
        if not critical:
            profiles.append(p)

    logger.info(f"Loaded {len(profiles)} profiles for Mem0")

    config = RunConfig(filler_turns=2, temporal_k=[], min_recall_threshold=0.0)

    async def judge_fn(prompt): return await call_claude(prompt=prompt, max_tokens=256)
    judge = LeakageJudge(llm_fn=judge_fn, model_name="claude-sonnet-4")

    _mem0_counter = [0]
    def make_mem0():
        _mem0_counter[0] += 1
        return Mem0System(llm_fn=call_claude, user_id=f"gen_{os.getpid()}_{_mem0_counter[0]}")

    results = []
    for i, profile in enumerate(profiles):
        logger.info(f"  [Mem0] {i+1}/{len(profiles)}: {profile.profile_id}")
        system = make_mem0()
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

    # Merge
    filename = "Mem0_generated.json"
    existing = json.loads((RESULTS_DIR / filename).read_text())
    merged = existing + [r.model_dump() for r in results]
    (RESULTS_DIR / filename).write_text(json.dumps(merged, indent=2, default=str))

    probes = [p for r in merged for p in r["post_retraction_results"]]
    leaked = sum(1 for p in probes if p.get("ensemble_verdict") == "leaked")
    logger.info(f"Mem0: {len(merged)} profiles, {len(probes)} probes, "
                 f"{leaked/len(probes)*100:.1f}% CMLR")


if __name__ == "__main__":
    asyncio.run(main())
