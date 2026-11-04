#!/usr/bin/env python3
"""Run all 5 systems on pilot profiles. The full comparison.

Systems: Long-context, Naive, Mem0, Letta, Oracle
Backend: Claude Sonnet (default) or OpenAI (--backend openai)

Usage:
    python scripts/run_all_systems.py --pilots 1
    python scripts/run_all_systems.py --pilots 1 2 3 4 5
    python scripts/run_all_systems.py --pilots 1 --backend openai
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.schema.types import Profile, RunConfig, ProbeType, LeakageVerdict
from memleak.harness.oracle import OracleExplicit
from memleak.harness.naive import NaiveBaseline
from memleak.harness.long_context import LongContextSystem
from memleak.harness.letta_wrapper import LettaSimulated
from memleak.harness.runner import run_evaluation
from memleak.judge.leakage_judge import LeakageJudge
from memleak.metrics.cmlr import compute_cmlr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PILOT_DIR = Path(__file__).parent.parent / "data" / "pilots"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


def load_pilot(n: int) -> Profile:
    return Profile.model_validate_json((PILOT_DIR / f"pilot_{n:03d}.json").read_text())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilots", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--backend", choices=["claude", "openai"], default="claude")
    args = parser.parse_args()

    profiles = [load_pilot(n) for n in args.pilots]
    logger.info(f"Loaded {len(profiles)} profiles, backend={args.backend}")

    config = RunConfig(
        filler_turns=2,
        temporal_k=[],
        min_recall_threshold=0.0,
    )

    # Set up LLM functions based on backend
    if args.backend == "claude":
        from memleak.llm import call_claude, call_claude_with_system
        llm_fn = call_claude
        llm_sys_fn = call_claude_with_system
        judge_name = "claude-sonnet-4"
    else:
        from memleak.llm import call_openai, call_openai_with_system
        llm_fn = call_openai
        llm_sys_fn = call_openai_with_system
        judge_name = "gpt-4o-mini"

    # Judge
    async def judge_fn(prompt: str) -> str:
        return await llm_fn(prompt=prompt, max_tokens=256)
    judge = LeakageJudge(llm_fn=judge_fn, model_name=judge_name)

    # Define all 5 systems
    systems = {
        "LongContext": lambda: LongContextSystem(llm_fn=llm_fn, model_name=args.backend),
        "Naive": lambda: NaiveBaseline(llm_fn=llm_sys_fn),
        "Letta": lambda: LettaSimulated(llm_fn=llm_fn),
        "Oracle": lambda: OracleExplicit(llm_fn=llm_sys_fn),
    }

    all_results = {}

    for sys_name, make_system in systems.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"System: {sys_name}")
        logger.info(f"{'='*50}")

        sys_results = []
        for profile in profiles:
            logger.info(f"  {profile.profile_id}...")
            system = make_system()

            if hasattr(system, "load_profile"):
                system.load_profile(profile)
            if hasattr(system, "reset"):
                system.reset()
                if hasattr(system, "load_profile"):
                    system.load_profile(profile)

            result = await run_evaluation(system, profile, config)

            # Judge
            fact_map = {f.fact_id: f for f in profile.facts}
            for pr in result.post_retraction_results:
                fact = fact_map.get(pr.fact_id)
                if fact:
                    matching = next((p for p in fact.probes if p.probe_type == pr.probe_type), None)
                    if matching:
                        jr = await judge.judge(fact.text, matching.query, pr.agent_response, pr.fact_id)
                        pr.judge_results = [jr]
                        pr.ensemble_verdict = jr.verdict

            sys_results.append(result)

        all_results[sys_name] = sys_results

    # Print results
    print("\n" + "=" * 80)
    print(f"MEMLEAK — ALL SYSTEMS COMPARISON ({args.backend} backend)")
    print("=" * 80)

    header = f"{'System':<15} {'Overall':>8} {'P1':>8} {'P3':>8} {'P4':>8} {'n':>6}"
    print(header)
    print("-" * len(header))

    for sys_name, results in all_results.items():
        probes = [p for r in results for p in r.post_retraction_results]
        if not probes:
            print(f"{sys_name:<15} {'N/A':>8}")
            continue

        overall = compute_cmlr(probes)
        p1 = compute_cmlr([p for p in probes if p.probe_type == ProbeType.TEXT_RETRACT_IMAGE_PROBE])
        p3 = compute_cmlr([p for p in probes if p.probe_type == ProbeType.INVISIBLE_PERSISTENCE])
        p4 = compute_cmlr([p for p in probes if p.probe_type == ProbeType.CROSS_IMAGE_RECONSTRUCTION])

        print(f"{sys_name:<15} {overall.value:>7.1%} {p1.value:>7.1%} {p3.value:>7.1%} {p4.value:>7.1%} {overall.n:>6}")

    print("-" * len(header))

    # Save
    for sys_name, results in all_results.items():
        out = RESULTS_DIR / f"{sys_name}_allsystems.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([r.model_dump() for r in results], indent=2, default=str))

    logger.info(f"\nResults saved to {RESULTS_DIR}/*_allsystems.json")


if __name__ == "__main__":
    asyncio.run(main())
