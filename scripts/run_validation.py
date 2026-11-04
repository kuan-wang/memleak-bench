#!/usr/bin/env python3
"""$5 validation run: 5 pilots × 2 systems → CMLR table.

This is the minimum viable experiment to validate the structural floor claim.
Run this before spending more money.

Usage:
    python scripts/run_validation.py
    python scripts/run_validation.py --pilots 1 2 3  # specific pilots
    python scripts/run_validation.py --no-llm         # rule-based only (free)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.schema.types import (
    EvaluationResult,
    LeakageVerdict,
    ProbeResult,
    ProbeType,
    Profile,
    RunConfig,
)
from memleak.harness.oracle import OracleExplicit
from memleak.harness.long_context import LongContextSystem
from memleak.harness.runner import run_evaluation
from memleak.judge.leakage_judge import LeakageJudge
from memleak.metrics.cmlr import compute_cmlr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PILOT_DIR = Path(__file__).parent.parent / "data" / "pilots"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


def load_pilot(n: int) -> Profile:
    path = PILOT_DIR / f"pilot_{n:03d}.json"
    return Profile.model_validate_json(path.read_text())


async def run_system(
    system_name: str,
    system,
    profiles: list[Profile],
    config: RunConfig,
) -> list[EvaluationResult]:
    """Run one system on all profiles."""
    results = []
    for profile in profiles:
        logger.info(f"  Running {system_name} on {profile.profile_id}...")
        if hasattr(system, "load_profile"):
            system.load_profile(profile)

        out_dir = RESULTS_DIR / system_name / profile.profile_id
        result = await run_evaluation(system, profile, config, output_dir=out_dir)
        results.append(result)

        # Reset for next profile
        system.reset()
        if hasattr(system, "load_profile"):
            pass  # will be loaded next iteration

    return results


async def judge_results(
    results: list[EvaluationResult],
    judge: LeakageJudge,
    profiles: list[Profile],
) -> list[EvaluationResult]:
    """Score all probe results with the judge."""
    profile_map = {p.profile_id: p for p in profiles}

    for result in results:
        profile = profile_map.get(result.profile_id)
        if not profile:
            continue

        fact_map = {f.fact_id: f for f in profile.facts}

        for probe_result in result.post_retraction_results:
            fact = fact_map.get(probe_result.fact_id)
            if not fact:
                continue

            # Find the matching probe definition
            matching_probe = None
            for p in fact.probes:
                if p.probe_type == probe_result.probe_type:
                    matching_probe = p
                    break

            if matching_probe:
                judge_result = await judge.judge(
                    fact=fact.text,
                    query=matching_probe.query,
                    response=probe_result.agent_response,
                    probe_id=f"{probe_result.fact_id}_{probe_result.probe_type.value}",
                )
                probe_result.judge_results = [judge_result]
                probe_result.ensemble_verdict = judge_result.verdict

    return results


def print_results_table(all_results: dict[str, list[EvaluationResult]]):
    """Print the headline CMLR table."""
    print("\n" + "=" * 80)
    print("MEMLEAK VALIDATION RESULTS")
    print("=" * 80)

    # Collect all probe results across systems
    for system_name, results in all_results.items():
        all_probes = []
        for r in results:
            all_probes.extend(r.post_retraction_results)

        if not all_probes:
            print(f"\n{system_name}: No probes collected (all profiles may have been skipped)")
            continue

        # Overall CMLR
        overall = compute_cmlr(all_probes)

        # By probe type
        p1_probes = [p for p in all_probes if p.probe_type == ProbeType.TEXT_RETRACT_IMAGE_PROBE]
        p3_probes = [p for p in all_probes if p.probe_type == ProbeType.INVISIBLE_PERSISTENCE]
        p4_probes = [p for p in all_probes if p.probe_type == ProbeType.CROSS_IMAGE_RECONSTRUCTION]

        p1_cmlr = compute_cmlr(p1_probes)
        p3_cmlr = compute_cmlr(p3_probes)
        p4_cmlr = compute_cmlr(p4_probes)

        print(f"\n--- {system_name} ---")
        print(f"  Overall CMLR:     {overall.value:.1%} (n={overall.n})")
        print(f"  Probe 1 (linked): {p1_cmlr.value:.1%} (n={p1_cmlr.n})")
        print(f"  Probe 3 (unreach):{p3_cmlr.value:.1%} (n={p3_cmlr.n})")
        print(f"  Probe 4 (derived):{p4_cmlr.value:.1%} (n={p4_cmlr.n})")

        # The key delta
        if p1_cmlr.n > 0 and p3_cmlr.n > 0:
            delta = p3_cmlr.value - p1_cmlr.value
            print(f"  ** Structural floor delta (P3 - P1): {delta:+.1%} **")

    # Summary table
    print("\n" + "-" * 80)
    print(f"{'System':<25} {'Overall':>10} {'P1 (linked)':>12} {'P3 (unreach)':>13} {'Delta':>10}")
    print("-" * 80)
    for system_name, results in all_results.items():
        all_probes = [p for r in results for p in r.post_retraction_results]
        if not all_probes:
            print(f"{system_name:<25} {'N/A':>10}")
            continue
        overall = compute_cmlr(all_probes)
        p1 = compute_cmlr([p for p in all_probes if p.probe_type == ProbeType.TEXT_RETRACT_IMAGE_PROBE])
        p3 = compute_cmlr([p for p in all_probes if p.probe_type == ProbeType.INVISIBLE_PERSISTENCE])
        delta = (p3.value - p1.value) if p1.n > 0 and p3.n > 0 else float("nan")
        print(f"{system_name:<25} {overall.value:>9.1%} {p1.value:>11.1%} {p3.value:>12.1%} {delta:>+9.1%}")
    print("-" * 80)
    print()


async def main():
    parser = argparse.ArgumentParser(description="MemLeak $5 validation run")
    parser.add_argument("--pilots", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--no-llm", action="store_true", help="Use rule-based judge only (free)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for system calls")
    parser.add_argument("--backend", choices=["openai", "claude"], default="claude",
                        help="LLM backend: claude or openai")
    args = parser.parse_args()

    # Load pilots
    profiles = [load_pilot(n) for n in args.pilots]
    logger.info(f"Loaded {len(profiles)} pilot profiles")

    # Config: minimal for validation
    config = RunConfig(
        filler_turns=2,
        temporal_k=[],  # skip temporal for validation
        min_recall_threshold=0.0,  # bypass baseline (placeholder responses)
    )

    all_results: dict[str, list[EvaluationResult]] = {}

    # --- System 1: Oracle ---
    logger.info("=" * 40)
    logger.info("System 1: OracleExplicit")
    logger.info("=" * 40)

    if args.no_llm:
        oracle = OracleExplicit()  # no LLM, placeholder responses
    elif args.backend == "claude":
        from memleak.llm import call_claude_with_system
        oracle = OracleExplicit(llm_fn=call_claude_with_system)
    else:
        from memleak.llm import call_openai_with_system
        oracle = OracleExplicit(llm_fn=call_openai_with_system)

    oracle_results = []
    for profile in profiles:
        oracle.reset()
        oracle.load_profile(profile)
        logger.info(f"  Running oracle on {profile.profile_id}...")
        result = await run_evaluation(
            oracle, profile, config,
            output_dir=RESULTS_DIR / "oracle" / profile.profile_id,
        )
        oracle_results.append(result)
    all_results["OracleExplicit"] = oracle_results

    # --- System 2: Long-Context ---
    logger.info("=" * 40)
    logger.info("System 2: LongContext (GPT-4o-mini)")
    logger.info("=" * 40)

    if args.no_llm:
        lc = LongContextSystem()
    elif args.backend == "claude":
        from memleak.llm import call_claude
        lc = LongContextSystem(llm_fn=call_claude, model_name="claude-sonnet-4")
    else:
        from memleak.llm import call_openai
        lc = LongContextSystem(llm_fn=call_openai, model_name=args.model)

    lc_results = []
    for profile in profiles:
        lc.reset()
        logger.info(f"  Running long-context on {profile.profile_id}...")
        result = await run_evaluation(
            lc, profile, config,
            output_dir=RESULTS_DIR / "long_context" / profile.profile_id,
        )
        lc_results.append(result)
    all_results["LongContext"] = lc_results

    # --- Judge all results ---
    logger.info("=" * 40)
    logger.info("Judging results...")
    logger.info("=" * 40)

    if args.no_llm:
        judge = LeakageJudge()  # rule-based
    elif args.backend == "claude":
        from memleak.llm import call_claude
        async def judge_llm(prompt: str) -> str:
            return await call_claude(prompt=prompt, max_tokens=256)
        judge = LeakageJudge(llm_fn=judge_llm, model_name="claude-sonnet-4")
    else:
        from memleak.llm import call_openai
        async def judge_llm(prompt: str) -> str:
            return await call_openai(prompt=prompt, model="gpt-4o-mini", max_tokens=256)
        judge = LeakageJudge(llm_fn=judge_llm, model_name="gpt-4o-mini")

    for system_name in all_results:
        all_results[system_name] = await judge_results(
            all_results[system_name], judge, profiles
        )

    # --- Print results ---
    print_results_table(all_results)

    # --- Save raw results ---
    for system_name, results in all_results.items():
        out_path = RESULTS_DIR / f"{system_name}_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            [r.model_dump() for r in results],
            indent=2, default=str,
        ))
        logger.info(f"Saved {system_name} results to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
