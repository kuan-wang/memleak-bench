#!/usr/bin/env python3
"""Run experiments on generated profiles (text-only, no images).

Usage:
    python scripts/run_generated.py                    # all usable profiles, oracle+long_context
    python scripts/run_generated.py --max 20           # first 20 only
    python scripts/run_generated.py --system oracle    # oracle only
    python scripts/run_generated.py --system naive     # naive baseline
    python scripts/run_generated.py --system letta     # letta simulated
    python scripts/run_generated.py --system mem0      # mem0 wrapper
    python scripts/run_generated.py --system all       # all 5 systems
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
import subprocess
from datetime import datetime, timezone
from uuid import uuid4
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.schema.types import (
    EvaluationResult,
    FactCategory,
    LeakageVerdict,
    ProbeResult,
    ProbeType,
    Profile,
    RunConfig,
)
from memleak.harness.oracle import OracleExplicit
from memleak.harness.long_context import LongContextSystem
from memleak.harness.naive import NaiveBaseline
from memleak.harness.letta_wrapper import LettaSimulated
from memleak.harness.mem0_wrapper import Mem0System
from memleak.harness.runner import run_evaluation
from memleak.judge.leakage_judge import LeakageJudge
from memleak.metrics.cmlr import compute_cmlr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

GEN_DIR = Path(__file__).parent.parent / "data" / "generated"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "revision_runs"


def load_usable_profiles(max_count: int | None = None) -> list[Profile]:
    """Load generated profiles that pass basic validation."""
    from memleak.schema.validators import validate_profile

    profiles = []
    for f in sorted(GEN_DIR.glob("*.json")):
        p = Profile.model_validate_json(f.read_text())
        if len(p.facts) != 20:
            continue
        errors = validate_profile(p, strict=False)
        critical = [e for e in errors if "Expected 20" in e or "Expected 10 forget" in e
                     or "expected 2 forget targets" in e]
        if not critical:
            profiles.append(p)
            if max_count and len(profiles) >= max_count:
                break

    return profiles


async def run_system_on_profiles(
    system_name: str,
    make_system,
    profiles: list[Profile],
    config: RunConfig,
    judge: LeakageJudge,
    output_dir: Path | None = None,
) -> list[EvaluationResult]:
    """Run one system on all profiles, judge results."""
    results = []

    for i, profile in enumerate(profiles):
        logger.info(f"  [{system_name}] {i+1}/{len(profiles)}: {profile.profile_id}")
        system = make_system()

        trace_dir = output_dir / system_name / profile.profile_id if output_dir else None
        result = await run_evaluation(system, profile, config, output_dir=trace_dir)
        results.append(result)

        # Judge probe results
        fact_map = {f.fact_id: f for f in profile.facts}
        for pr in result.post_retraction_results:
            fact = fact_map.get(pr.fact_id)
            if not fact:
                continue
            matching_probe = next(
                (p for p in fact.probes if p.probe_type == pr.probe_type), None
            )
            if matching_probe:
                jr = await judge.judge(
                    fact=fact.text,
                    query=matching_probe.query,
                    response=pr.agent_response,
                    probe_id=f"{pr.fact_id}_{pr.probe_type.value}",
                )
                pr.judge_results = [jr]
                pr.ensemble_verdict = jr.verdict

        if trace_dir:
            (trace_dir / "result.json").write_text(result.model_dump_json(indent=2))

    return results


def build_manifest(profiles: list[Profile], config: RunConfig, backend: str) -> dict:
    """Record inputs and source without collecting credentials or environment variables."""
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    paths = (
        sorted((root / "memleak").rglob("*.py"))
        + [Path(__file__).resolve(), root / "pyproject.toml"]
    )
    packages = {}
    for name in ("memleak", "pydantic", "openai", "anthropic", "mem0ai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "protocol_version": "acquisition-checked-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "git_commit": commit,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "profile_sha256": {
            profile.profile_id: hashlib.sha256(
                profile.model_dump_json().encode()
            ).hexdigest()
            for profile in profiles
        },
        "python": sys.version,
        "packages": packages,
        "backend_route": backend,
        "transport": "anthropic_api" if backend == "claude" else "openai_api",
        "requested_judge_model": "claude-sonnet-4-6" if backend == "claude" else "gpt-4o-mini",
        "model_resolution": "See recorded source; provider-resolved model IDs are not captured.",
        "judge_protocol": "single_judge_complete_response",
        "baseline_protocol": "nonempty_indicator_substring_match",
        "config": config.model_dump(mode="json"),
    }


def print_results(all_results: dict[str, list[EvaluationResult]]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("MEMLEAK RESULTS — GENERATED PROFILES")
    print("=" * 80)

    for system_name, results in all_results.items():
        all_probes = [p for r in results for p in r.post_retraction_results]
        skipped = sum(1 for r in results if r.skipped)

        if not all_probes:
            print(f"\n{system_name}: No probes (all skipped?)")
            continue

        overall = compute_cmlr(all_probes)

        # By probe type
        by_type = {}
        for pt in ProbeType:
            typed = [p for p in all_probes if p.probe_type == pt]
            if typed:
                by_type[pt.value] = compute_cmlr(typed)

        print(f"\n--- {system_name} ({len(results)} profiles, {skipped} skipped) ---")
        print(f"  Overall CMLR: {overall.value:.1%} (n={overall.n})")
        for pt_name, m in by_type.items():
            print(f"  {pt_name}: {m.value:.1%} (n={m.n})")

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'System':<20} {'Profiles':>8} {'Overall':>10} {'n':>6}")
    print("-" * 70)
    for system_name, results in all_results.items():
        all_probes = [p for r in results for p in r.post_retraction_results]
        overall = compute_cmlr(all_probes)
        print(f"{system_name:<20} {len(results):>8} {overall.value:>9.1%} {overall.n:>6}")
    print("-" * 70)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Max profiles to run")
    parser.add_argument("--system", choices=["oracle", "long_context", "naive", "letta", "mem0", "both", "all"],
                        default="both", help="System(s) to run. 'both'=oracle+long_context, 'all'=all 5")
    parser.add_argument("--backend", choices=["openai", "claude"], default="claude",
                        help="API backend (this entry point calls the provider APIs directly)")
    parser.add_argument("--min-recall-threshold", type=float, default=0.8,
                        help="Profile admission threshold; fact acquisition is always required")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for results, complete traces, and run manifest")
    args = parser.parse_args()

    profiles = load_usable_profiles(args.max)
    logger.info(f"Loaded {len(profiles)} usable profiles")

    if not profiles:
        logger.error("No usable profiles found!")
        return

    config = RunConfig(
        filler_turns=2,
        temporal_k=[],
        min_recall_threshold=args.min_recall_threshold,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or RESULTS_DIR / f"{stamp}-{uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(profiles, config, args.backend)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Judge — use same backend as systems
    if args.backend == "claude":
        from memleak.llm import call_claude
        async def judge_fn(prompt: str) -> str:
            return await call_claude(prompt=prompt, model="claude-sonnet-4-6", max_tokens=256)
        judge = LeakageJudge(llm_fn=judge_fn, model_name="claude-sonnet-4-6")
    else:
        from memleak.llm import call_openai
        async def judge_fn(prompt: str) -> str:
            return await call_openai(prompt=prompt, model="gpt-4o-mini", max_tokens=256)
        judge = LeakageJudge(llm_fn=judge_fn, model_name="gpt-4o-mini")

    all_results: dict[str, list[EvaluationResult]] = {}

    # Determine which systems to run
    if args.system == "all":
        run_systems = ["oracle", "long_context", "naive", "letta", "mem0"]
    elif args.system == "both":
        run_systems = ["oracle", "long_context"]
    else:
        run_systems = [args.system]

    # Set up LLM functions
    if args.backend == "claude":
        from memleak.llm import call_claude, call_claude_with_system
        llm_fn = call_claude
        llm_sys_fn = call_claude_with_system
    else:
        from memleak.llm import call_openai, call_openai_with_system
        llm_fn = call_openai
        llm_sys_fn = call_openai_with_system

    # System factories
    def make_oracle():
        return OracleExplicit(llm_fn=llm_sys_fn)

    def make_lc():
        lc = LongContextSystem(llm_fn=llm_fn, model_name=args.backend)
        lc.reset()
        return lc

    def make_naive():
        return NaiveBaseline(llm_fn=llm_sys_fn)

    def make_letta():
        return LettaSimulated(llm_fn=llm_fn)

    _mem0_counter = [0]
    def make_mem0():
        _mem0_counter[0] += 1
        return Mem0System(llm_fn=llm_fn, user_id=f"gen_{os.getpid()}_{_mem0_counter[0]}")

    system_factories = {
        "oracle": ("Oracle", make_oracle),
        "long_context": ("LongContext", make_lc),
        "naive": ("Naive", make_naive),
        "letta": ("Letta", make_letta),
        "mem0": ("Mem0", make_mem0),
    }

    for sys_key in run_systems:
        sys_name, factory = system_factories[sys_key]
        logger.info("=" * 40)
        logger.info(f"Running {sys_name}")
        logger.info("=" * 40)

        all_results[sys_name] = await run_system_on_profiles(
            sys_name, factory, profiles, config, judge, output_dir=output_dir,
        )

    print_results(all_results)

    # Save
    for system_name, results in all_results.items():
        out_path = output_dir / f"{system_name}_generated.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            [r.model_dump() for r in results], indent=2, default=str
        ))
        logger.info(f"Saved to {out_path}")

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["systems"] = list(all_results)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
