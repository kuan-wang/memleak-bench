#!/usr/bin/env python3
"""Deletion-policy ablations on pilot profiles.

Tests different deletion scopes:
1. Delete nothing (LongContext) — already have
2. Delete text only (Naive) — already have
3. Delete text + tagged images (new)
4. Delete text + all same-profile images (new)
5. Delete text + images + embeddings (Oracle) — already have

Reports CMLR for each policy to show how deletion scope affects leakage.
"""

import asyncio, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.schema.types import Profile, RunConfig, ProbeType, LeakageVerdict
from memleak.harness.oracle import OracleExplicit, OracleMemoryStore
from memleak.harness.base import ForgetResult
from memleak.harness.runner import run_evaluation
from memleak.judge.leakage_judge import LeakageJudge
from memleak.metrics.cmlr import compute_cmlr
from memleak.llm import call_claude, call_claude_with_system

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PILOT_DIR = Path(__file__).parent.parent / "data" / "pilots"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"


class AblationTaggedImages(OracleExplicit):
    """Delete text + tagged images only (no embeddings, no graph)."""
    async def forget(self, instruction: str) -> ForgetResult:
        if not self.profile or not self.profile.retraction_point:
            return ForgetResult(success=False, method="ablation_tagged_images")
        deleted = []
        for fid in self.profile.retraction_point.target_facts:
            if fid in self.store.text_entries:
                deleted.append(f"text:{fid}")
                del self.store.text_entries[fid]
            if fid in self.store.images:
                for img in self.store.images[fid]:
                    deleted.append(f"image:{img}")
                del self.store.images[fid]
            self.store.deleted_facts.add(fid)
        return ForgetResult(success=True, deleted_items=deleted, method="ablation_tagged_images")


class AblationAllImages(OracleExplicit):
    """Delete text + ALL images in the profile (aggressive)."""
    async def forget(self, instruction: str) -> ForgetResult:
        if not self.profile or not self.profile.retraction_point:
            return ForgetResult(success=False, method="ablation_all_images")
        deleted = []
        for fid in self.profile.retraction_point.target_facts:
            if fid in self.store.text_entries:
                deleted.append(f"text:{fid}")
                del self.store.text_entries[fid]
            self.store.deleted_facts.add(fid)
        # Delete ALL images (aggressive policy)
        for fid in list(self.store.images.keys()):
            for img in self.store.images[fid]:
                deleted.append(f"image:{img}")
            del self.store.images[fid]
        return ForgetResult(success=True, deleted_items=deleted, method="ablation_all_images")


async def main():
    profiles = [Profile.model_validate_json((PILOT_DIR / f"pilot_{i:03d}.json").read_text()) for i in range(1, 6)]
    config = RunConfig(filler_turns=2, temporal_k=[], min_recall_threshold=0.0)

    async def judge_fn(prompt):
        return await call_claude(prompt=prompt, max_tokens=256)
    judge = LeakageJudge(llm_fn=judge_fn, model_name="claude-sonnet-4")

    ablations = {
        "text+tagged_images": AblationTaggedImages,
        "text+ALL_images": AblationAllImages,
    }

    results = {}
    for abl_name, SystemClass in ablations.items():
        logger.info(f"\n{'='*40}\nAblation: {abl_name}\n{'='*40}")
        all_probes = []

        for profile in profiles:
            system = SystemClass(llm_fn=call_claude_with_system)
            system.reset()
            system.load_profile(profile)

            result = await run_evaluation(system, profile, config)

            fact_map = {f.fact_id: f for f in profile.facts}
            for pr in result.post_retraction_results:
                fact = fact_map.get(pr.fact_id)
                if fact:
                    matching = next((p for p in fact.probes if p.probe_type == pr.probe_type), None)
                    if matching:
                        jr = await judge.judge(fact.text, matching.query, pr.agent_response, pr.fact_id)
                        pr.judge_results = [jr]
                        pr.ensemble_verdict = jr.verdict
            all_probes.extend(result.post_retraction_results)

        cmlr = compute_cmlr(all_probes)
        results[abl_name] = {"cmlr": cmlr.value, "n": cmlr.n}
        logger.info(f"  {abl_name}: CMLR = {cmlr.value:.1%} (n={cmlr.n})")

    # Add existing results for comparison
    results["delete_nothing (LongContext)"] = {"cmlr": 0.636, "n": 55}
    results["text_only (Naive)"] = {"cmlr": 0.055, "n": 55}
    results["text+images+embeddings (Oracle)"] = {"cmlr": 0.055, "n": 55}

    print(f"\n{'='*60}")
    print("DELETION-POLICY ABLATION RESULTS")
    print(f"{'='*60}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["cmlr"], reverse=True):
        print(f"  {name:<40} CMLR={r['cmlr']:.1%}  n={r['n']}")

    Path(RESULTS_DIR / "ablations_full.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
