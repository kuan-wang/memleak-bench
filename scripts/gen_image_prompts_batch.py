#!/usr/bin/env python3
"""Generate image prompts for structural floor profiles, then submit
a Gemini Nano Banana 2 batch job for image generation.

Step 1: Generate prompts via GPT-4o-mini
Step 2: Save as Gemini batch JSONL
Step 3: Submit batch job

Usage: python scripts/gen_image_prompts_batch.py
"""

from __future__ import annotations
import asyncio, json, logging, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.schema.types import Profile, VisualGrounding
from memleak.datagen.image_prompts import generate_image_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Load .env
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

GEN_DIR = Path(__file__).parent.parent / "data" / "generated"
PILOT_DIR = Path(__file__).parent.parent / "data" / "pilots"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "image_batch"


def select_profiles(n_generated: int = 33) -> list[Profile]:
    """Select profiles for structural floor: 5 pilots + top N generated."""
    results = json.loads((RESULTS_DIR / "Oracle_generated.json").read_text())
    result_ids = {r["profile_id"] for r in results}

    # Load generated profiles, ranked by testable forget-targets
    candidates = []
    for f in sorted(GEN_DIR.glob("*.json")):
        p = Profile.model_validate_json(f.read_text())
        if p.profile_id not in result_ids:
            continue
        testable = sum(1 for fact in p.facts
                       if fact.forget_target and fact.visual_grounding != VisualGrounding.ABSENT)
        candidates.append((testable, p))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in candidates[:n_generated]]

    # Add pilots
    for i in range(1, 6):
        p = Profile.model_validate_json((PILOT_DIR / f"pilot_{i:03d}.json").read_text())
        selected.append(p)

    return selected


async def generate_all_prompts(profiles: list[Profile]) -> list[dict]:
    """Generate image prompts for all non-absent facts across all profiles."""
    tasks = []
    for profile in profiles:
        for fact in profile.facts:
            if fact.visual_grounding == VisualGrounding.ABSENT:
                continue
            tasks.append({
                "profile_id": profile.profile_id,
                "fact_id": fact.fact_id,
                "fact_text": fact.text,
                "category": fact.category.value,
                "grounding": fact.visual_grounding.value,
            })

    logger.info(f"Generating prompts for {len(tasks)} facts using GPT-5.4...")

    semaphore = asyncio.Semaphore(10)
    results = []
    failed = []

    async def gen_one(task):
        async with semaphore:
            for attempt in range(3):
                prompt_data = await generate_image_prompt(
                    task["fact_text"], task["category"], task["grounding"],
                )
                if prompt_data and prompt_data.get("prompt"):
                    task["image_prompt"] = prompt_data["prompt"]
                    task["negative_prompt"] = prompt_data.get("negative", "")
                    results.append(task)
                    logger.info(f"  [{len(results)}/{len(tasks)}] {task['profile_id']}/{task['fact_id']}")
                    return
                await asyncio.sleep(1)
            failed.append(task)
            logger.warning(f"  FAILED after 3 attempts: {task['profile_id']}/{task['fact_id']}")

    await asyncio.gather(*[gen_one(t) for t in tasks])
    logger.info(f"Generated {len(results)} prompts, {len(failed)} failed")
    return results


def create_batch_jsonl(prompts: list[dict], output_path: Path):
    """Create Gemini batch JSONL for Nano Banana 2 image generation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for p in prompts:
            key = f"{p['profile_id']}_{p['fact_id']}"
            request = {
                "key": key,
                "request": {
                    "contents": [{"parts": [{"text": p["image_prompt"]}]}],
                    "generation_config": {
                        "responseModalities": ["IMAGE"],
                        "image_config": {"image_size": "1K"},
                    },
                },
            }
            f.write(json.dumps(request) + "\n")

    logger.info(f"Wrote {len(prompts)} requests to {output_path}")


def submit_batch(jsonl_path: Path) -> str:
    """Submit batch job to Gemini."""
    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    # Upload JSONL file
    logger.info("Uploading JSONL file...")
    uploaded_file = client.files.upload(file=str(jsonl_path), config={"mime_type": "application/jsonl"})
    logger.info(f"Uploaded: {uploaded_file.name}")

    # Create batch job
    logger.info("Submitting batch job...")
    batch_job = client.batches.create(
        model="gemini-3.1-flash-image-preview",
        src=uploaded_file.name,
        config={"display_name": "memleak-structural-floor-images"},
    )
    logger.info(f"Batch job created: {batch_job.name}")
    logger.info(f"State: {batch_job.state}")

    return batch_job.name


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Select profiles
    profiles = select_profiles(n_generated=33)
    total_facts = sum(1 for p in profiles for f in p.facts
                      if f.visual_grounding != VisualGrounding.ABSENT)
    total_probes = sum(1 for p in profiles for f in p.facts
                       if f.forget_target and f.visual_grounding != VisualGrounding.ABSENT)
    logger.info(f"Selected {len(profiles)} profiles: {total_facts} images, {total_probes} probes")

    # Step 2: Generate prompts
    prompts = await generate_all_prompts(profiles)
    logger.info(f"Generated {len(prompts)} image prompts")

    # Save prompts for reference
    prompts_path = OUTPUT_DIR / "image_prompts.json"
    prompts_path.write_text(json.dumps(prompts, indent=2))
    logger.info(f"Saved prompts to {prompts_path}")

    # Step 3: Create batch JSONL
    jsonl_path = OUTPUT_DIR / "batch_requests.jsonl"
    create_batch_jsonl(prompts, jsonl_path)

    # Step 4: Submit batch
    job_name = submit_batch(jsonl_path)
    logger.info(f"\nBatch job submitted: {job_name}")
    logger.info("Check status with: python -c \"from google import genai; ...")
    logger.info("Results will be available within 24 hours.")

    # Save job info
    job_info = {
        "job_name": job_name,
        "n_profiles": len(profiles),
        "n_images": len(prompts),
        "n_probes": total_probes,
        "estimated_cost": len(prompts) * 0.034,
    }
    (OUTPUT_DIR / "batch_job_info.json").write_text(json.dumps(job_info, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
