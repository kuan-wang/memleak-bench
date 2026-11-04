"""Auto-generate image prompts from fact text using LLM.

For generated profiles (not pilots), we don't have hand-crafted image
prompts. This module generates them from fact text + grounding type.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


PROMPT_GEN_TEMPLATE = """Generate an image generation prompt for the following fact about a person.

Fact: "{fact_text}"
Category: {category}
Visual grounding type: {grounding}

Rules based on grounding type:
- "explicit": The image should DIRECTLY show the fact (e.g., visible text labels, badges, brand names, signs)
- "implicit": The image should IMPLY the fact through visual context WITHOUT directly stating it (e.g., architectural style implies location, equipment implies profession). Do NOT include text labels or direct identifiers.
- "absent": No image needed, return null.

Output JSON only:
{{"prompt": "detailed image generation prompt, photorealistic", "negative": "what to avoid"}}

For "absent" grounding, output: {{"prompt": null, "negative": null}}"""


async def generate_image_prompt(
    fact_text: str,
    category: str,
    grounding: str,
    model: str = "gpt-5.4",
) -> dict | None:
    """Generate an image prompt for one fact."""
    if grounding == "absent":
        return None

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = PROMPT_GEN_TEMPLATE.format(
        fact_text=fact_text,
        category=category,
        grounding=grounding,
    )

    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        # GPT-5+ models use max_completion_tokens instead of max_tokens
        if model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"):
            kwargs["max_completion_tokens"] = 300
        else:
            kwargs["max_tokens"] = 300
        response = await client.chat.completions.create(**kwargs)
        data = json.loads(response.choices[0].message.content)
        if data.get("prompt"):
            return data
        return None
    except Exception as e:
        logger.error(f"Failed to generate prompt for: {fact_text[:50]}... — {e}")
        return None


async def generate_prompts_for_profile(
    profile_path: Path,
    model: str = "gpt-4o-mini",
) -> dict[str, dict]:
    """Generate image prompts for all facts with visual grounding in a profile."""
    import asyncio
    from memleak.schema.types import Profile, VisualGrounding

    profile = Profile.model_validate_json(profile_path.read_text())
    prompts: dict[str, dict] = {}

    semaphore = asyncio.Semaphore(5)

    async def gen_one(fact):
        if fact.visual_grounding == VisualGrounding.ABSENT:
            return
        async with semaphore:
            result = await generate_image_prompt(
                fact.text, fact.category.value, fact.visual_grounding.value, model
            )
            if result:
                key = f"{profile.profile_id}_{fact.fact_id}"
                prompts[key] = result

    tasks = [gen_one(f) for f in profile.facts]
    await asyncio.gather(*tasks)

    return prompts
