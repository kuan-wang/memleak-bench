#!/usr/bin/env python3
"""Regenerate the 11 malformed profiles with retry until validation passes.

Usage: python scripts/regen_bad_profiles.py
"""

from __future__ import annotations
import asyncio, json, logging, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.datagen.profiles import generate_profile, PROFILE_THEMES
from memleak.schema.validators import validate_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "generated"

# The 11 profile IDs that need regeneration
BAD_IDS = [6, 35, 42, 63, 69, 70, 72, 77, 101, 113, 125]

MAX_RETRIES = 5


async def main():
    # Reproduce the same theme shuffle used in generation
    themes = PROFILE_THEMES.copy()
    random.seed(42)
    random.shuffle(themes)

    start_idx = 6  # same as original generation

    regenerated = 0
    failed = []

    for idx in BAD_IDS:
        i = idx - start_idx
        prefix = f"p{idx:03d}"
        theme = themes[i % len(themes)]
        logger.info(f"Regenerating {prefix}: {theme[:60]}...")

        for attempt in range(1, MAX_RETRIES + 1):
            profile = await generate_profile(theme, prefix, model="gpt-4o-mini")
            if profile is None:
                logger.warning(f"  {prefix} attempt {attempt}: generation failed")
                continue

            errors = validate_profile(profile, strict=False)
            critical = [e for e in errors if "Expected 20" in e or "Expected 10 forget" in e
                         or "expected 2 forget targets" in e]

            if len(profile.facts) == 20 and not critical:
                path = OUTPUT_DIR / f"{prefix}.json"
                path.write_text(profile.model_dump_json(indent=2))
                logger.info(f"  {prefix} attempt {attempt}: PASS ({len(errors)} minor warnings)")
                regenerated += 1
                break
            else:
                logger.warning(f"  {prefix} attempt {attempt}: FAIL "
                               f"(facts={len(profile.facts)}, critical={len(critical)})")
        else:
            logger.error(f"  {prefix}: FAILED after {MAX_RETRIES} attempts")
            failed.append(prefix)

    logger.info(f"\nDone: {regenerated} regenerated, {len(failed)} failed")
    if failed:
        logger.error(f"Failed profiles: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
