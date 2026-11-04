#!/usr/bin/env python3
"""Generate profiles at scale.

Usage:
    python scripts/generate_profiles.py --count 10   # small test batch
    python scripts/generate_profiles.py --count 100  # full dataset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.datagen.profiles import generate_profiles
from memleak.schema.validators import validate_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "generated"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-idx", type=int, default=6)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    profiles = await generate_profiles(
        count=args.count,
        start_idx=args.start_idx,
        model=args.model,
    )

    valid = 0
    invalid = 0
    for profile in profiles:
        errors = validate_profile(profile, strict=False)
        path = OUTPUT_DIR / f"{profile.profile_id}.json"
        path.write_text(profile.model_dump_json(indent=2))

        if errors:
            logger.warning(f"{profile.profile_id}: {len(errors)} validation warnings")
            for e in errors[:3]:
                logger.warning(f"  - {e}")
            invalid += 1
        else:
            valid += 1

    logger.info(f"\nDone: {valid} valid, {invalid} with warnings, {len(profiles)} total")
    logger.info(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
