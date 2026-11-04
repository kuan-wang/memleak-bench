#!/usr/bin/env python3
"""Generate images for all 5 pilot profiles with memleak.datagen.images.generate_image.

Replaces PLACEHOLDER: paths with actual generated images. The released pilot images were
rendered with a Qwen-Image endpoint that is not publicly available; in this release
generate_image uses the OpenAI backend, so re-generated images will differ.

Usage:
    python scripts/generate_pilot_images.py
    python scripts/generate_pilot_images.py --pilots 1  # specific pilot
    python scripts/generate_pilot_images.py --dry-run    # show prompts only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.datagen.images import generate_image, PILOT_IMAGE_PROMPTS
from memleak.schema.types import Profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PILOT_DIR = Path(__file__).parent.parent / "data" / "pilots"
IMAGE_DIR = Path(__file__).parent.parent / "data" / "images"


def generate_pilot_images(pilot_num: int, dry_run: bool = False) -> dict[str, str]:
    """Generate all images for one pilot profile.

    Returns: dict mapping PLACEHOLDER path -> real image path.
    """
    profile_path = PILOT_DIR / f"pilot_{pilot_num:03d}.json"
    profile = Profile.model_validate_json(profile_path.read_text())
    out_dir = IMAGE_DIR / profile.profile_id

    # Collect all PLACEHOLDER paths from the profile
    placeholders = set()
    for fact in profile.facts:
        for img in fact.images:
            if img.path.startswith("PLACEHOLDER:"):
                placeholders.add(img.path)
        for probe in fact.probes:
            for qimg in probe.query_images:
                if qimg.startswith("PLACEHOLDER:"):
                    placeholders.add(qimg)

    # Also check interaction stream
    for turn in profile.interaction_stream:
        for img in turn.images:
            if img.startswith("PLACEHOLDER:"):
                placeholders.add(img)

    logger.info(f"Pilot {pilot_num}: {len(placeholders)} unique images to generate")

    path_map: dict[str, str] = {}
    generated = 0
    skipped = 0

    for placeholder in sorted(placeholders):
        if placeholder not in PILOT_IMAGE_PROMPTS:
            logger.warning(f"  No prompt for {placeholder} — skipping")
            skipped += 1
            continue

        prompt_info = PILOT_IMAGE_PROMPTS[placeholder]
        # Filename from placeholder
        name = placeholder.replace("PLACEHOLDER:", "").replace("/", "_")
        real_path = out_dir / f"{name}.jpg"

        if real_path.exists():
            logger.info(f"  Already exists: {real_path}")
            path_map[placeholder] = str(real_path)
            continue

        if dry_run:
            logger.info(f"  [DRY RUN] Would generate: {name}")
            logger.info(f"    Prompt: {prompt_info['prompt'][:80]}...")
            path_map[placeholder] = str(real_path)
            continue

        try:
            generate_image(
                prompt=prompt_info["prompt"],
                output_path=real_path,
                seed=42 + hash(placeholder) % 10000,
                negative_prompt=prompt_info.get("negative", ""),
            )
            path_map[placeholder] = str(real_path)
            generated += 1
            time.sleep(0.5)  # gentle rate limiting
        except Exception as e:
            logger.error(f"  Failed to generate {name}: {e}")
            skipped += 1

    logger.info(f"Pilot {pilot_num}: generated={generated}, skipped={skipped}, total={len(placeholders)}")
    return path_map


def update_profile_paths(pilot_num: int, path_map: dict[str, str]) -> None:
    """Update a pilot profile JSON, replacing PLACEHOLDER paths with real paths."""
    profile_path = PILOT_DIR / f"pilot_{pilot_num:03d}.json"
    profile = Profile.model_validate_json(profile_path.read_text())

    changes = 0
    for fact in profile.facts:
        for img in fact.images:
            if img.path in path_map:
                img.path = path_map[img.path]
                changes += 1
        for probe in fact.probes:
            probe.query_images = [
                path_map.get(p, p) for p in probe.query_images
            ]

    for turn in profile.interaction_stream:
        turn.images = [path_map.get(p, p) for p in turn.images]

    profile_path.write_text(profile.model_dump_json(indent=2))
    logger.info(f"Updated pilot_{pilot_num:03d}.json with {changes} image paths")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilots", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-update", action="store_true", help="Generate images but don't update profile JSON")
    args = parser.parse_args()

    total_generated = 0
    for n in args.pilots:
        logger.info(f"\n{'='*40}")
        logger.info(f"Pilot {n}")
        logger.info(f"{'='*40}")
        path_map = generate_pilot_images(n, dry_run=args.dry_run)

        if not args.dry_run and not args.no_update and path_map:
            update_profile_paths(n, path_map)

        total_generated += len([v for v in path_map.values() if Path(v).exists()])

    logger.info(f"\nDone. Total images: {total_generated}")


if __name__ == "__main__":
    main()
