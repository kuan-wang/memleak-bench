#!/usr/bin/env python3
"""Source real CC-licensed images from Unsplash to replace synthetic images.

For each fact in the target profiles, generates a search query from the fact text,
downloads the top Unsplash result, and saves it to data/real_images/<profile_id>/<fact_id>.jpg.

Usage:
    python scripts/source_real_images.py
    python scripts/source_real_images.py --profiles p036,p088,p092
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
GEN_DIR = BASE / "data" / "generated"
PILOT_DIR = BASE / "data" / "pilots"
IMAGES_DIR = BASE / "data" / "images"
REAL_IMAGES_DIR = BASE / "data" / "real_images"
BATCH_DIR = BASE / "data" / "image_batch"

# Load .env
for line in (BASE / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_profile(pid: str) -> dict:
    pfile = PILOT_DIR / f"{pid}.json" if pid.startswith("pilot") else GEN_DIR / f"{pid}.json"
    return json.loads(pfile.read_text())


def fact_to_query(fact: dict) -> str:
    """Convert a fact to an Unsplash search query."""
    text = fact["text"]
    cat = fact["category"]

    # Strip common prefixes that don't help search
    for prefix in ["He ", "She ", "They ", "His ", "Her ", "Their "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    # For location facts, extract the place name
    if cat == "location":
        # Try to find proper nouns / place names
        return text

    # For possession, focus on the object
    if cat == "possession":
        for word in ["owns ", "has ", "drives ", "keeps ", "possesses ", "carries "]:
            if word in text.lower():
                idx = text.lower().index(word) + len(word)
                return text[idx:].strip(" .")
        return text

    # For profession, search for the role + workplace
    if cat == "profession":
        return text

    # For preference/relationship, use the full text
    return text


def search_unsplash(query: str, per_page: int = 5) -> list[dict]:
    """Search Unsplash for photos matching the query."""
    try:
        r = requests.get(
            "https://unsplash.com/napi/search/photos",
            params={"query": query, "per_page": per_page},
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("results", [])
        else:
            logger.warning(f"Unsplash search failed ({r.status_code}): {query}")
            return []
    except Exception as e:
        logger.warning(f"Unsplash search error: {e}")
        return []


def download_image(url: str, save_path: Path) -> bool:
    """Download an image from URL."""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(r.content)
            return True
        return False
    except Exception as e:
        logger.warning(f"Download error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=str, default=None,
                        help="Comma-separated profile IDs. Default: top 25 by image count.")
    args = parser.parse_args()

    if args.profiles:
        profile_ids = [p.strip() for p in args.profiles.split(",")]
    else:
        # Top 25 profiles by image count
        prompts = json.loads((BATCH_DIR / "image_prompts.json").read_text())
        all_pids = sorted(set(p["profile_id"] for p in prompts))

        # Count images per profile
        counts = []
        for pid in all_pids:
            pfile = PILOT_DIR / f"{pid}.json" if pid.startswith("pilot") else GEN_DIR / f"{pid}.json"
            if not pfile.exists():
                continue
            p = json.loads(pfile.read_text())
            n = 0
            for fact in p["facts"]:
                if fact.get("visual_grounding", "absent") == "absent":
                    continue
                img = IMAGES_DIR / pid / f"{fact['fact_id']}.jpg"
                if img.exists() or img.with_suffix(".png").exists():
                    n += 1
            counts.append((pid, n))
        counts.sort(key=lambda x: x[1], reverse=True)
        profile_ids = [pid for pid, _ in counts[:25]]

    logger.info(f"Processing {len(profile_ids)} profiles")

    stats = {"total": 0, "found": 0, "failed": 0, "skipped": 0}
    profile_stats = {}

    for pid in profile_ids:
        p = load_profile(pid)
        p_found = 0
        p_total = 0

        for fact in p["facts"]:
            vg = fact.get("visual_grounding", "absent")
            if vg == "absent":
                continue

            # Check if synthetic image exists
            fid = fact["fact_id"]
            syn_img = IMAGES_DIR / pid / f"{fid}.jpg"
            if not (syn_img.exists() or syn_img.with_suffix(".png").exists()):
                continue

            save_path = REAL_IMAGES_DIR / pid / f"{fid}.jpg"
            if save_path.exists():
                stats["skipped"] += 1
                p_found += 1
                p_total += 1
                continue

            stats["total"] += 1
            p_total += 1

            query = fact_to_query(fact)
            results = search_unsplash(query)

            if not results:
                # Try a simpler query
                words = query.split()
                if len(words) > 4:
                    results = search_unsplash(" ".join(words[:4]))

            if results:
                # Download the first result (regular size, ~1080px)
                url = results[0].get("urls", {}).get("regular", "")
                if url and download_image(url, save_path):
                    desc = results[0].get("alt_description", "")[:60]
                    logger.info(f"  {pid}/{fid}: OK ({desc})")
                    stats["found"] += 1
                    p_found += 1
                else:
                    logger.warning(f"  {pid}/{fid}: download failed")
                    stats["failed"] += 1
            else:
                logger.warning(f"  {pid}/{fid}: no results for '{query[:50]}'")
                stats["failed"] += 1

            # Rate limit: be polite to Unsplash
            time.sleep(1.0)

        profile_stats[pid] = {"found": p_found, "total": p_total}
        rate = p_found / p_total * 100 if p_total > 0 else 0
        logger.info(f"  {pid}: {p_found}/{p_total} ({rate:.0f}%)")

    # Summary
    print(f"\n{'='*60}")
    print(f"REAL IMAGE SOURCING SUMMARY")
    print(f"{'='*60}")
    print(f"Profiles: {len(profile_ids)}")
    print(f"Images searched: {stats['total']}")
    print(f"Found: {stats['found']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped (already exists): {stats['skipped']}")
    print(f"Success rate: {stats['found']/(stats['total'] or 1)*100:.1f}%")

    # Per-profile
    print(f"\nPer-profile (>60% = usable):")
    usable = 0
    for pid in profile_ids:
        s = profile_stats.get(pid, {"found": 0, "total": 0})
        rate = s["found"] / s["total"] * 100 if s["total"] > 0 else 0
        marker = "OK" if rate >= 60 else "SKIP"
        if rate >= 60:
            usable += 1
        print(f"  {pid}: {s['found']}/{s['total']} ({rate:.0f}%) [{marker}]")
    print(f"\nUsable profiles (>=60%): {usable}/{len(profile_ids)}")


if __name__ == "__main__":
    main()
