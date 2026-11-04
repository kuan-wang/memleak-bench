#!/usr/bin/env python3
"""Download completed Gemini batch image generation results.

Reads the batch result JSONL, decodes base64 images, and saves them
to data/images/{profile_id}/{fact_id}.png
"""

import base64, json, logging, os, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
IMAGES_DIR = BASE / "data" / "images"
BATCH_DIR = BASE / "data" / "image_batch"

# Load .env
for line in (BASE / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main():
    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    # Get batch job result file
    job_info = json.loads((BATCH_DIR / "batch_job_info.json").read_text())
    job = client.batches.get(name=job_info["job_name"])

    if not job.done:
        logger.error(f"Batch job not done yet: {job.state}")
        sys.exit(1)

    result_file = job.dest.file_name
    logger.info(f"Downloading results from {result_file}...")

    # Download the result JSONL
    raw = client.files.download(file=result_file)
    result_text = raw.decode("utf-8")

    lines = [l for l in result_text.strip().split("\n") if l.strip()]
    logger.info(f"Got {len(lines)} results")

    # Load prompts for key→profile/fact mapping
    prompts = json.loads((BATCH_DIR / "image_prompts.json").read_text())
    key_to_prompt = {}
    for p in prompts:
        key = f"{p['profile_id']}_{p['fact_id']}"
        key_to_prompt[key] = p

    saved = 0
    failed = 0

    for line in lines:
        entry = json.loads(line)
        key = entry.get("key", "")
        response = entry.get("response", {})

        if not key or key not in key_to_prompt:
            logger.warning(f"Unknown key: {key}")
            failed += 1
            continue

        prompt_info = key_to_prompt[key]
        profile_id = prompt_info["profile_id"]
        fact_id = prompt_info["fact_id"]

        # Extract image from response
        candidates = response.get("candidates", [])
        if not candidates:
            logger.warning(f"No candidates for {key}")
            failed += 1
            continue

        parts = candidates[0].get("content", {}).get("parts", [])
        image_data = None
        for part in parts:
            if "inlineData" in part:
                image_data = part["inlineData"]
                break
            elif "inline_data" in part:
                image_data = part["inline_data"]
                break

        if not image_data:
            logger.warning(f"No image data for {key}")
            failed += 1
            continue

        b64 = image_data.get("data", "")
        mime = image_data.get("mimeType", image_data.get("mime_type", "image/png"))

        if not b64:
            logger.warning(f"Empty image data for {key}")
            failed += 1
            continue

        # Determine extension
        ext = "png" if "png" in mime else "jpg"

        # Save
        out_dir = IMAGES_DIR / profile_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fact_id}.{ext}"
        out_path.write_bytes(base64.b64decode(b64))
        saved += 1

        if saved % 50 == 0:
            logger.info(f"  Saved {saved}/{len(lines)}...")

    logger.info(f"Done: {saved} saved, {failed} failed out of {len(lines)} results")

    # Save summary
    summary = {
        "total_results": len(lines),
        "saved": saved,
        "failed": failed,
        "output_dir": str(IMAGES_DIR),
    }
    (BATCH_DIR / "download_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary saved to {BATCH_DIR / 'download_summary.json'}")


if __name__ == "__main__":
    main()
