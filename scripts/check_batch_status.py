#!/usr/bin/env python3
"""Check Gemini batch job status. Exits 0 when done, 1 when still running."""
import os, sys
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from google import genai
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
job = client.batches.get(name="batches/ls7mcuqgh6yz3yz4hhviigi6gkw5xpnr9obl")

state = job.state.name if hasattr(job.state, "name") else str(job.state)
stats = ""
if hasattr(job, "batch_stats") and job.batch_stats:
    s = job.batch_stats
    stats = f" total={s.total_request_count} success={s.success_request_count} failed={s.failed_request_count}"

print(f"Batch: {state}{stats}")

if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
    sys.exit(0)
sys.exit(1)
