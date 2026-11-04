#!/usr/bin/env python3
"""Generate a CSV for human validation of the LLM judge ensemble.

Samples 100 probes stratified by mode (retained focus) and verdict,
with enough leaked cases to compute meaningful agreement stats.

Output columns:
  id, mode, profile_id, fact_id, category, fact_text,
  vlm_response, ensemble_verdict, vote_count, human_verdict (blank)
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
RESULTS = BASE / "data" / "results" / "structural_floor_scale_llm_judged.json"
OUT = BASE / "data" / "results" / "human_validation_sample.csv"

random.seed(42)


def main():
    data = json.loads(RESULTS.read_text())

    # Collect all probes with full judge verdicts
    all_probes = []
    for mode in ["same_fact", "retained", "negative"]:
        for r in data[mode]:
            j = r.get("llm_judge", {})
            if "error" in str(j):
                continue
            all_probes.append({
                "mode": mode,
                "profile_id": r["profile_id"],
                "fact_id": r["fact_id"],
                "category": r["category"],
                "fact_text": r["fact_text"],
                "vlm_response": r["vlm_response"],
                "ensemble_verdict": j.get("ensemble_verdict", "unknown"),
                "vote_count": j.get("vote_count", ""),
                "leaked_llm": r.get("leaked_llm", False),
            })

    # Stratified sampling:
    # - 60 retained (oversample leaked: ~20 leaked + 40 safe)
    # - 25 same_fact (mix of leaked/safe)
    # - 15 negative (mostly safe, sanity check)
    retained = [p for p in all_probes if p["mode"] == "retained"]
    same_fact = [p for p in all_probes if p["mode"] == "same_fact"]
    negative = [p for p in all_probes if p["mode"] == "negative"]

    retained_leaked = [p for p in retained if p["leaked_llm"]]
    retained_safe = [p for p in retained if not p["leaked_llm"]]
    same_fact_leaked = [p for p in same_fact if p["leaked_llm"]]
    same_fact_safe = [p for p in same_fact if not p["leaked_llm"]]

    sample = []

    # Retained: all leaked (up to 45) + random safe to reach 60
    n_ret_leaked = min(len(retained_leaked), 30)
    sample.extend(random.sample(retained_leaked, n_ret_leaked))
    n_ret_safe = 60 - n_ret_leaked
    sample.extend(random.sample(retained_safe, min(n_ret_safe, len(retained_safe))))

    # Same_fact: 15 leaked + 10 safe
    sample.extend(random.sample(same_fact_leaked, min(15, len(same_fact_leaked))))
    sample.extend(random.sample(same_fact_safe, min(10, len(same_fact_safe))))

    # Negative: 15 random
    sample.extend(random.sample(negative, min(15, len(negative))))

    random.shuffle(sample)

    # Write CSV
    fields = [
        "id", "mode", "profile_id", "fact_id", "category", "fact_text",
        "vlm_response", "ensemble_verdict", "vote_count", "human_verdict",
    ]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, p in enumerate(sample, 1):
            w.writerow({
                "id": i,
                "mode": p["mode"],
                "profile_id": p["profile_id"],
                "fact_id": p["fact_id"],
                "category": p["category"],
                "fact_text": p["fact_text"],
                "vlm_response": p["vlm_response"][:1000],  # truncate long responses
                "ensemble_verdict": p["ensemble_verdict"],
                "vote_count": p["vote_count"],
                "human_verdict": "",
            })

    print(f"Wrote {len(sample)} probes to {OUT}")

    # Summary
    from collections import Counter
    modes = Counter(p["mode"] for p in sample)
    verdicts = Counter(p["ensemble_verdict"] for p in sample)
    print(f"  By mode: {dict(modes)}")
    print(f"  By verdict: {dict(verdicts)}")


if __name__ == "__main__":
    main()
