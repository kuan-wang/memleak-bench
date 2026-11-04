#!/usr/bin/env python3
"""Summarize all generated-profile results for paper update.

Usage: python scripts/summarize_generated.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memleak.schema.types import ProbeType, LeakageVerdict
from memleak.metrics.cmlr import compute_cmlr
from memleak.metrics.stats import bootstrap_ci

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

SYSTEM_FILES = {
    "Oracle": "Oracle_generated.json",
    "LongContext": "LongContext_generated.json",
    "Naive": "Naive_generated.json",
    "Letta": "Letta_generated.json",
    "Mem0": "Mem0_generated.json",
}


def load_probes(filepath):
    """Load probes from a generated results file."""
    if not filepath.exists():
        return None, 0
    data = json.loads(filepath.read_text())
    profiles = len(data)
    probes = []
    for r in data:
        for p in r.get("post_retraction_results", []):
            # Reconstruct a minimal ProbeResult-like dict
            probes.append(p)
    return probes, profiles


def compute_stats(probes):
    """Compute CMLR overall + by probe type with bootstrap CI."""
    if not probes:
        return {}

    # Overall
    leaked = [1 if p.get("ensemble_verdict") == "leaked" else 0 for p in probes]
    overall = sum(leaked) / len(leaked) if leaked else 0
    ci = bootstrap_ci(leaked) if len(leaked) >= 10 else (0, 0)

    # By probe type
    by_type = {}
    for pt in ["text_retract_image_probe", "invisible_persistence", "cross_image_reconstruction"]:
        typed = [p for p in probes if p.get("probe_type") == pt]
        if typed:
            t_leaked = [1 if p.get("ensemble_verdict") == "leaked" else 0 for p in typed]
            t_cmlr = sum(t_leaked) / len(t_leaked)
            t_ci = bootstrap_ci(t_leaked) if len(t_leaked) >= 10 else (0, 0)
            by_type[pt] = {"cmlr": t_cmlr, "n": len(typed), "ci": t_ci}

    return {
        "overall": overall,
        "n": len(probes),
        "ci": ci,
        "by_type": by_type,
    }


def main():
    print("=" * 80)
    print("MEMLEAK — SCALED RESULTS SUMMARY (Generated Profiles)")
    print("=" * 80)

    header = f"{'System':<15} {'n':>6} {'Overall':>10} {'95% CI':>16} {'P1':>10} {'P3':>10}"
    print(header)
    print("-" * len(header))

    for sys_name, filename in SYSTEM_FILES.items():
        filepath = RESULTS_DIR / filename
        probes, n_profiles = load_probes(filepath)

        if probes is None:
            print(f"{sys_name:<15} {'(missing)':>6}")
            continue

        stats = compute_stats(probes)
        if not stats:
            print(f"{sys_name:<15} {'(empty)':>6}")
            continue

        p1 = stats["by_type"].get("text_retract_image_probe", {})
        p3 = stats["by_type"].get("invisible_persistence", {})

        ci_str = f"[{stats['ci'][0]:.1%}, {stats['ci'][1]:.1%}]"
        p1_str = f"{p1['cmlr']:.1%}" if p1 else "N/A"
        p3_str = f"{p3['cmlr']:.1%}" if p3 else "N/A"

        print(f"{sys_name:<15} {stats['n']:>6} {stats['overall']:>9.1%} {ci_str:>16} {p1_str:>10} {p3_str:>10}")

    print("-" * len(header))

    # LaTeX table
    print("\n--- LaTeX table rows ---")
    for sys_name, filename in SYSTEM_FILES.items():
        filepath = RESULTS_DIR / filename
        probes, n_profiles = load_probes(filepath)
        if probes is None:
            continue
        stats = compute_stats(probes)
        if not stats:
            continue

        p1 = stats["by_type"].get("text_retract_image_probe", {})
        p3 = stats["by_type"].get("invisible_persistence", {})
        p1_str = f"{p1['cmlr']:.1%}" if p1 else "---"
        p3_str = f"{p3['cmlr']:.1%}" if p3 else "---"

        print(f"{sys_name} & Claude Sonnet & {stats['overall']:.1%} & {p1_str} & {p3_str} & {stats['n']} \\\\")


if __name__ == "__main__":
    main()
