#!/usr/bin/env python3
"""Judge-robustness and channel analyses. Pure reanalysis of stored results.

1. Raw LEAKED/NOT_LEAKED/AMBIGUOUS verdict counts per mode
2. Judge robustness: per-judge rates + Claude-excluded aggregations
3. Full 2x2 text/image recovery table
4. Entailment stratification cross-tabs from existing labels

Outputs: data/results/judge_robustness_analysis.json + printed markdown summary.
"""
import json
import math
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "data/results"
JUDGES = ["gpt41", "claude_sonnet", "gemini_flash"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (100 * (center - half), 100 * (center + half))


def rate_str(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} = {100*k/n:.1f}% [{lo:.1f}, {hi:.1f}]"


def votes(rec: dict, exclude: tuple = ()) -> list[str]:
    return [
        v["verdict"]
        for k, v in rec["llm_judge"].items()
        if k in JUDGES and k not in exclude and isinstance(v, dict)
    ]


def main():
    sf = json.load(open(RESULTS / "structural_floor_scale_llm_judged.json"))
    out: dict = {}

    # ---- 1. Raw verdict counts ----
    print("## 1. Raw verdict counts (individual judge verdicts + ensemble)\n")
    out["verdict_counts"] = {}
    for mode, recs in sf.items():
        indiv = Counter()
        per_judge_ambig = Counter()
        ens = Counter()
        vote_counts = Counter()
        for r in recs:
            for jk in JUDGES:
                jv = r["llm_judge"].get(jk)
                if isinstance(jv, dict):
                    indiv[jv["verdict"]] += 1
                    if jv["verdict"] == "ambiguous":
                        per_judge_ambig[jk] += 1
            ens[r["llm_judge"].get("ensemble_verdict", "?")] += 1
            vote_counts[r["llm_judge"].get("vote_count", "?")] += 1
        out["verdict_counts"][mode] = {
            "individual": dict(indiv),
            "ambiguous_by_judge": dict(per_judge_ambig),
            "ensemble_verdict": dict(ens),
            "vote_count_dist": {str(k): v for k, v in sorted(vote_counts.items())},
            "n_probes": len(recs),
        }
        # Upper bound on underestimation: re-run majority vote with
        # every ambiguous verdict counted as leaked.
        amb_as_leaked = sum(
            1
            for r in recs
            if sum(v in ("leaked", "ambiguous") for v in votes(r)) >= 2
        )
        out["verdict_counts"][mode]["ambiguous_as_leaked_majority"] = amb_as_leaked
        total = sum(indiv.values())
        amb = indiv.get("ambiguous", 0)
        baseline = sum(1 for r in recs if r.get("leaked_llm"))
        print(f"**{mode}** (n={len(recs)} probes, {total} individual verdicts):")
        print(f"  individual: {dict(indiv)}  -> ambiguous {amb}/{total} = {100*amb/total:.1f}%")
        print(f"  ensemble_verdict: {dict(ens)}")
        print(f"  leaked-vote distribution: {dict(sorted(vote_counts.items()))}")
        print(
            f"  upper bound (all ambiguous counted as leaked): "
            f"{rate_str(amb_as_leaked, len(recs))}  vs baseline {rate_str(baseline, len(recs))}\n"
        )

    # ---- 2. Judge robustness ----
    print("## 2. Judge robustness: per-judge rates + Claude-excluded aggregations\n")
    out["judge_robustness"] = {}
    for mode, recs in sf.items():
        m: dict = {"per_judge": {}}
        print(f"**{mode}** (n={len(recs)}):")
        for jk in JUDGES:
            pairs = [
                r["llm_judge"][jk]["verdict"]
                for r in recs
                if isinstance(r["llm_judge"].get(jk), dict)
            ]
            k = sum(v == "leaked" for v in pairs)
            m["per_judge"][jk] = {"leaked": k, "n": len(pairs)}
            print(f"  {jk}: {rate_str(k, len(pairs))}")
        full = sum(1 for r in recs if r.get("leaked_llm"))
        n = len(recs)
        m["full_ensemble"] = {"leaked": full, "n": n}
        print(f"  full 3-judge majority: {rate_str(full, n)}")
        # Leave-one-out for every judge (symmetric — Claude exclusion is not cherry-picked)
        m["leave_one_out"] = {}
        for excl in JUDGES:
            unan = sum(1 for r in recs if votes(r, (excl,)).count("leaked") == 2)
            any_ = sum(1 for r in recs if votes(r, (excl,)).count("leaked") >= 1)
            m["leave_one_out"][excl] = {"unanimous": unan, "any": any_, "n": n}
            print(
                f"  without {excl}: unanimous-of-2 {rate_str(unan, n)}, "
                f"at-least-one {rate_str(any_, n)}"
            )
        out["judge_robustness"][mode] = m
        print()

    # ---- 3. 2x2 text/image recovery table ----
    print("## 3. 2x2 text/image recovery table (retained channel, n=300)\n")
    text = json.load(open(RESULTS / "retained_text_only_baseline.json"))
    text_leaked = {r["fact_id"] for r in text["results"] if r["leaked"]}
    text_ids = {r["fact_id"] for r in text["results"]}
    img_recs = sf["retained"]
    img_ids = {r["fact_id"] for r in img_recs}
    img_leaked = {r["fact_id"] for r in img_recs if r.get("leaked_llm")}
    assert text_ids == img_ids, (
        f"fact_id mismatch: text-only={len(text_ids - img_ids)}, image-only={len(img_ids - text_ids)}"
    )
    both = text_leaked & img_leaked
    img_only = img_leaked - text_leaked
    text_only = text_leaked - img_leaked
    neither = img_ids - text_leaked - img_leaked
    out["two_by_two"] = {
        "both": len(both),
        "image_only": len(img_only),
        "text_only": len(text_only),
        "neither": len(neither),
        "image_only_fact_ids": sorted(img_only),
        "both_fact_ids": sorted(both),
    }
    print(f"  joined on fact_id, n={len(img_ids)} (identical probe sets confirmed)")
    print(f"  | | image leaked | image not leaked | total |")
    print(f"  |---|---|---|---|")
    print(f"  | text leaked | {len(both)} | {len(text_only)} | {len(text_leaked)} |")
    print(f"  | text not leaked | {len(img_only)} | {len(neither)} | {300 - len(text_leaked)} |")
    print(f"  | total | {len(img_leaked)} | {300 - len(img_leaked)} | 300 |")
    print(
        f"  image-only share of image leaks: {len(img_only)}/{len(img_leaked)}"
        f" = {100*len(img_only)/len(img_leaked):.0f}%  (paper claims 17/36 = 47%)\n"
    )

    # ---- 4. Entailment cross-tabs (existing labels) ----
    print("## 4. Entailment stratification cross-tabs (36 image leaks)\n")
    ent = json.load(open(RESULTS / "entailment_stratification.json"))
    labels = {r["fact_id"]: r["label"] for r in ent["results"]}
    lab_counts = Counter(labels.values())
    by_cat = Counter((r["category"], r["label"]) for r in ent["results"])
    # cross with text-recoverability
    cross = Counter(
        (labels[fid], "text_recoverable" if fid in text_leaked else "not_text_recoverable")
        for fid in labels
    )
    out["entailment"] = {
        "label_counts": dict(lab_counts),
        "by_category": {f"{c}/{l}": v for (c, l), v in sorted(by_cat.items())},
        "label_x_text_recoverable": {f"{l}/{t}": v for (l, t), v in sorted(cross.items())},
        "unknown_fact_ids": sorted(f for f, l in labels.items() if l == "UNKNOWN"),
    }
    print(f"  labels: {dict(lab_counts)}  (UNKNOWN = classifier parse failures, need API re-run)")
    print(f"  label x text-recoverable: {dict(sorted(cross.items()))}\n")

    with open(RESULTS / "judge_robustness_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {RESULTS / 'judge_robustness_analysis.json'}")


if __name__ == "__main__":
    main()
