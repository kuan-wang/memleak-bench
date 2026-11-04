#!/usr/bin/env python3
"""Analyze a text-comparison run directory.

Writes <run-dir>/analysis/{summary.json, tables.md, latex_rows.tex}.

Denominators per system:
  attempted  forget-target probes of the configured probe types in the
             profile set (facts with no such probe are counted separately);
  acquired   attempted probes whose fact passed the pre-retraction acquisition
             check (3-judge majority RETAINED on the baseline answer);
  valid      acquired probes with a complete 3-judge leakage vote.
Leakage rate = leaked / valid, with a Wilson 95% CI and a percentile
profile (cluster) bootstrap 95% CI (10,000 resamples of profiles).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from memleak.schema.types import Profile, RunConfig  # noqa: E402

SYSTEM_ORDER = ["LongContext", "Naive", "Mem0", "Letta-server", "Oracle"]
JUDGES = ["gpt-4.1", "claude-sonnet-4.5", "gemini-2.5-flash"]
PROBE_TYPE_LABEL = {"text_retract_image_probe": "P1", "invisible_persistence": "P3",
                    "cross_image_reconstruction": "P4"}

PUBLISHED = {  # legacy text-only results (v1.0), n = 1,053 probes per system, one judge per probe
    "LongContext": {"label": "Long-context", "k": 179, "n": 1053, "pct": "17.0", "ci": "14.9, 19.4"},
    "Naive": {"label": "Naive", "k": 3, "n": 1053, "pct": "0.3", "ci": "0.1, 0.8"},
    "Mem0": {"label": "Mem0 (prod.)", "k": 10, "n": 1053, "pct": "0.9", "ci": "0.5, 1.6"},
    "Letta-server": {"label": "Letta (sim.)", "k": 5, "n": 1053, "pct": "0.5", "ci": "0.2, 1.1"},
    "Oracle": {"label": "Oracle", "k": 4, "n": 1053, "pct": "0.4", "ci": "0.1, 1.0"},
}
TABLE_COLUMNS = {  # Deletion / Deleted / Persists columns of the paper's system table
    "LongContext": ("Long-context", r"Behav.\ supp.\ ", "None", "All nodes"),
    "Naive": ("Naive", r"Text-delete", "Txt", r"Img, impl., deriv.\ "),
    "Mem0": ("Mem0 (prod.)", r"Sem.\ search", r"Txt, txt emb.\ ", r"Img, impl., deriv.\ "),
    "Letta-server": ("Letta (server)", r"Agent-decided", r"Core-mem.\ txt",
                     r"Msg.\ history, unarticulated"),
    "Oracle": ("Oracle", r"Prov.\ cascade", "Txt, emb., img, graph", r"Impl., deriv., behav.\ "),
}

# USD per million tokens, list prices used only for the cost estimate.
PRICES = {
    "claude-sonnet-4-5": {"in": 3.00, "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
}


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cluster_bootstrap(per_profile: list[tuple[int, int]], b: int = 10_000,
                      seed: int = 20261021) -> tuple[float, float]:
    """Percentile CI for sum(k)/sum(n), resampling profiles with replacement."""
    rows = [(k, n) for k, n in per_profile if n > 0]
    if not rows:
        return (float("nan"), float("nan"))
    ks = np.array([r[0] for r in rows], dtype=float)
    ns = np.array([r[1] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(b, len(rows)))
    rates = ks[idx].sum(axis=1) / ns[idx].sum(axis=1)
    return (float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975)))


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] / n * cb[l] / n for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(counts: list[list[int]]) -> float:
    """counts[i][j] = raters assigning item i to category j (equal raters per item)."""
    if not counts:
        return float("nan")
    m = np.array(counts, dtype=float)
    n_items, raters = m.shape[0], m[0].sum()
    p_j = m.sum(axis=0) / (n_items * raters)
    p_i = ((m * m).sum(axis=1) - raters) / (raters * (raters - 1))
    p_bar, pe = p_i.mean(), (p_j ** 2).sum()
    return 1.0 if pe == 1 else float((p_bar - pe) / (1 - pe))


def fmt_pct(x: float) -> str:
    return "nan" if x != x else f"{100 * x:.1f}"


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_run(run_dir: Path) -> tuple[dict, dict[str, Profile]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    profiles = {pid: Profile.model_validate_json(
        (run_dir / "inputs" / "profiles" / f"{pid}.json").read_text()) for pid in manifest["profiles"]}
    return manifest, profiles


def attempted_units(profile: Profile, probe_types: set[str]) -> tuple[list[tuple[str, str]], int]:
    probes, no_probe = [], 0
    for fact in profile.facts:
        if not fact.forget_target:
            continue
        typed = [p for p in fact.probes if p.probe_type.value in probe_types]
        if not typed:
            no_probe += 1
        probes += [(fact.fact_id, p.probe_type.value) for p in typed]
    return probes, no_probe


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ----------------------------------------------------------------------
# Per-system analysis
# ----------------------------------------------------------------------
def analyze_system(name: str, run_dir: Path, profiles: dict[str, Profile], probe_types: set[str]) -> dict:
    base = run_dir / "systems" / name
    out: dict = {"system": name}
    attempted = acquired = valid = leaked = 0
    facts_attempted = facts_acquired = facts_no_probe = 0
    completed, failed, skipped = [], [], []
    per_profile: list[tuple[int, int]] = []
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    votes_dist = Counter()
    judge_labels: dict[str, list[str]] = {j: [] for j in JUDGES}
    fleiss_rows: list[list[int]] = []
    acq_cross = Counter()
    acq_judge_labels: dict[str, list[str]] = {j: [] for j in JUDGES}
    acq_votes = Counter()
    empty_responses = 0
    malformed = Counter()
    utility = {"retained_present": 0, "retained_total": 0, "forgot_acquired_present": 0,
               "forgot_acquired_total": 0, "profiles": 0, "dump_invalid": 0}
    deletion = defaultdict(list)

    for pid, profile in profiles.items():
        units, no_probe = attempted_units(profile, probe_types)
        attempted += len(units)
        facts_no_probe += no_probe
        facts_attempted += len({fid for fid, _ in units})
        pdir = base / pid
        if not (pdir / "result.json").exists():
            (failed if (pdir / "error.txt").exists() or pdir.exists() else failed).append(pid)
            continue
        completed.append(pid)
        result = json.loads((pdir / "result.json").read_text())
        if result.get("skipped"):
            skipped.append(pid)
        acq_facts = {fid for fid, ok in result["baseline_results"].items() if ok}
        acq_units = [(fid, pt) for fid, pt in units if fid in acq_facts]
        acquired += len(acq_units)
        facts_acquired += len({fid for fid, _ in acq_units})

        trace = read_jsonl(pdir / "trace.jsonl")
        for row in trace:
            if row.get("phase") == "baseline" and "acquisition_details" in row:
                acq_cross[(bool(row["recalled"]), bool(row.get("substring_recalled")))] += 1
                det = row["acquisition_details"]
                if det.get("complete"):
                    acq_votes[det["n_positive"]] += 1
                    for j in JUDGES:
                        acq_judge_labels[j].append(det["votes"][j]["label"] == "retained")
            if row.get("phase") == "retraction":
                meta = row["result"].get("metadata", {})
                deletion["deleted_items"].append(len(row["result"].get("deleted_items", [])))
                if "tool_calls" in meta:
                    deletion["tool_calls"].append(len(meta["tool_calls"]))
                    deletion["tool_names"].extend(c["name"] for c in meta["tool_calls"])
                    deletion["recall_messages_retained"].append(meta.get("recall_messages_retained"))
            if row.get("phase") == "memory_state" and name == "Mem0":
                st = row["state"]
                if isinstance(st, dict) and "total_memories" in st:
                    deletion[f"mem0_memories_{row['label']}"].append(st["total_memories"])

        k_p = n_p = 0
        for j in json.loads((pdir / "judgments.json").read_text()):
            ens = j["ensemble"]
            if not j["response"].strip():
                empty_responses += 1
            for jk, v in ens["votes"].items():
                if v.get("label") is not None and not v.get("well_formed", True):
                    malformed[jk] += 1
            if not ens["complete"]:
                continue
            valid += 1
            n_p += 1
            is_leak = bool(ens["majority"])
            leaked += is_leak
            k_p += is_leak
            by_type[j["probe_type"]][0] += is_leak
            by_type[j["probe_type"]][1] += 1
            votes_dist[ens["n_positive"]] += 1
            row = [0, 0]
            for jk in JUDGES:
                lab = ens["votes"][jk]["label"]
                judge_labels[jk].append(lab)
                row[0 if lab == "leaked" else 1] += 1
            fleiss_rows.append(row)
        per_profile.append((k_p, n_p))

        dpath = pdir / "dump.json"
        if dpath.exists():
            dump = json.loads(dpath.read_text())
            present = dump["presence"]["majority_present"]
            if present is None:
                utility["dump_invalid"] += 1
            else:
                utility["profiles"] += 1
                for idx, (fid, is_target) in enumerate(zip(dump["fact_ids"], dump["forget_target"]), 1):
                    if not is_target:
                        utility["retained_total"] += 1
                        utility["retained_present"] += idx in present
                    elif fid in acq_facts:
                        utility["forgot_acquired_total"] += 1
                        utility["forgot_acquired_present"] += idx in present

    rate = leaked / valid if valid else float("nan")
    out.update({
        "profiles_total": len(profiles), "profiles_completed": len(completed),
        "profiles_failed_or_missing": sorted(failed), "profiles_skipped_zero_acquisition": sorted(skipped),
        "facts_attempted": facts_attempted, "facts_acquired": facts_acquired,
        "forget_facts_without_probe": facts_no_probe,
        "attempted": attempted, "acquired": acquired, "valid": valid, "leaked": leaked,
        "rate": rate, "wilson95": wilson(leaked, valid),
        "profile_bootstrap95": cluster_bootstrap(per_profile),
        "profiles_contributing": sum(1 for _, n in per_profile if n > 0),
        "by_probe_type": {PROBE_TYPE_LABEL.get(t, t): {"leaked": k, "valid": n, "rate": k / n if n else None,
                                                       "wilson95": wilson(k, n)}
                          for t, (k, n) in sorted(by_type.items())},
        "vote_distribution": {f"{i}/3": votes_dist.get(i, 0) for i in range(4)},
        "empty_responses": empty_responses,
        "malformed_judge_outputs": dict(malformed),
        "acquisition_semantic_vs_substring": {
            f"semantic={a},substring={s}": c for (a, s), c in sorted(acq_cross.items())},
        "acquisition_vote_distribution": {f"{i}/3": acq_votes.get(i, 0) for i in range(4)},
        "utility_dump": {
            **utility,
            "retained_recall_rate": (utility["retained_present"] / utility["retained_total"]
                                     if utility["retained_total"] else None),
            "forgotten_in_dump_rate": (utility["forgot_acquired_present"] / utility["forgot_acquired_total"]
                                       if utility["forgot_acquired_total"] else None),
        },
        "deletion_stats": {k: (Counter(v) if k == "tool_names" else
                               {"mean": float(np.mean([x for x in v if x is not None])) if v else None,
                                "min": min(v) if v else None, "max": max(v) if v else None, "n": len(v)})
                           for k, v in deletion.items()},
    })
    # Judge agreement on leakage
    per_judge = {}
    for jk in JUDGES:
        labs = judge_labels[jk]
        per_judge[jk] = {"leaked": sum(l == "leaked" for l in labs), "n": len(labs),
                         "rate": (sum(l == "leaked" for l in labs) / len(labs)) if labs else None}
    pairs = {}
    for a in range(len(JUDGES)):
        for b in range(a + 1, len(JUDGES)):
            ja, jb = JUDGES[a], JUDGES[b]
            la, lb = judge_labels[ja], judge_labels[jb]
            pairs[f"{ja} vs {jb}"] = {
                "kappa_binary": cohen_kappa([x == "leaked" for x in la], [x == "leaked" for x in lb]),
                "kappa_3class": cohen_kappa(la, lb),
                "agreement_binary": (sum((x == "leaked") == (y == "leaked") for x, y in zip(la, lb)) / len(la))
                if la else None}
    loo = {}
    for drop in JUDGES:
        keep = [j for j in JUDGES if j != drop]
        both = sum(1 for i in range(valid) if all(judge_labels[j][i] == "leaked" for j in keep))
        either = sum(1 for i in range(valid) if any(judge_labels[j][i] == "leaked" for j in keep))
        loo[f"without {drop}"] = {"both_remaining_leaked": both, "either_remaining_leaked": either,
                                  "n": valid}
    out["judges"] = {"per_judge": per_judge, "pairwise": pairs,
                     "fleiss_kappa_binary": fleiss_kappa(fleiss_rows),
                     "unanimous": votes_dist.get(0, 0) + votes_dist.get(3, 0),
                     "leave_one_out": loo}
    out["acquisition_judges"] = {
        f"{JUDGES[a]} vs {JUDGES[b]}": cohen_kappa(acq_judge_labels[JUDGES[a]], acq_judge_labels[JUDGES[b]])
        for a in range(len(JUDGES)) for b in range(a + 1, len(JUDGES))}
    return out


# ----------------------------------------------------------------------
# Usage / cost
# ----------------------------------------------------------------------
def usage_summary(run_dir: Path) -> dict:
    out: dict = defaultdict(lambda: defaultdict(float))
    for path in sorted((run_dir / "logs").glob("backbone-*.jsonl")) + sorted((run_dir / "logs").glob("mem0-internal-llm-*.jsonl")):
        for rec in read_jsonl(path):
            if rec.get("failed") or "usage" not in rec:
                continue
            key = f"{rec.get('system')}:{rec.get('kind')}"
            out[key]["calls"] += 1
            out[key]["input_tokens"] += rec["usage"].get("input_tokens") or 0
            out[key]["output_tokens"] += rec["usage"].get("output_tokens") or 0
    for path in sorted((run_dir / "systems" / "Letta-server").glob("*/letta_exchanges.jsonl")):
        for ex in read_jsonl(path):
            u = ex.get("usage") or {}
            key = "Letta-server:letta_agent"
            out[key]["requests"] += 1
            out[key]["llm_steps"] += u.get("step_count") or 0
            out[key]["input_tokens"] += u.get("prompt_tokens") or 0
            out[key]["output_tokens"] += u.get("completion_tokens") or 0
            out[key]["cached_input_tokens"] += u.get("cached_input_tokens") or 0
            out[key]["cache_write_tokens"] += u.get("cache_write_tokens") or 0
    for path in sorted((run_dir / "logs").glob("judge-*.jsonl")):
        for rec in read_jsonl(path):
            if rec.get("error"):
                key = f"judge:{rec.get('judge')}"
                out[key]["errors"] += 1
                continue
            key = f"judge:{rec.get('judge')}"
            out[key]["calls"] += 1
            out[key]["input_tokens"] += (rec.get("usage") or {}).get("input_tokens") or 0
            out[key]["output_tokens"] += (rec.get("usage") or {}).get("output_tokens") or 0
    cost = {}
    for key, u in out.items():
        if key.startswith("judge:gpt"):
            p = PRICES["gpt-4.1"]
        elif key.startswith("judge:gemini"):
            p = PRICES["gemini-2.5-flash"]
        else:
            p = PRICES["claude-sonnet-4-5"]
        cached = u.get("cached_input_tokens", 0)
        written = u.get("cache_write_tokens", 0)
        uncached = max(0.0, u["input_tokens"] - cached - written)
        cost[key] = round((uncached * p["in"] + u["output_tokens"] * p["out"]
                           + cached * p.get("cache_read", p["in"])
                           + written * p.get("cache_write", p["in"])) / 1e6, 2)
    return {"tokens": {k: dict(v) for k, v in out.items()}, "estimated_usd": cost,
            "estimated_usd_total": round(sum(cost.values()), 2), "price_assumptions_per_mtok": PRICES}


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render(summary: dict) -> tuple[str, str]:
    lines = ["| System | Profiles done | Attempted | Acquired | Valid | Leaked | Rate | Wilson 95% | Profile-bootstrap 95% |",
             "|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for name in SYSTEM_ORDER:
        s = summary["systems"].get(name)
        if not s:
            continue
        w, b = s["wilson95"], s["profile_bootstrap95"]
        lines.append(f"| {name} | {s['profiles_completed']}/{s['profiles_total']} | {s['attempted']} | "
                     f"{s['acquired']} | {s['valid']} | {s['leaked']} | {fmt_pct(s['rate'])}% | "
                     f"[{fmt_pct(w[0])}, {fmt_pct(w[1])}] | [{fmt_pct(b[0])}, {fmt_pct(b[1])}] |")
        label, deletion, deleted, persists = TABLE_COLUMNS[name]
        tex.append(f"{label} & {deletion} & {deleted} & {persists} & "
                   f"{fmt_pct(s['rate'])}\\% [\\,{fmt_pct(w[0])}, {fmt_pct(w[1])}\\,] \\\\")
    lines += ["", "| System | Legacy (n=1,053, 1 judge) | This protocol | Valid n |", "|---|---|---|---|"]
    for name in SYSTEM_ORDER:
        s = summary["systems"].get(name)
        if not s:
            continue
        p = PUBLISHED[name]
        lines.append(f"| {name} | {p['pct']}% [{p['ci']}] ({p['k']}/{p['n']}) | {fmt_pct(s['rate'])}% "
                     f"({s['leaked']}/{s['valid']}) | {s['valid']} |")
    return "\n".join(lines) + "\n", "\n".join(tex) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--systems", default=",".join(SYSTEM_ORDER))
    args = parser.parse_args()
    manifest, profiles = load_run(args.run_dir)
    probe_types = set(manifest["run_config"]["probe_types"])
    summary = {"run_dir": str(args.run_dir), "protocol": manifest["protocol"],
               "n_profiles": len(profiles), "systems": {}}
    for name in args.systems.split(","):
        if (args.run_dir / "systems" / name).exists():
            summary["systems"][name] = analyze_system(name, args.run_dir, profiles, probe_types)
    summary["usage"] = usage_summary(args.run_dir)
    out = args.run_dir / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md, tex = render(summary)
    (out / "tables.md").write_text(md)
    (out / "latex_rows.tex").write_text(tex)
    print(md)
    print(tex)
    print(json.dumps(summary["usage"]["estimated_usd"], indent=1), "total", summary["usage"]["estimated_usd_total"])


if __name__ == "__main__":
    main()
